import warnings, torch, os, math, numpy
from sklearn.metrics import accuracy_score
from sklearn import metrics
from operator import itemgetter

def print_write(type, text, score_file): # A helper function to print the text and write the log
	if type == 'T': # Classification training without LGL (Baseline)
		epoch, loss, acc, nselects = text
		print("%d epoch, LOSS %f, ACC %.2f%%, nselects %.2f%%\n"%(epoch, loss, acc, nselects))
		score_file.write("[T], %d epoch, LOSS %f, ACC %.2f%%, nselects %.2f%%\n"%(epoch, loss, acc, nselects))	
	elif type == 'L': # Classification training with LGL (Propose)
		epoch, loss, acc, nselects, gate = text
		print("%d epoch, LOSS %f, ACC %.2f%%, nselects %.2f%%, Gate %.1f \n"%(epoch, loss, acc, nselects, gate))
		score_file.write("[L], %d epoch, LOSS %f, ACC %.2f%%, nselects %.2f%%, Gate %.1f \n"%(epoch, loss, acc, nselects, gate))	
	elif type == 'C': # Clustering step
		epoch, NMI = text
		print("%d epoch, NMI %.2f\n"%(epoch, NMI))
		score_file.write("[C], %d epoch, NMI %.2f\n"%(epoch, NMI))
	elif type == 'E': # Evaluation step
		epoch, EER, minDCF = text
		print("EER %2.2f%%, minDCF %2.3f%%\n"%(EER, minDCF))
		score_file.write("[E], %d epoch, EER %2.2f%%, minDCF %2.3f%%\n"%(epoch, EER, minDCF))
	score_file.flush()

def check_clustering(score_path, LGL): # Read the score.txt file, judge the next stage
	lines = open(score_path).read().splitlines()

	if LGL == True: # For LGL, the order is 
		# Iteration 1: (C-T-T...-T-L-L...-L-) 
		# Iteration 2: (C-T-T...-T-L-L...-L-) 
		# ...
		EERs_T, epochs_T, EERs_L, epochs_L = [], [], [], []
		iteration = 0
		train_type = 'T'
		for line in lines:
			if line.split(',')[0] == '[C]': # Clear all results after clustering
				EERs_T, EERs_L, epochs_T, epochs_L = [], [], [], []
				train_type = 'T'
				iteration += 1
			elif line.split(',')[0] == '[E]': # Save the evaluation result in this iteration
				epoch = int(line.split(',')[1].split()[0])
				EER = float(line.split(',')[-2].split()[-1][:-1])
				if train_type == 'T':
					epochs_T.append(epoch) 
					EERs_T.append(EER) # Result in [T]
				elif train_type == 'L':
					epochs_L.append(epoch)
					EERs_L.append(EER) # Result in [L]
			elif line.split(',')[0] == '[T]': # If the stage is [T], record it
				train_type = 'T'
			elif line.split(',')[0] == '[L]': # If the stage is [L], record it
				train_type = 'L'

		if train_type == 'T': # The stage is [T], so need to judge the next step is keeping [T]? Or do LGL for [L] ?
			if len(EERs_T) < 4: # Too short training epoch, keep training
				return 'T', None, None, iteration
			else:
				if EERs_T[-1] > min(EERs_T) and EERs_T[-2] > min(EERs_T) and EERs_T[-3] > min(EERs_T): # Get the best training result already, go LGL
					best_epoch = epochs_T[EERs_T.index(min(EERs_T))]
					next_epoch = epochs_T[-1]
					return 'L', best_epoch, next_epoch, iteration
				else:
					return 'T', None, None, iteration # EER can still drop, keep training 

		elif train_type == 'L':
			if len(EERs_L) < 4: # Too short training epoch, keep LGL training
				return 'L', None, None, iteration
			else:
				if EERs_L[-1] > min(EERs_L) and EERs_L[-2] > min(EERs_L) and EERs_L[-3] > min(EERs_L): # Get the best LGL result already, go clustering
					best_epoch = epochs_L[EERs_L.index(min(EERs_L))]
					next_epoch = epochs_L[-1]
					return 'C', best_epoch, next_epoch, iteration # Clustering based on the best epoch is more robust
				else:
					return 'L', None, None, iteration # EER can still drop, keep training 

	else: # Baseline approach without LGL
		EERs_T, epochs_T = [], []
		iteration = 0
		for line in lines:
			if line.split(',')[0] == '[C]': # Clear all results after clustering
				EERs_T, epochs_T = [], []
				iteration += 1
			elif line.split(',')[0] == '[E]': # Save the evaluation result
				epoch = int(line.split(',')[1].split()[0])
				EER = float(line.split(',')[-2].split()[-1][:-1])
				epochs_T.append(epoch)
				EERs_T.append(EER)

		if len(EERs_T) < 4: # Too short training epoch, keep training
			return 'T', None, None, iteration
		else:
			if EERs_T[-1] > min(EERs_T) and EERs_T[-2] > min(EERs_T) and EERs_T[-3] > min(EERs_T): # Get the best training result, go clustering
				best_epoch = epochs_T[EERs_T.index(min(EERs_T))]
				next_epoch = epochs_T[-1]
				return 'C', best_epoch, next_epoch, iteration
			else:
				return 'T', None, None, iteration # EER can still drop, keep training 

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
	
	fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
	fnr = 1 - tpr
	tunedThreshold = [];
	if target_fr:
		for tfr in target_fr:
			idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
			tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	for tfa in target_fa:
		idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
		tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
	eer  = max(fpr[idxE],fnr[idxE])*100
	
	return tunedThreshold, eer, fpr, fnr

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

	  # Sort the scores from smallest to largest, and also get the corresponding
	  # indexes of the sorted scores.  We will treat the sorted scores as the
	  # thresholds at which the the error-rates are evaluated.
	  sorted_indexes, thresholds = zip(*sorted(
		  [(index, threshold) for index, threshold in enumerate(scores)],
		  key=itemgetter(1)))
	  sorted_labels = []
	  labels = [labels[i] for i in sorted_indexes]
	  fnrs = []
	  fprs = []

	  # At the end of this loop, fnrs[i] is the number of errors made by
	  # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
	  # is the total number of times that we have correctly accepted scores
	  # greater than thresholds[i].
	  for i in range(0, len(labels)):
		  if i == 0:
			  fnrs.append(labels[i])
			  fprs.append(1 - labels[i])
		  else:
			  fnrs.append(fnrs[i-1] + labels[i])
			  fprs.append(fprs[i-1] + 1 - labels[i])
	  fnrs_norm = sum(labels)
	  fprs_norm = len(labels) - fnrs_norm

	  # Now divide by the total number of false negative errors to
	  # obtain the false positive rates across all thresholds
	  fnrs = [x / float(fnrs_norm) for x in fnrs]

	  # Divide by the total number of corret positives to get the
	  # true positive rate.  Subtract these quantities from 1 to
	  # get the false positive rates.
	  fprs = [1 - x / float(fprs_norm) for x in fprs]
	  return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
	min_c_det = float("inf")
	min_c_det_threshold = thresholds[0]
	for i in range(0, len(fnrs)):
		# See Equation (2).  it is a weighted sum of false negative
		# and false positive errors.
		c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
		if c_det < min_c_det:
			min_c_det = c_det
			min_c_det_threshold = thresholds[i]
	# See Equations (3) and (4).  Now we normalize the cost.
	c_def = min(c_miss * p_target, c_fa * (1 - p_target))
	min_dcf = min_c_det / c_def
	return min_dcf, min_c_det_threshold

def accuracy(output, target, topk=(1,)):

	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	
	return res
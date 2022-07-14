import torch, sys, os, tqdm, numpy, time, faiss, gc, soundfile
import torch.nn as nn
import torch.nn.functional as F

from tools import *
from loss import *
from encoder import *
from collections import defaultdict
from sklearn.metrics.cluster import normalized_mutual_info_score

class trainer(nn.Module):
	def __init__(self, lr, n_cluster, **kwargs):
		super(trainer, self).__init__()
		self.Network = ECAPA_TDNN(C = 512).cuda() # Speaker encoder 
		self.Loss = LossFunction(n_class = n_cluster).cuda() # Classification layer
		self.Optim = torch.optim.Adam(list(self.Network.parameters()) + list(self.Loss.parameters()), lr = lr) # Adam, learning rate is fixed

	def train_network(self, epoch, loader, gate):
		self.train()
		loss, index, nselects, top1 = 0, 0, 0, 0
		time_start = time.time()
		for num, (data, label) in enumerate(loader, start = 1):
			self.zero_grad()
			out = self.Network.forward(data.cuda()) # input segment and the output speaker embedding
			nloss, prec1, nselect = self.Loss.forward(out, label.cuda(), gate) # Get the loss, training acc and the number of selected data
			nloss.backward()
			self.Optim.step()
			loss += nloss.detach().cpu().numpy()
			index += len(label)
			nselects += nselect
			top1 += prec1
			time_used = time.time() - time_start
			sys.stderr.write(" [%2d] %.2f%% (est %.1f mins), Loss: %.3f, ACC: %.2f%%, select: %.2f%%, gate: %.1f\r" %\
			(epoch, 100 * (num / loader.__len__()), time_used * loader.__len__() / num / 60, \
			loss/num, top1/nselects, nselects/index*100, gate))
			sys.stderr.flush()
		sys.stdout.write("\n")
		torch.cuda.empty_cache()
		return loss / num, top1/nselects, nselects/index*100

	def cluster_network(self, loader, n_cluster):
		self.eval()
		out_all, filenames_all, labels_all = [], [], []
		for data, filenames, labels in tqdm.tqdm(loader):  
			with torch.no_grad():				
				out = self.Network.forward(data[0].cuda()) # Get the embeddings
				out = F.normalize(out, p=2, dim=1) # Normalization
				for i in range(len(filenames)): # Save the filname, labels, and the embedding into the list [labels is used to compute NMI]
					filenames_all.append(filenames[i][0])
					labels_all.append(labels[i].cpu().numpy()[0])
					out_all.append(out[i].detach().cpu().numpy())
		out_all = numpy.array(out_all)
		# Clustering using faiss https://github.com/facebookresearch/deepcluster
		clus = faiss.Clustering(out_all.shape[1], n_cluster)
		n, d = out_all.shape
		flat_config = faiss.GpuIndexFlatConfig()
		flat_config.useFloat16 = False
		flat_config.device = 0
		index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), d, flat_config)
		clus.train(out_all, index) # Clustering
		preds = [int(i[0]) for i in index.search(out_all, 1)[1]] # Get the results 
		del out_all
		gc.collect()
		dic_label = defaultdict(list) # Pseudo label dict

		for i in range(len(preds)):
			pred_label = preds[i] # pseudo label
			filename = filenames_all[i] # its filename
			dic_label[filename] = pred_label # save into the dic
		NMI = normalized_mutual_info_score(labels_all, preds) * 100 # Compute the NMI.
		torch.cuda.empty_cache()
		return dic_label, NMI

	def eval_network(self, val_list, val_path, **kwargs):
		self.eval()
		files, feats = [], {}
		for line in open(val_list).read().splitlines():
			data = line.split()
			files.append(data[1])
			files.append(data[2])
		setfiles = list(set(files))
		setfiles.sort()  # Read the list of wav files
		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _ = soundfile.read(os.path.join(val_path, file))
			feat = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()
			with torch.no_grad():
				ref_feat = self.Network.forward(feat).detach().cpu()
			feats[file]     = ref_feat # Extract features for each data, get the feature dict
		scores, labels  = [], []
		for line in open(val_list).read().splitlines():
			data = line.split()
			ref_feat = F.normalize(feats[data[1]].cuda(), p=2, dim=1) # feature 1
			com_feat = F.normalize(feats[data[2]].cuda(), p=2, dim=1) # feature 2
			score = numpy.mean(torch.matmul(ref_feat, com_feat.T).detach().cpu().numpy()) # Get the score
			scores.append(score)
			labels.append(int(data[0]))
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
		return [EER, minDCF]

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		print("Model %s loaded!"%(path))
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					# print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)
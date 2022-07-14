import glob, numpy, os, random, soundfile, torch, wave
from scipy import signal
from tools import *

def get_Loader(args, dic_label = None, cluster_only = False):
	# Get the loader for the cluster, batch_size is set as 1 to handlle the variable length input. Details see 1.2 part from here: https://github.com/TaoRuijie/TalkNet-ASD/blob/main/FAQ.md 
	clusterLoader = cluster_loader(**vars(args))
	clusterLoader = torch.utils.data.DataLoader(clusterLoader, batch_size = 1, shuffle = True, num_workers = args.n_cpu, drop_last = False)
	
	if cluster_only == True: # Only do clustering
		return clusterLoader
	# Get the loader for training
	trainLoader = train_loader(dic_label = dic_label, **vars(args))
	trainLoader = torch.utils.data.DataLoader(trainLoader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

	return trainLoader, clusterLoader

class train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, max_frames, dic_label, **kwargs):
		self.train_path = train_path
		self.max_frames = max_frames * 160 + 240 # Length of segment for training
		self.dic_label = dic_label # Pseudo labels dict
		self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-4] not in self.noiselist:
				self.noiselist[file.split('/')[-4]] = []
			self.noiselist[file.split('/')[-4]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
		self.data_list = []
		lines = open(train_list).read().splitlines()
		for index, line in enumerate(lines):
			file_name     = line.split()[1]
			self.data_list.append(file_name)

	def __getitem__(self, index):
		file = self.data_list[index] # Get the filename
		label = self.dic_label[file] # Load the pseudo label
		segments = self.load_wav(file = file) # Load the augmented segment
		segments = torch.FloatTensor(numpy.array(segments))
		return segments, label

	def load_wav(self, file):
		utterance, _ = soundfile.read(os.path.join(self.train_path, file)) # Read the wav file
		if utterance.shape[0] <= self.max_frames: # Padding if less than required length
			shortage = self.max_frames - utterance.shape[0]
			utterance = numpy.pad(utterance, (0, shortage), 'wrap')
		startframe = random.choice(range(0, utterance.shape[0] - (self.max_frames))) # Choose the startframe randomly
		segment = numpy.expand_dims(numpy.array(utterance[int(startframe):int(startframe)+self.max_frames]), axis = 0)
		
		if random.random() <= 0.5:
			segment = self.add_rev(segment, length = self.max_frames) # Rever
		if random.random() <= 0.5:
			segment = self.add_noise(segment, random.choice(['music', 'speech', 'noise']), length = self.max_frames) # Noise
			
		return segment[0]

	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio, length):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(numpy.float),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:length]

	def add_noise(self, audio, noisecat, length):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiselength = wave.open(noise, 'rb').getnframes() # Read the length of the noise file			
			if noiselength <= length:
				noiseaudio, _ = soundfile.read(noise)
				noiseaudio = numpy.pad(noiseaudio, (0, length - noiselength), 'wrap')
			else:
				start_frame = numpy.int64(random.random()*(noiselength-length)) # If length is enough
				noiseaudio, _ = soundfile.read(noise, start = start_frame, stop = start_frame + length) # Only read some part to improve speed
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio

class cluster_loader(object):
	def __init__(self, train_list, train_path, **kwargs):        
		self.data_list, self.data_length, self.data_label = [], [], []
		self.train_path = train_path
		lines = open(train_list).read().splitlines()
		# Get the ground-truth labels, that is used to compute the NMI for post-analyze.
		dictkeys = list(set([x.split()[0] for x in lines]))
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

		for lidx, line in enumerate(lines):
			data = line.split()
			file_name = data[1]
			file_length = float(data[-1])
			speaker_label = dictkeys[data[0]]
			self.data_list.append(file_name)  # Filename
			self.data_length.append(file_length) # Filelength
			self.data_label.append(speaker_label) # GT Speaker label

		# sort the training set by the length of the audios, audio with similar length are saved togethor.
		inds = numpy.array(self.data_length).argsort()
		self.data_list, self.data_length, self.data_label = numpy.array(self.data_list)[inds], \
															numpy.array(self.data_length)[inds], \
															numpy.array(self.data_label)[inds]
		self.minibatch = []
		start = 0
		while True: # Genearte each minibatch, audio with similar length are saved togethor.
			frame_length = self.data_length[start]
			minibatch_size = max(1, int(1600 // frame_length)) 
			end = min(len(self.data_list), start + minibatch_size)
			self.minibatch.append([self.data_list[start:end], frame_length, self.data_label[start:end]])
			if end == len(self.data_list):
				break
			start = end

	def __getitem__(self, index):
		data_lists, frame_length, data_labels = self.minibatch[index] # Get one minibatch
		filenames, labels, segments = [], [], []
		for num in range(len(data_lists)):
			filename = data_lists[num] # Read filename
			label = data_labels[num] # Read GT label
			audio, sr = soundfile.read(os.path.join(self.train_path, filename))
			if len(audio) < int(frame_length * sr):
				shortage    = int(frame_length * sr) - len(audio) + 1
				audio       = numpy.pad(audio, (0, shortage), 'wrap')
			audio = numpy.array(audio[:int(frame_length * sr)]) # Get clean utterance, better for clustering
			segments.append(audio)
			filenames.append(filename)
			labels.append(label)
		segments = torch.FloatTensor(numpy.array(segments))
		return segments, filenames, labels

	def __len__(self):
		return len(self.minibatch)
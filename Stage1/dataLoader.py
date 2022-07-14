import torch, numpy, random, os, math, glob, soundfile
from torch.utils.data import Dataset, DataLoader
from scipy import signal

class train_loader(Dataset):
    def __init__(self, max_frames, train_list, train_path, musan_path, **kwargs):
        self.max_frames = max_frames
        self.data_list = []
        self.noisetypes = ['noise','speech','music'] # Type of noise
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]} # The range of SNR
        self.noiselist = {} 
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav')) # All noise files in list
        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file) # All noise files in dic
        self.rir_files = numpy.load('rir.npy') # Load the rir file
        for line in open(train_list).read().splitlines():
            filename = os.path.join(train_path, line.split()[1])
            self.data_list.append(filename) # Load the training data list
                
    def __getitem__(self, index):
        audio = loadWAVSplit(self.data_list[index], self.max_frames).astype(numpy.float) # Load one utterance
        augment_profiles, audio_aug = [], []
        for ii in range(0,2): # Two segments of one utterance
            rir_gains = numpy.random.uniform(-7,3,1)
            rir_filts = random.choice(self.rir_files)
            noisecat    = random.choice(self.noisetypes)
            noisefile   = random.choice(self.noiselist[noisecat].copy()) # Augmentation information for each segment
            snr = [random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])]
            p = random.random()
            if p < 0.25:  # Add rir only
                augment_profiles.append({'rir_filt':rir_filts, 'rir_gain':rir_gains, 'add_noise': None, 'add_snr': None})
            elif p < 0.50: # Add noise only
                augment_profiles.append({'rir_filt':None, 'rir_gain':None, 'add_noise': noisefile, 'add_snr': snr})
            else: # Add both
                augment_profiles.append({'rir_filt':rir_filts, 'rir_gain':rir_gains, 'add_noise': noisefile, 'add_snr': snr})
        audio_aug.append(self.augment_wav(audio[0],augment_profiles[0])) # Segment 0 with augmentation method 0
        audio_aug.append(self.augment_wav(audio[1],augment_profiles[0])) # Segment 1 with augmentation method 0, used for AAT
        audio_aug.append(self.augment_wav(audio[1],augment_profiles[1])) # Segment 1 with augmentation method 1
        audio_aug = numpy.concatenate(audio_aug,axis=0) # Concate and return
        return torch.FloatTensor(audio_aug)

    def __len__(self):
        return len(self.data_list)

    def augment_wav(self,audio,augment):
        if augment['rir_filt'] is not None:
            rir     = numpy.multiply(augment['rir_filt'], pow(10, 0.1 * augment['rir_gain']))    
            audio   = signal.convolve(audio, rir, mode='full')[:len(audio)]
        if augment['add_noise'] is not None:
            noiseaudio  = loadWAV(augment['add_noise'], self.max_frames).astype(numpy.float)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
            noise = numpy.sqrt(10 ** ((clean_db - noise_db - augment['add_snr']) / 10)) * noiseaudio
            audio = audio + noise
        else:
            audio = numpy.expand_dims(audio, 0)
        return audio

def loadWAV(filename, max_frames):
    max_audio = max_frames * 160 + 240 # 240 is for padding, for 15ms since window is 25ms and step is 10ms.
    audio, _ = soundfile.read(filename)
    audiosize = audio.shape[0]
    if audiosize <= max_audio: # Padding if the length is not enough
        shortage    = math.floor( ( max_audio - audiosize + 1 ) / 2 )
        audio       = numpy.pad(audio, (shortage, shortage), 'wrap')
        audiosize   = audio.shape[0]
    startframe = numpy.int64(random.random()*(audiosize-max_audio)) # Randomly select a start frame to extract audio
    feat = numpy.stack([audio[int(startframe):int(startframe)+max_audio]],axis=0)
    return feat

def loadWAVSplit(filename, max_frames): # Load two segments
    max_audio = max_frames * 160 + 240
    audio, _ = soundfile.read(filename)
    audiosize = audio.shape[0]
    if audiosize <= max_audio:
        shortage    = math.floor( ( max_audio - audiosize) / 2 )
        audio       = numpy.pad(audio, (shortage, shortage), 'wrap')
        audiosize   = audio.shape[0]
    randsize = audiosize - (max_audio*2) # Select two segments
    startframe = random.sample(range(0, randsize), 2)
    startframe.sort()
    startframe[1] += max_audio # Non-overlapped two segments
    startframe = numpy.array(startframe)
    numpy.random.shuffle(startframe)
    feats = []
    for asf in startframe: # Startframe[0] means the 1st segment, Startframe[1] means the 2nd segment
        feats.append(audio[int(asf):int(asf)+max_audio])
    feat = numpy.stack(feats,axis=0)
    return feat

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

def get_loader(args): # Define the data loader
    trainLoader = train_loader(**vars(args))
    trainLoader = torch.utils.data.DataLoader(
        trainLoader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=5,
    )
    return trainLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy, tqdm, sys, time, soundfile

from loss import *
from encoder import *
from tools import *

class model(nn.Module):
    def __init__(self, lr, lr_decay, **kwargs):
        super(model, self).__init__()
        self.Network = ECAPA_TDNN().cuda() # Speaker encoder
        self.Loss = LossFunction().cuda() # Contrastive loss
        self.AATNet  = AATNet().cuda() # AAT, which is used to improve the performace
        self.Reverse = Reverse().cuda() # AAT
        self.OptimNet = torch.optim.Adam(list(self.Network.parameters()) + list(self.Loss.parameters()), lr = lr)
        self.OptimAAT = torch.optim.Adam(self.AATNet.parameters(), lr = lr)
        self.Scheduler = torch.optim.lr_scheduler.StepLR(self.OptimNet, step_size = 5, gamma=lr_decay)
        print("Model para number = %.2f"%(sum(param.numel() for param in self.Network.parameters()) / 1024 / 1024))

    def train_network(self, loader, epoch):
        # Contrastive learning with AAT, for more details about AAT, please check here: https://github.com/joonson/voxceleb_unsupervised
        self.train()
        self.Scheduler.step(epoch - 1) # Update the learning rate
        loss, top1 = 0, 0
        lr = self.OptimNet.param_groups[0]['lr'] # Read the current learning rate
        criterion   = torch.nn.CrossEntropyLoss() # Use for AAT
        AAT_labels = torch.LongTensor([1]*loader.batch_size+[0]*loader.batch_size).cuda() # AAT labels
        tstart = time.time() # Used to monitor the training speed
        for counter, data in enumerate(loader, start = 1):     
            data = data.transpose(0,1)
            feat = []
            for inp in data:
                feat.append(self.Network.forward(torch.FloatTensor(inp).cuda())) # Feed the segments to get the speaker embeddings
            feat = torch.stack(feat,dim=1).squeeze()
            self.zero_grad()
            # Train discriminator
            out_a, out_s, out_p = feat[:,0,:].detach(), feat[:,1,:].detach(), feat[:,2,:].detach()
            in_AAT = torch.cat((torch.cat((out_a,out_s),1),torch.cat((out_a,out_p),1)),0)
            out_AAT = self.AATNet(in_AAT)
            dloss  = criterion(out_AAT, AAT_labels)
            dloss.backward()
            self.OptimAAT.step()
            # Train model
            self.zero_grad()
            in_AAT = torch.cat((torch.cat((feat[:,0,:],feat[:,1,:]),1),torch.cat((feat[:,0,:],feat[:,2,:]),1)),0)
            out_AAT = self.AATNet(self.Reverse(in_AAT))
            closs   = criterion(out_AAT, AAT_labels) # AAT loss
            sloss, prec1 = self.Loss.forward(feat[:,[0,2],:])  # speaker loss              
            nloss = sloss + closs * 3 # Total loss
            loss    += nloss.detach().cpu()
            top1    += prec1 # Training acc
            nloss.backward()
            self.OptimNet.step() 
            time_used = time.time() - tstart # Time for this epoch
            sys.stdout.write("[%2d] Lr: %5f, %.2f%% (est %.1f mins) Loss %f EER/TAcc %2.3f%% \r"%(epoch, lr, 100 * (counter / loader.__len__()), time_used * loader.__len__() / counter / 60, loss/counter, top1/counter))
            sys.stdout.flush()
        sys.stdout.write("\n")
        return loss/counter, top1/counter, lr

    def evaluate_network(self, val_list, val_path, **kwargs):
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
        return EER, minDCF

    def save_network(self, path): # Save the model
        torch.save(self.state_dict(), path)

    def load_network(self, path): # Load the parameters of the pretrain model
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        print("Model %s loaded!"%(path))
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)


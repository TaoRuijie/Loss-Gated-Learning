import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import *
import numpy

# This function is modified from https://github.com/HobbitLong/SupContrast/blob/master/losses.py
class LossFunction(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs): # No temp param
        super(LossFunction, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, features):
        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(torch.device('cuda'))
        count = features.shape[1]
        feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        dot_feature  = F.cosine_similarity(feature.unsqueeze(-1),feature.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        dot_feature = dot_feature * self.w + self.b # We add this from angle protocol loss. 
        logits_max, _ = torch.max(dot_feature, dim=1, keepdim=True)
        logits = dot_feature - logits_max.detach()
        mask = mask.repeat(count, count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * count).view(-1, 1).to(torch.device('cuda')),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        loss = -(mask * log_prob).sum(1) / mask.sum(1)
        loss = loss.view(count, batch_size).mean()
        n         = batch_size * 2
        label     = torch.from_numpy(numpy.asarray(list(range(batch_size - 1,batch_size*2 - 1)) + list(range(0,batch_size)))).cuda()
        logits    = logits.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
        prec1, _  = accuracy(logits.detach().cpu(), label.detach().cpu(), topk=(1, 2)) # Compute the training acc
        
        return loss, prec1
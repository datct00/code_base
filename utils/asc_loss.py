import torch
import torch.nn as nn
from nets.project_head import ProjHead

class ASC_loss(nn.Module):
    def __init__(self, batch_size,device,sur_siml,pHead_sur):
        super(ASC_loss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.sur_siml = sur_siml
        self.pHead_sur = pHead_sur
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.similar_dice = BinaryDice_xent()
        
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        if self.pHead_sur == 'set_true':
            projHead_sur = ProjHead().cuda()
            z_i_head = projHead_sur(z_i)
            z_j_head = projHead_sur(z_j)
            z_head = torch.cat((z_i_head, z_j_head), dim=0)
        
        z = torch.cat((z_i, z_j), dim=0)
        if self.sur_siml == 'cos' and self.pHead_sur == 'set_false':
            z_flatten = torch.flatten(z, start_dim=1)
            sim_sur = self.similarity_f(z_flatten.unsqueeze(1), z_flatten.unsqueeze(0))
        elif self.sur_siml == 'dice' and self.pHead_sur == 'set_false':
            sim_sur = self.similar_dice(z.unsqueeze(1), z.unsqueeze(0))
        elif self.pHead_sur == 'set_true':
            sim_sur = self.similarity_f(z_head.unsqueeze(1), z_head.unsqueeze(0))

        sim = sim_sur
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            N, 1
        )
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class BinaryDice_xent(nn.Module):
    def __init__(self):
        super(BinaryDice_xent, self).__init__()
    def _dice(self, score, target):
        smooth = 1e-6
        dim_len = len(score.size())
        if dim_len == 5:
            dim=(2,3,4)
        elif dim_len == 4:
            dim=(2,3)
        intersect = torch.sum(score * target,dim=dim)
        y_sum = torch.sum(target * target,dim=dim)
        z_sum = torch.sum(score * score,dim=dim)
        dice_sim = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return dice_sim

    def forward(self, inputs, target):
        assert inputs.size()[2:] == target.size()[2:], 'predict & target shape do not match'
        dice_sim = self._dice(inputs, target)
        return dice_sim

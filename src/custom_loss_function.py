import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CustomMSE(nn.Module):
    def __init__(self, reduce=True):
        super(CustomMSE, self).__init__()
        self.reduce = reduce

    def forward(self, output, class_value, inequality, target):
        target = Variable(target.expand(output.size(1), output.size(0)).transpose(1,0), requires_grad=True)
        transformed_output = F.sigmoid(output)
        #print(transformed_output.size(), target.size())
        mask = np.zeros((output.size(0), output.size(1)))
        mask[range(output.size(0)),class_value.view(-1)] = 1 
        mask = Variable(torch.tensor(mask).cuda(), requires_grad=True)

        inequality_mask = np.zeros((output.size(0),output.size(1)))
        inequality_mask[range(output.size(0)),class_value.view(-1)] = inequality

        diff = (transformed_output - target)
        diff = diff.cpu().detach().numpy()
        #print('0')
        diff1 = (inequality_mask==1).astype(float)
        diff1 = Variable(torch.tensor(diff1).cuda(), requires_grad=True)
        loss1 = F.mse_loss(transformed_output.float(), target.float(), reduce=False)*diff1
     
        #print('1')
        diff2 = (inequality_mask==2).astype(float)
        diff2 = diff2*(diff > 0).astype(float)
        diff2 = Variable(torch.tensor(diff2).cuda(), requires_grad=True)
        loss2 = F.mse_loss(transformed_output.float(), target.float(), reduce=False)*diff2
        
        #print('2')
        diff3 = (inequality_mask==3).astype(float)
        diff3 = diff3*(diff < 0).astype(float)
        diff3 = Variable(torch.tensor(diff3).cuda(), requires_grad=True)
        loss3 = F.mse_loss(transformed_output.float(), target.float(), reduce=False)*diff3
        
        diff = Variable((torch.tensor(np.abs(diff))).cuda(), requires_grad=True)
        loss = (loss1 + loss2 + loss3)*diff
        
        return loss.sum()

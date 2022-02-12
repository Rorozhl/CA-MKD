from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np



class CAMKD(nn.Module):
    def __init__(self):
        super(CAMKD, self).__init__()
        # self.crit_ce = nn.CrossEntropyLoss()
        self.crit_ce = nn.CrossEntropyLoss(reduction='none')
        self.crit_mse = nn.MSELoss(reduction='none')
        # self.crit_mse = nn.MSELoss(reduction='mean')

    def forward(self, trans_feat_s_list, mid_feat_t_list, output_feat_t_list, target):
    
        bsz = target.shape[0]
        loss_t = [self.crit_ce(logit_t, target) for logit_t in output_feat_t_list]
        num_teacher = len(trans_feat_s_list)
        loss_t = torch.stack(loss_t, dim=0)
        weight = (1.0 - F.softmax(loss_t, dim=0)) / (num_teacher - 1)
        loss_st = []
        for mid_feat_s, mid_feat_t in zip(trans_feat_s_list, mid_feat_t_list):
            tmp_loss_st = self.crit_mse(mid_feat_s, mid_feat_t).reshape(bsz, -1).mean(-1)
            loss_st.append(tmp_loss_st)
        loss_st = torch.stack(loss_st, dim=0)
        loss = torch.mul(weight, loss_st).sum()
        # loss = torch.mul(attention, loss_st).sum()
        loss /= (1.0*bsz*num_teacher)

        # avg weight
        # loss_st = []
        # for mid_feat_s, mid_feat_t in zip(trans_feat_s_list, mid_feat_t_list):
        #     tmp_loss_st = self.crit_mse(mid_feat_s, mid_feat_t)
        #     loss_st.append(tmp_loss_st)
        # loss_st = torch.stack(loss_st, dim=0)
        # loss = loss_st.mean(0)
        return loss, weight



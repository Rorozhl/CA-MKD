import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from thundersvm import OneClassSVM as tsvm
from sklearn.svm import OneClassSVM as ssvm
from torch.autograd import Variable
from torch.nn.parameter import Parameter

alpha_grad_opt = []


def find_optimal_svm(vecs, method="ssvm-precomputed", nu=-1, gpu_id=0, is_norm=True):
    m = vecs.shape[0] 
    vec_tmp = vecs.reshape(vecs.shape[0], vecs.shape[1], -1)
    vec_mean = torch.mean(vec_tmp, dim=1)
    vec_norm = vec_mean.norm(dim=1, keepdim=True)
    if is_norm:
        vec_mean = vec_mean * (1 / vec_norm)
    G = torch.matmul(vec_mean, vec_mean.permute(1, 0))

    if nu == -1:
        nu = 1 / m
    elif nu > 1:
        nu = 1
    elif nu < 1 / m:
        nu = 1 / m
    ret = np.zeros(m)

    if method == "ssvm-precomputed":
        if G.is_cuda:
            G_cpu = G.cpu()
        else:
            G_cpu = G
        G_cpu = G_cpu.detach().numpy()
        svm = ssvm(kernel="precomputed", nu=nu, tol=1e-6)
        svm.fit(G_cpu)
    else:
        raise NotImplementedError

    if is_norm:
        if vec_norm.is_cuda:
            vec_norm = vec_norm.cpu()
        vec_norm = vec_norm.squeeze().detach().numpy()

    ret[svm.support_] = svm.dual_coef_ / (m * nu)
    if is_norm:
        ret_normalize = ret * (1 / vec_norm)
        ret_normalize = ret_normalize / np.sum(ret_normalize)
        ret_final = torch.from_numpy(ret_normalize).float()
    else:
        ret_final = torch.from_numpy(ret).float()

    return ret_final

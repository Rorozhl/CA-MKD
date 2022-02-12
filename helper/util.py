from __future__ import print_function

import json
import torch
import numpy as np
from collections import Counter
import torch.distributed as dist

LAYER = {'resnet20': np.arange(1, (20 - 2) // 2 + 1),  # 9
         'resnet56': np.arange(1, (56 - 2) // 2 + 1),  # 27
         'resnet110': np.arange(2, (110 - 2) // 2 + 1, 2),  # 27
         'wrn40x2': np.arange(1, (40 - 4) // 2 + 1),  # 18
         'wrn28x2': np.arange(1, (28 - 4) // 2 + 1),  # 12
         'wrn16x2': np.arange(1, (16 - 4) // 2 + 1),  # 6
         'resnet34': np.arange(1, (34 - 2) // 2 + 1),  # 16
         'resnet18': np.arange(1, (18 - 2) // 2 + 1),  # 8
         'resnet34im': np.arange(1, (34 - 2) // 2 + 1),  # 16
         'resnet18im': np.arange(1, (18 - 2) // 2 + 1),  # 8
         'resnet8x4': np.arange(1, (8 - 2) // 2 + 1),  # 3
         'resnet32x4': np.arange(1, (32 - 2) // 2 + 1),  # 15
         }

def unique_shape(s_shapes):
    n_s = []
    unique_shapes = []
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)
            n += 1
        n_s.append(n)
    return n_s, unique_shapes

def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_cifar(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def adjust_learning_rate(optimizer, epoch, step, len_epoch, old_lr):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    factor = epoch // 30

    # if epoch >= 80:
    #     factor = factor + 1

    lr = old_lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim = 1, largest = True, sorted = True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_list(output_list: list, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    batch_size = target.size(0)
    output_avg = torch.mean(torch.stack(output_list), dim=0)
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output_avg.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k == 1:
                top1_list = []
                for output in output_list:
                    maxk = max(topk)
                    _, pred = output.topk(maxk, 1, True, True)
                    pred = pred.t()
                    top1_list.append(pred[0])
                top1_array = np.array([top1.cpu().numpy()
                                       for top1 in top1_list]).transpose()
                top1 = [Counter(top1_array[i]).most_common(1)[0][0]
                        for i in range(batch_size)]
                top1 = torch.Tensor(top1).long().view(-1, batch_size)
                if torch.cuda.is_available():
                    top1 = top1.cuda()
                correct_top1 = top1.eq(target.view(1, -1).expand_as(top1))
                correct_k = correct_top1[:1].view(
                    -1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                correct_k = correct[:k].reshape(-1).float().sum(0,
                                                                keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))

    return res


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

def load_json_to_dict(json_path):
    """Loads json file to dict 

    Args:
        json_path: (string) path to json file
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def reduce_tensor(tensor, world_size = 1, op='avg'):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size > 1:
        rt = torch.true_divide(rt, world_size)
    return rt

if __name__ == '__main__':

    pass

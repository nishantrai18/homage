import torch
import numpy as np
import pickle
import os
from datetime import datetime
import glob
import re
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from collections import deque
from tqdm import tqdm 
from torchvision import transforms


def save_checkpoint(state, mode, is_best=0, gap=1, filename='models/checkpoint.pth.tar', keep_all=False):
    torch.save(state, filename)
    last_epoch_path = os.path.join(
        os.path.dirname(filename), 'mode_' + mode + '_epoch%s.pth.tar' % str(state['epoch']-gap))
    alternate_last_epoch_path = os.path.join(os.path.dirname(filename), 'epoch%s.pth.tar' % str(state['epoch']-gap))
    if not keep_all:
        try:
            os.remove(last_epoch_path)
        except:
            try:
                os.remove(alternate_last_epoch_path)
            except:
                print("Couldn't remove last_epoch_path: ", last_epoch_path, alternate_last_epoch_path)
                pass
    if is_best:
        past_best = glob.glob(os.path.join(os.path.dirname(filename), 'mode_' + mode + '_model_best_*.pth.tar'))
        for i in past_best:
            try: os.remove(i)
            except: pass
        torch.save(
            state,
            os.path.join(
                os.path.dirname(filename),
                'mode_' + mode + '_model_best_epoch%s.pth.tar' % str(state['epoch'])
            )
        )


def write_log(content, epoch, filename):
    if not os.path.exists(filename):
        log_file = open(filename, 'w')
    else:
        log_file = open(filename, 'a')
    log_file.write('## Epoch %d:\n' % epoch)
    log_file.write('time: %s\n' % str(datetime.now()))
    log_file.write(content + '\n\n')
    log_file.close()


def calc_topk_accuracy(output, target, topk=(1,)):
    '''
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, 
    calculate top-k accuracies.
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def calc_accuracy(output, target):
    '''output: (B, N); target: (B)'''
    target = target.squeeze()
    _, pred = torch.max(output, 1)
    return torch.mean((pred == target).float())

def calc_accuracy_binary(output, target):
    '''output, target: (B, N), output is logits, before sigmoid '''
    pred = output > 0
    acc = torch.mean((pred == target.byte()).float())
    del pred, output, target
    return acc

def get_topk_single_from_multilabel_pred(pred, target, ks):
    assert ks == sorted(ks)
    pred = pred.clone()
    out = []
    valid_indices = torch.zeros_like(pred, dtype=bool)
    for k in range(1, ks[-1] + 1):
        max_indices = torch.argmax(pred, dim=1)
        valid_indices[:, max_indices] = pred[:, max_indices] > 0
        if k in ks:
            topk_single = torch.sum(valid_indices & target)
            out.append(topk_single)
        pred[:, max_indices] = 0

    return out

def calc_per_class_multilabel_counts(output, target, num_ths=101):
    tp, tn, fp, fn = [torch.zeros((num_ths, output.shape[1])) for _ in range(4)]
    target = target > 0
    single_mask = target.sum(dim=-1) == 1
    for th in range(num_ths):
        pred = output >= th / (num_ths - 1)
        tp[th] = torch.sum(pred & target, axis=0)
        fp[th] = torch.sum(pred & ~target, axis=0)
        fn[th] = torch.sum(~pred & target, axis=0)
        tn[th] = torch.sum(~pred & ~target, axis=0)
    # single label metrics
    top1_single, top3_single = get_topk_single_from_multilabel_pred(output[single_mask], target[single_mask], [1, 3])
    all_single = torch.sum(single_mask)
    return tp, tn, fp, fn, top1_single, top3_single, all_single

def calc_hamming_loss(tp, tn, fp, fn, num_ths=101):
    th = (num_ths - 1) // 2
    tp, fp, tn, fn = tp[th].sum(), fp[th].sum(), tn[th].sum(), fn[th].sum()
    return ((fp + fn) / (fp + fn + tp + tn)).cpu().data.numpy()

def calc_mAP(tp, fp, num_ths=101):
    # clamp to > 0.5
    tp, fp = tp[(num_ths - 1) // 2 : -1], fp[(num_ths - 1) // 2 : -1]
    m = (tp + fp) > 0
    precision = torch.zeros_like(tp)
    precision[m] = tp[m] / (tp[m] + fp[m])
    return precision[m].mean().cpu().data.numpy()

def denorm(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    assert len(mean) == len(std) == 3
    inv_mean = [-mean[i]/std[i] for i in range(3)]
    inv_std = [1/i for i in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {} # save all data values here
        self.save_dict = {} # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=100):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)

    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def __len__(self):
        return self.count


class AccuracyTable(object):
    '''compute accuracy for each class'''
    def __init__(self, names):
        self.names = names
        self.dict = {}

    def update(self, pred, tar):
        pred = pred.flatten()
        tar = tar.flatten()
        for i, j in zip(pred, tar):
            i = int(i)
            j = int(j)
            if j not in self.dict.keys():
                self.dict[j] = {'count':0,'correct':0}
            self.dict[j]['count'] += 1
            if i == j:
                self.dict[j]['correct'] += 1

    def print_table(self):
        for key in sorted(self.dict.keys()):
            acc = self.dict[key]['correct'] / self.dict[key]['count']
            print('%25s: %5d, acc: %3d/%3d = %0.6f' \
                % (self.names[key], key, self.dict[key]['correct'], self.dict[key]['count'], acc))

    def print_dict(self):
        acc_dict = {}
        for key in sorted(self.dict.keys()):
            acc_dict[self.names[key].lower()] = self.dict[key]['correct'] / self.dict[key]['count']
        print(acc_dict)

class ConfusionMeter(object):
    '''compute and show confusion matrix'''
    def __init__(self, num_class):
        self.num_class = num_class
        self.mat = np.zeros((num_class, num_class))
        self.precision = []
        self.recall = []

    def update(self, pred, tar):
        pred, tar = pred.cpu().numpy(), tar.cpu().numpy()
        pred = np.squeeze(pred)
        tar = np.squeeze(tar)
        for p,t in zip(pred.flat, tar.flat):
            self.mat[p][t] += 1

    def print_mat(self):
        print('Confusion Matrix: (target in columns)')
        print(self.mat)

    def plot_mat(self, path, dictionary=None, annotate=False):
        plt.figure(dpi=600)
        plt.imshow(self.mat,
            cmap=plt.cm.jet,
            interpolation=None,
            extent=(0.5, np.shape(self.mat)[0]+0.5, np.shape(self.mat)[1]+0.5, 0.5))
        width, height = self.mat.shape
        if annotate:
            for x in range(width):
                for y in range(height):
                    plt.annotate(str(int(self.mat[x][y])), xy=(y+1, x+1),
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 fontsize=8)

        if dictionary is not None:
            plt.xticks([i+1 for i in range(width)],
                       [dictionary[i] for i in range(width)],
                       rotation='vertical')
            plt.yticks([i+1 for i in range(height)],
                       [dictionary[i] for i in range(height)])
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path, format='svg')
        plt.clf()

        # for i in range(width):
        #     if np.sum(self.mat[i,:]) != 0:
        #         self.precision.append(self.mat[i,i] / np.sum(self.mat[i,:]))
        #     if np.sum(self.mat[:,i]) != 0:
        #         self.recall.append(self.mat[i,i] / np.sum(self.mat[:,i]))
        # print('Average Precision: %0.4f' % np.mean(self.precision))
        # print('Average Recall: %0.4f' % np.mean(self.recall))





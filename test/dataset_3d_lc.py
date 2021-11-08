import torch
from torch.utils import data
from torchvision import transforms
import os
import sys
import time
import pickle
import csv
import glob
import pandas as pd
import numpy as np
import cv2

sys.path.append('../train')
import model_utils as mu

sys.path.append('../utils')
from augmentation import *
from tqdm import tqdm
from joblib import Parallel, delayed


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


toTensor = transforms.ToTensor()
toPILImage = transforms.ToPILImage()
def flow_loader(path):
    try:
        img = Image.open(path)
    except:
        return None
    f = toTensor(img)
    if f.mean() > 0.3:
        f -= 0.5
    return f


def fetch_imgs_seq(vpath, idx_block):
    seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
    return seq


def fill_nones(l):
    l = [l[i-1] if l[i] is None else l[i] for i in range(len(l))]
    l = [l[i-1] if l[i] is None else l[i] for i in range(len(l))]
    try:
        nonNoneL = [item for item in l if item is not None][0]
    except:
        nonNoneL = torch.zeros((1, 256, 256))
    return [torch.zeros(nonNoneL.shape) if l[i] is None else l[i] for i in range(len(l))]


def get_u_flow_path_list(vpath, idx_block):
    dataset = 'ucf101' if 'ucf101' in vpath else 'hmdb51'
    flow_base_path = os.path.join('/dev/shm/data/nishantr/flow/', dataset + '_flow/')
    vid_name = os.path.basename(os.path.normpath(vpath))
    return [os.path.join(flow_base_path, 'u', vid_name, 'frame%06d.jpg' % (i + 1)) for i in idx_block]


def get_v_flow_path_list(vpath, idx_block):
    dataset = 'ucf101' if 'ucf101' in vpath else 'hmdb51'
    flow_base_path = os.path.join('/dev/shm/data/nishantr/flow/', dataset + '_flow/')
    vid_name = os.path.basename(os.path.normpath(vpath))
    return [os.path.join(flow_base_path, 'v', vid_name, 'frame%06d.jpg' % (i + 1)) for i in idx_block]


def fetch_flow_seq(vpath, idx_block):
    u_flow_list = get_u_flow_path_list(vpath, idx_block)
    v_flow_list = get_v_flow_path_list(vpath, idx_block)

    u_seq = fill_nones([flow_loader(f) for f in u_flow_list])
    v_seq = fill_nones([flow_loader(f) for f in v_flow_list])

    seq = [toPILImage(torch.cat([u, v])) for u, v in zip(u_seq, v_seq)]
    return seq


def get_class_vid(vpath):
    return os.path.normpath(vpath).split('/')[-2:]


def load_detectron_feature(fdir, idx, opt):
    # opt is either hm or seg

    shape = (192, 256)
    num_channels = 17 if opt == 'hm' else 1

    def load_feature(path):
        try:
            x = np.load(path)[opt]
        except:
            x = np.zeros((0, 0, 0))

        # Match non-existent values
        if x.shape[1] == 0:
            x = np.zeros((num_channels, shape[0], shape[1]))

        x = torch.tensor(x, dtype=torch.float) / 255.0

        # Add extra channel in case it's not present
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        return x

    suffix = 'heatmap' if opt == 'hm' else 'segmask'
    fpath = os.path.join(fdir, suffix + '_%05d.npz' % idx)
    if os.path.isfile(fpath):
        return load_feature(fpath)
    else:
        # We do not have results lower than idx=2
        idx = max(3, idx)
        # We assume having all results for every two frames
        fpath0 = os.path.join(fdir, suffix + '_%05d.npz' % (idx - 1))
        fpath1 = os.path.join(fdir, suffix + '_%05d.npz' % (idx + 1))
        # This is not guaranteed to exist
        if not os.path.isfile(fpath1):
            fpath1 = fpath0
        a0, a1 = load_feature(fpath0), load_feature(fpath1)
        try:
            a_avg = (a0 + a1) / 2.0
        except:
            a_avg = None
        return a_avg


def fetch_kp_heatmap_seq(vpath, idx_block):
    assert '/frame/' in vpath, "Incorrect vpath received: {}".format(vpath)
    feature_vpath = vpath.replace('/frame/', '/heatmaps/')
    seq = fill_nones([load_detectron_feature(feature_vpath, idx, opt='hm') for idx in idx_block])

    if len(set([x.shape for x in seq])) > 1:
        # We now know the invalid paths, so no need to print them
        # print("Invalid path:", vpath)
        seq = [seq[len(seq) // 2] for _ in seq]
    return seq


def fetch_seg_mask_seq(vpath, idx_block):
    assert '/frame/' in vpath, "Incorrect vpath received: {}".format(vpath)
    feature_vpath = vpath.replace('/frame/', '/segmasks/')
    seq = fill_nones([load_detectron_feature(feature_vpath, idx, opt='seg') for idx in idx_block])
    return seq


class UCF101_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq =1,
                 downsample=3,
                 epsilon=5,
                 which_split=1,
                 modality=mu.ImgMode):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split
        self.modality = modality

        # splits
        if mode == 'train':
            split = '../data/ucf101/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            split = '../data/ucf101/test_split%02d.csv' % self.which_split # use test for val, temporary
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        action_file = os.path.join('../data/ucf101', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        # if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        downsample = self.downsample
        if (vlen - (self.num_seq * self.seq_len * self.downsample)) <= 0:
            downsample = ((vlen - 1) / (self.num_seq * self.seq_len * 1.0)) * 0.9

        n = 1
        if self.mode == 'test':
            seq_idx_block = np.arange(0, vlen, downsample) # all possible frames with downsampling
            seq_idx_block = seq_idx_block.astype(int)
            return [seq_idx_block, vpath]
        start_idx = np.random.choice(range(vlen-int(self.num_seq*self.seq_len*downsample)), n)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1)*downsample*self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len),0)*downsample
        seq_idx_block = seq_idx_block.astype(int)
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath)

        idx_block, vpath = items
        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, self.seq_len)
            idx_block = idx_block.reshape(self.num_seq*self.seq_len)

        seq = None
        if self.modality == mu.ImgMode:
            seq = fetch_imgs_seq(vpath, idx_block)
        elif self.modality == mu.FlowMode:
            seq = fetch_flow_seq(vpath, idx_block)
        elif self.modality == mu.KeypointHeatmap:
            seq = fetch_kp_heatmap_seq(vpath, idx_block)
        elif self.modality == mu.SegMask:
            seq = fetch_seg_mask_seq(vpath, idx_block)

        if self.modality in [mu.KeypointHeatmap, mu.SegMask]:
            seq = torch.stack(seq)

        # if self.mode == 'test':
        #     # apply same transform
        #     t_seq = [self.transform(seq) for _ in range(5)]
        # else:
        t_seq = self.transform(seq) # apply same transform
        # Convert tensor into list of tensors
        if self.modality in [mu.KeypointHeatmap, mu.SegMask]:
            t_seq = [t_seq[idx] for idx in range(t_seq.shape[0])]

        num_crop = None
        try:
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
        except:
            (C, H, W) = t_seq[0][0].size()
            tmp = [torch.stack(i, 0) for i in t_seq]
            assert len(tmp) == 5
            num_crop = 5
            t_seq = torch.stack(tmp, 1)

        if self.mode == 'test':
            # return all available clips, but cut into length = num_seq
            SL = t_seq.size(0)
            clips = []; i = 0
            while i+self.seq_len <= SL:
                clips.append(t_seq[i:i+self.seq_len, :])
                # i += self.seq_len//2
                i += self.seq_len
            if num_crop:
                # half overlap:
                clips = [torch.stack(clips[i:i+self.num_seq], 0).permute(2,0,3,1,4,5) for i in range(0,len(clips)+1-self.num_seq,self.num_seq//2)]
                NC = len(clips)
                t_seq = torch.stack(clips, 0).view(NC*num_crop, self.num_seq, C, self.seq_len, H, W)
            else:
                # half overlap:
                clips = [torch.stack(clips[i:i+self.num_seq], 0).transpose(1,2) for i in range(0,len(clips)+1-self.num_seq,self.num_seq//2)]
                t_seq = torch.stack(clips, 0)
        else:
            t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        try:
            vname = vpath.split('/')[-3]
            vid = self.encode_action(vname)
        except:
            vname = vpath.split('/')[-2]
            vid = self.encode_action(vname)

        label = torch.LongTensor([vid])
        idx = torch.LongTensor([index])

        return t_seq, label, idx

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class HMDB51_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=1,
                 downsample=1,
                 epsilon=5,
                 which_split=1,
                 modality=mu.ImgMode):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split
        self.modality = modality

        # splits
        if mode == 'train':
            split = '../data/hmdb51/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            split = '../data/hmdb51/test_split%02d.csv' % self.which_split # use test for val, temporary
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        action_file = os.path.join('../data/hmdb51', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        # if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        downsample = self.downsample
        if (vlen - (self.num_seq * self.seq_len * self.downsample)) <= 0:
            downsample = ((vlen - 1) / (self.num_seq * self.seq_len * 1.0)) * 0.9

        n=1
        if self.mode == 'test':
            seq_idx_block = np.arange(0, vlen, downsample) # all possible frames with downsampling
            seq_idx_block = seq_idx_block.astype(int)
            return [seq_idx_block, vpath]
        start_idx = np.random.choice(range(vlen-int(self.num_seq*self.seq_len*downsample)), n)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1)*downsample*self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len),0)*downsample
        seq_idx_block = seq_idx_block.astype(int)
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath)

        idx_block, vpath = items
        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, self.seq_len)
            idx_block = idx_block.reshape(self.num_seq*self.seq_len)

        seq = None
        if self.modality == mu.ImgMode:
            seq = fetch_imgs_seq(vpath, idx_block)
        elif self.modality == mu.FlowMode:
            seq = fetch_flow_seq(vpath, idx_block)
        elif self.modality == mu.KeypointHeatmap:
            seq = fetch_kp_heatmap_seq(vpath, idx_block)
        elif self.modality == mu.SegMask:
            seq = fetch_seg_mask_seq(vpath, idx_block)

        if self.modality in [mu.KeypointHeatmap, mu.SegMask]:
            seq = torch.stack(seq)

        t_seq = self.transform(seq) # apply same transform
        # Convert tensor into list of tensors
        if self.modality in [mu.KeypointHeatmap, mu.SegMask]:
            t_seq = [t_seq[idx] for idx in range(t_seq.shape[0])]

        num_crop = None
        try:
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
        except:
            (C, H, W) = t_seq[0][0].size()
            tmp = [torch.stack(i, 0) for i in t_seq]
            assert len(tmp) == 5
            num_crop = 5
            t_seq = torch.stack(tmp, 1)
        # print(t_seq.size())
        # import ipdb; ipdb.set_trace()
        if self.mode == 'test':
            # return all available clips, but cut into length = num_seq
            SL = t_seq.size(0)
            clips = []; i = 0
            while i+self.seq_len <= SL:
                clips.append(t_seq[i:i+self.seq_len, :])
                # i += self.seq_len//2
                i += self.seq_len
            if num_crop:
                # half overlap:
                clips = [torch.stack(clips[i:i+self.num_seq], 0).permute(2,0,3,1,4,5) for i in range(0,len(clips)+1-self.num_seq,self.num_seq//2)]
                NC = len(clips)
                t_seq = torch.stack(clips, 0).view(NC*num_crop, self.num_seq, C, self.seq_len, H, W)
            else:
                # half overlap:
                clips = [torch.stack(clips[i:i+self.num_seq], 0).transpose(1,2) for i in range(0,len(clips)+1-self.num_seq,3*self.num_seq//4)]
                t_seq = torch.stack(clips, 0)
        else:
            t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        try:
            vname = vpath.split('/')[-3]
            vid = self.encode_action(vname)
        except:
            vname = vpath.split('/')[-2]
            vid = self.encode_action(vname)

        label = torch.LongTensor([vid])
        idx = torch.LongTensor([index])

        return t_seq, label, idx

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


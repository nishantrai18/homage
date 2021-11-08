import torch
from torch.utils import data
from torchvision import transforms
import os
import sys
import time
import pickle
import glob
import csv
import scipy.io
import pandas as pd
import numpy as np
import cv2
import random

import model_utils as mu

sys.path.append('../utils')

from copy import deepcopy
from augmentation import *
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict


def pil_loader(path):
    img = Image.open(path)
    return img.convert('RGB')


toTensor = transforms.ToTensor()
toPILImage = transforms.ToPILImage()
def flow_loader(path):
    try:
        img = Image.open(path)
    except:
        return None
    return toTensor(img)


class BaseDataloader(data.Dataset):

    def __init__(
        self,
        mode,
        transform,
        seq_len,
        num_seq,
        downsample,
        which_split,
        vals_to_return,
        sampling_method,
        dataset,
        debug=False,
        postfix='',
        multilabel_supervision=False,
    ):
        super(BaseDataloader, self).__init__()

        self.dataset = dataset
        self.mode = mode
        self.debug = debug
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.which_split = which_split
        # Describes which particular items to return e.g. ["imgs", "poses", "labels"]
        self.vals_to_return = set(vals_to_return)
        self.sampling_method = sampling_method
        self.postfix = postfix
        self.num_classes = mu.get_num_classes(self.dataset if not postfix else '-'.join((self.dataset, postfix)))
        self.multilabel_supervision = multilabel_supervision

        if self.sampling_method == "random":
            assert "imgs" not in self.vals_to_return, \
                "Invalid sampling method provided for imgs: {}".format(self.sampling_method)

        # splits
        mode_str = "test" if ((mode == 'val') or (mode == 'test')) else mode
        mode_split_str = '/' + mode_str + '_split%02d.csv' % self.which_split

        split = '../data/' + self.dataset + mode_split_str

        if "panasonic" in dataset:
            # FIXME: change when access is changed
            split = os.path.join('{}/panasonic/{}_split{}.csv'.format(
                                     os.environ['BASE_DIR'], mode,
                                     '_' + postfix if postfix else ''))
            # maximum 15 values
            video_info = pd.read_csv(split, header=None, names=list(range(20)))
        else:
            video_info = pd.read_csv(split, header=None)

        # Debug mode in order to test for overfitting
        if self.debug:
            video_info = video_info.sample(n=25, random_state=42)

        # poses_mat_dict: vpath to poses_mat
        self.poses_dict = {}

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        action_file = os.path.join('../data/' + self.dataset, 'classInd.txt')
        if "panasonic" in dataset:
            action_file = os.path.join('../data/' + self.dataset, 'classInd{}.txt'.format('_' + postfix if postfix else ''))
        self.action_dict_decode, self.action_dict_encode = self.get_action_idx(action_file)

        drop_idx = set()
        # track duplicate categories
        dup_cat_dict = defaultdict(list)
        
        # filter out too short videos:
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
            # FIXME: make dataloader more modular. This only works for panasonic data
            num_views = int(len([i for i in np.array(row) if i == i]) / 4)
            # drop indices with no ego-view
            view_names = [row[i * 4].split('/')[-1].split('_')[2] for i in range(num_views)]
            if not 'v000' in view_names:
                drop_idx.add(idx)
                continue
            # drop indices with only a single view
            if num_views < 2:
                drop_idx.add(idx)
                continue
            # drop indices with multiple categories
            p, r, _, a = row[0].split('/')[-1].split('_')
            s = row[1]
            e = row[2]
            key = (p, r, a, s, e)
            dup_cat_dict[key].append(idx)

            vpath, vstart, vend, vname = row[:4]
            vlen = int(vend - vstart + 1)
            if self.sampling_method == 'disjoint':
                if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                    drop_idx.add(idx)
            else:
                if vlen <= 0:
                    drop_idx.add(idx)

        dup_cat_dict = {k: v for k, v in dup_cat_dict.items() if len(v) > 1}
        dup_keys = []
        if self.multilabel_supervision:
            # merge repetitive lines
            for v in dup_cat_dict.values():
                video_info.iloc[v[0], 3] = ','.join([video_info.iloc[i, 3] for i in v])
            # drop segments that are repetitive
            dup_keys = [i for v in dup_cat_dict.values() for i in v[1:]]
        else:
            # drop segments with multiple assigned categories
            dup_keys = [i for v in dup_cat_dict.values() for i in v]

        for i in dup_keys:
            drop_idx.add(i)

        self.drop_idx = list(drop_idx)
        self.video_info = video_info.drop(self.drop_idx, axis=0)

        # FIXME: panasonic data don't need val sampling here. Try making this more modular!
        # elif self.mode == 'val':
        #     self.video_info = self.video_info.sample(frac=0.3)
        #     # self.video_info = self.video_info.head(int(0.3 * len(self.video_info)))

        self.idx_sampler = None
        if self.sampling_method == "dynamic":
            self.idx_sampler = self.idx_sampler_dynamic
        if self.sampling_method == "disjoint":
            self.idx_sampler = self.idx_sampler_disjoint
        elif self.sampling_method == "random":
            self.idx_sampler = self.idx_sampler_random

        if self.mode == 'test':
            self.idx_sampler = self.idx_sampler_test

        if mu.FlowMode in self.vals_to_return:
            self.setup_flow_modality()

        # shuffle not required due to external sampler

    def get_action_idx(self, action_file):
        action_dict_decode, action_dict_encode = {}, {}
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            action_dict_decode[act_id] = act_name
            action_dict_encode[act_name] = act_id
        return action_dict_decode, action_dict_encode

    def setup_flow_modality(self):
        '''Can be overriden in the derived classes'''
        vpath, _ = self.video_info.iloc[0]
        vpath = vpath.rstrip('/')
        base_dir = vpath.split(self.dataset)[0]
        print("Base dir for flow:", base_dir)
        self.flow_base_path = os.path.join(base_dir, 'flow', self.dataset + '_flow/')

    def idx_sampler_test(self, seq_len, num_seq, vlen, vpath):
        '''
        sample index uniformly from a video
        '''

        downsample = self.downsample
        if (vlen - (num_seq * seq_len * self.downsample)) <= 0:
            downsample = ((vlen - 1) / (num_seq * seq_len * 1.0)) * 0.9

        seq_idx = np.expand_dims(np.arange(num_seq), -1) * downsample * seq_len
        seq_idx_block = seq_idx + np.expand_dims(np.arange(seq_len), 0) * downsample
        seq_idx_block = seq_idx_block.astype(int)

        return [seq_idx_block, vpath]

    def idx_sampler_dynamic(self, seq_len, num_seq, vlen, vpath, idx_offset=0, start_idx=-1):
        '''sample index from a video'''
        downsample = self.downsample
        if (vlen - (num_seq * seq_len * self.downsample)) <= 0:
            downsample = ((vlen - 1) / (num_seq * seq_len * 1.0)) * 0.9

        n = 1
        if start_idx < 0:
            try:
                start_idx = np.random.choice(range(vlen - int(num_seq * seq_len * downsample)), n)
            except:
                print("Error!", vpath, vlen, num_seq, seq_len, downsample, n)

        seq_idx = np.expand_dims(np.arange(num_seq), -1) * downsample * seq_len + start_idx + idx_offset
        seq_idx_block = seq_idx + np.expand_dims(np.arange(seq_len), 0) * downsample
        seq_idx_block = seq_idx_block.astype(int)

        return [seq_idx_block, vpath], start_idx

    def idx_sampler_disjoint(self, seq_len, num_seq, vlen, vpath):
        '''sample index from a video'''

        if (vlen - (num_seq * seq_len * self.downsample)) <= 0:
            return None

        n = 1
        if self.mode == 'test':
            seq_idx_block = np.arange(0, vlen, self.downsample) # all possible frames with downsampling
            return [seq_idx_block, vpath]

        start_idx = np.random.choice(range(vlen - (num_seq * seq_len * self.downsample)), n)
        seq_idx = np.expand_dims(np.arange(num_seq), -1) * self.downsample * seq_len + start_idx
        # Shape num_seq x seq_len
        seq_idx_block = seq_idx + np.expand_dims(np.arange(seq_len), 0) * self.downsample

        return [seq_idx_block, vpath]

    def idx_sampler_random(self, seq_len, num_seq, vlen, vpath):
        '''sample index from a video'''

        # Here we compute the max downsampling we could perform
        max_ds = ((vlen - 1) // seq_len)

        if max_ds <= 0:
            return None

        if self.mode == 'test':
            seq_idx_block = np.arange(0, vlen, self.downsample)
            # all possible frames with downsampling
            return [seq_idx_block, vpath]

        seq_idx_block = []
        for i in range(num_seq):
            rand_ds = random.randint(1, max_ds)
            start_idx = random.randint(0, vlen - (seq_len * rand_ds) - 1)
            seq_idx = np.arange(start=start_idx, stop=(start_idx + (seq_len*rand_ds)), step=rand_ds)
            seq_idx_block.append(seq_idx)

        seq_idx_block = np.array(seq_idx_block)

        return [seq_idx_block, vpath]

    def fetch_imgs_seq(self, vpath, seq_len, idx_block):
        '''Can be overriden in the derived classes'''
        img_list = [os.path.join(vpath, 'image_%05d.jpg' % (i + 1)) for i in idx_block]
        seq = [pil_loader(f) for f in img_list]
        img_t_seq = self.transform["imgs"](seq)  # apply same transform
        (IC, IH, IW) = img_t_seq[0].size()
        img_t_seq = torch.stack(img_t_seq, 0)
        img_t_seq = img_t_seq.view(self.num_seq, seq_len, IC, IH, IW).transpose(1, 2)
        return img_t_seq

    def get_class_vid(self, vpath):
        return os.path.normpath(vpath).split('/')[-2:]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        # Remove trailing backslash if any
        vpath = vpath.rstrip('/')

        seq_len = self.seq_len
        if "tgt" in self.vals_to_return:
            seq_len = 2 * self.seq_len

        items = self.idx_sampler(seq_len, self.num_seq, vlen, vpath)
        if items is None:
            print(vpath)

        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, seq_len)
        idx_block = idx_block.reshape(self.num_seq * seq_len)

        vals = {}

        # Populate return list
        if mu.ImgMode in self.vals_to_return:
            img_t_seq = self.fetch_imgs_seq(vpath, seq_len, idx_block)
            vals[mu.ImgMode] = img_t_seq

        # Process double length target results
        if "tgt" in self.vals_to_return:
            orig_keys = list(vals.keys())
            for k in orig_keys:
                full_x = vals[k]
                vals[k] = full_x[:, :self.seq_len, ...]
                vals["tgt_" + k] = full_x[:, self.seq_len:, ...]
        if "labels" in self.vals_to_return:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            vals["labels"] = label

        # Add video index field
        vals["vnames"] = torch.LongTensor([index])

        return vals

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class BaseDataloaderHMDB(BaseDataloader):

    def __init__(
        self,
        mode,
        transform,
        seq_len,
        num_seq,
        downsample,
        which_split,
        vals_to_return,
        sampling_method,
        dataset
    ):
        super(BaseDataloaderHMDB, self).__init__(
            mode,
            transform,
            seq_len,
            num_seq,
            downsample,
            which_split,
            vals_to_return,
            sampling_method,
            dataset=dataset
        )


class HMDB51_3d(BaseDataloaderHMDB):
    def __init__(
        self,
        mode='train',
        transform=None,
        seq_len=5,
        num_seq=6,
        downsample=1,
        which_split=1,
        vals_to_return=["imgs"],
        sampling_method="dynamic"
    ):
        super(HMDB51_3d, self).__init__(
            mode,
            transform,
            seq_len,
            num_seq,
            downsample,
            which_split,
            vals_to_return,
            sampling_method,
            dataset="hmdb51"
        )


class JHMDB_3d(BaseDataloaderHMDB):
    def __init__(
        self,
        mode='train',
        transform=None,
        seq_len=5,
        num_seq=6,
        downsample=1,
        which_split=1,
        vals_to_return=["imgs"],
        sampling_method="dynamic"
    ):
        super(JHMDB_3d, self).__init__(
            mode,
            transform,
            seq_len,
            num_seq,
            downsample,
            which_split,
            vals_to_return,
            sampling_method,
            dataset="jhmdb"
        )

# Constant factor which converts from a 30fps video to the corresponding spectrogram
imgFrameIdxToAudioIdxFactor = 86.15 / 30.0


def get_spectrogram_window_length(seq_len, num_seq, downsample):
    return int(imgFrameIdxToAudioIdxFactor * seq_len * num_seq * downsample)


class Panasonic_3d(BaseDataloader):

    def __init__(
        self,
        mode='train',
        transform=None,
        seq_len=5,
        num_seq=6,
        downsample=3,
        which_split=1,
        vals_to_return=["imgs"],
        sampling_method="dynamic",
        debug=False,
        dataset="panasonic",
        postfix='',
        multilabel_supervision=False,
    ):
        super(Panasonic_3d, self).__init__(
            mode,
            transform,
            seq_len,
            num_seq,
            downsample,
            which_split,
            vals_to_return,
            sampling_method,
            dataset=dataset,
            debug=debug,
            postfix=postfix,
            multilabel_supervision=multilabel_supervision,
        )

        self.idx_sampler = self.idx_sampler_fixed_frame_rate

    def idx_sampler_test(self, seq_len, num_seq, vlen, vpath, idx_offset=0, start_idx=-1):
        '''sample index from a video'''
        downsample = self.downsample

        actStart, actEnd = idx_offset, idx_offset + vlen
        totFrames = int(num_seq * seq_len * downsample)

        # start, end of the clip area under consideration
        start = min(actStart, max(0, (actStart + actEnd - totFrames) // 2))
        end = max(actEnd, (actStart + actEnd + totFrames) // 2)

        print(actStart, actEnd, totFrames, start, end)

        n = 1
        if start_idx < 0:
            try:
                start_idx = np.random.choice(range(start, end - totFrames), n)
            except:
                print("Error!", vpath, vlen, num_seq, seq_len, downsample, n)

        seq_idx = np.expand_dims(np.arange(num_seq), -1) * downsample * seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(seq_len), 0) * downsample
        seq_idx_block = seq_idx_block.astype(int)

        return [seq_idx_block, vpath], start_idx

    def idx_sampler_fixed_frame_rate(self, seq_len, num_seq, vlen, vpath, idx_offset=0, start_idx=-1, hard_stop=True):
        '''sample index from a video'''
        downsample = self.downsample

        actStart, actEnd = int(idx_offset), int(idx_offset + vlen)
        totFrames = int(num_seq * seq_len * downsample)

        # start, end of the clip area under consideration
        start = int(min(actStart, max(0, (actStart + actEnd - totFrames) / 2)))
        end = int((actStart + actEnd + totFrames) / 2)
        if hard_stop:
            end = min(actEnd, end)
        else:
            end = max(actEnd, end)

        # Add dynamic sampling in case there are not enough frames
        if ((end - start) - totFrames) <= 0:
            downsample = ((end - start) / (num_seq * seq_len * 1.0)) * 0.95
            totFrames = int(num_seq * seq_len * downsample)

        n = 1
        if start_idx < 0:
            try:
                start_idx = np.random.choice(range(start, end - totFrames), n)
            except:
                print("Error!", vpath, vlen, num_seq, seq_len, downsample, n, start, end)

        seq_idx = np.expand_dims(np.arange(num_seq), -1) * downsample * seq_len + start_idx + 1
        seq_idx_block = seq_idx + np.expand_dims(np.arange(seq_len), 0) * downsample
        seq_idx_block = seq_idx_block.astype(int)

        return [seq_idx_block, vpath], start_idx

    def fetch_audio_seq(self, vpath, idx_block):
        '''Can be overriden in the derived classes'''
        # Load the spectrogram image
        person, vid = vpath.split('/')[-2:]
        spectrogram_path = '{}/panasonic/spectrogram/{}/{}_spec.jpg'.format(
            os.environ['BASE_DIR'], person, vid)

        # Get the cropped spectrogram
        minFrameIdx, maxFrameIdx = np.min(idx_block), np.max(idx_block)
        audioFrameLength = get_spectrogram_window_length(self.seq_len, self.num_seq, self.downsample)

        # Safely load image, otherwise pass zero tensor (We have around 20 audio files without spectrograms)
        if os.path.isfile(spectrogram_path):
            spectrogram = pil_loader(spectrogram_path)
        else:
            spectrogram = np.zeros((audioFrameLength + 1, 128, 3), dtype=np.float32)

        spectrogram = self.transform[mu.AudioMode](spectrogram)
        origShape = spectrogram.shape

        # Choose mid based on handling right edge case
        midAudioFrameIdx = int(imgFrameIdxToAudioIdxFactor * (minFrameIdx + maxFrameIdx) * 0.5)
        midAudioFrameIdx = min(midAudioFrameIdx, int(origShape[1] - (0.5 * audioFrameLength)))

        # Choose start based on handling left edge case. Start from 1 because negative indexing becomes an issue with 0
        startAudioIdx = max(1, int(midAudioFrameIdx - (0.5 * audioFrameLength)))

        # Note: We reverse the order of the spectrogram due to the nature it's dumped in
        cropped_spectrogram = spectrogram[:, -(startAudioIdx + audioFrameLength):-(startAudioIdx), ...]
        cropped_spectrogram = torch.flip(cropped_spectrogram, [1])

        assert cropped_spectrogram.shape[1] == audioFrameLength,\
            "Invalid shape: {}, Orig shape: {}, Misc: {}, {}, {}, {}, {}, Path: {}".format(
                cropped_spectrogram.shape, origShape, audioFrameLength,
                minFrameIdx, maxFrameIdx, startAudioIdx, midAudioFrameIdx, spectrogram_path)

        # Returned image is channels x time x 128
        return cropped_spectrogram

    def get_row_details(self, row):
        row = deepcopy(list(row))

        num_views = int(len([i for i in row if i == i]) / 4)
        i0 = [i for i in range(num_views) if 'v000' in row[i * 4]][0]
        i1 = np.random.choice(np.setdiff1d(range(num_views), [i0]))
        vpath0, vstart0, vend0, vname, vpath1, vstart1, vend1, _ = row[4*i0: 4*i0+4] + row[4*i1: 4*i1+4]

        return vpath0, vstart0, vend0, vname, vpath1, vstart1, vend1

    def __getitem__(self, index):
        row = np.array(self.video_info.iloc[index]).tolist()
        vpath0, vstart0, vend0, vname, vpath1, vstart1, vend1 = self.get_row_details(row)

        # FIXME: make sure the first frame is synchronized
        vstart = max(vstart0, vstart1)
        vend = min(vend0, vend1)
        vlen = int(vend - vstart + 1)

        # Remove trailing backslash if any
        vpath0 = vpath0.rstrip('/')
        vpath1 = vpath1.rstrip('/')

        seq_len = self.seq_len
        if "tgt" in self.vals_to_return:
            seq_len = 2 * self.seq_len

        items, start_idx = self.idx_sampler(seq_len, self.num_seq, vlen, vpath0, idx_offset=vstart)
        idx_block, vpath = items

        assert idx_block.shape == (self.num_seq, seq_len)
        idx_block = idx_block.reshape(self.num_seq * seq_len)

        vals = {}

        # FIXME: make more general
        vals_to_return = np.unique([i.split('-')[0] for i in self.vals_to_return])
        # Populate return list
        if mu.ImgMode in vals_to_return:
            img_t_seq0 = self.fetch_imgs_seq(vpath0, seq_len, idx_block)
            # 0 stands for the ego-view while 1 stands for the third-person view
            vals['{}-0'.format(mu.ImgMode)] = img_t_seq0
            img_t_seq1 = self.fetch_imgs_seq(vpath1, seq_len, idx_block)
            vals['{}-1'.format(mu.ImgMode)] = img_t_seq1
        if mu.AudioMode in self.vals_to_return:
            vals[mu.AudioMode] = self.fetch_audio_seq(vpath, idx_block)

        # Process double length target results
        if "tgt" in self.vals_to_return:
            orig_keys = list(vals.keys())
            for k in orig_keys:
                full_x = vals[k]
                vals[k] = full_x[:, :self.seq_len, ...]
                vals["tgt_" + k] = full_x[:, self.seq_len:, ...]
        if "labels" in self.vals_to_return:
            if self.multilabel_supervision:
                label = torch.zeros(self.num_classes).long()
                for n in vname.split(','):
                    label[self.encode_action(n)] = 1
            else:
                vid = self.encode_action(vname)
                label = torch.LongTensor([vid])
            vals["labels"] = label

        # Add video index field
        vals["vnames"] = torch.LongTensor([index])

        return vals


class HierarchicalPanasonic(Panasonic_3d):

    def __init__(
        self,
        mode='train',
        transform=None,
        seq_len=5,
        num_seq=6,
        downsample=3,
        which_split=1,
        vals_to_return=["imgs-0"],
        sampling="all",
        debug=False,
    ):
        super(HierarchicalPanasonic, self).__init__(
            mode,
            transform,
            seq_len,
            num_seq,
            downsample,
            which_split,
            vals_to_return,
            dataset="panasonic",
            debug=debug,
        )

        # sampling: all, single
        self.overlap = 0.25
        self.sampling = sampling
        self.sampler = self.sample_all_blocks if self.sampling == 'all' else self.sample_a_block

        # self.video_info contains information about video level stats
        self.num_classes = {'video': mu.get_num_classes('panasonic'), 'atomic': mu.get_num_classes('panasonic-atomic')}
        self.dense_labels, self.split, self.action_info, self.action_dict, self.row_info = \
            self.populate_video_annotations_details()
        self.video_names = list(self.dense_labels['video'].keys())

        random.Random(42).shuffle(self.video_names)

        # Randomly choose subsample as debug set
        if self.debug:
            self.video_names = self.video_names[:25]

        # Populate the number of frames in each video
        self.num_video_frames = {}
        for video in self.dense_labels['video'].keys():
            frame_ids = [int(f.replace('.jpg', '').split('_')[-1]) for f in os.listdir(video)]
            self.num_video_frames[video] = max(frame_ids)

    def populate_video_annotations_details(self):
        '''
        Populates video annotation details based on video-level and atomic-action level details
        '''

        levels = ['video', 'atomic']

        dense_labels, split, action_info, action_dict, row_info = {}, {}, {}, {}, {}

        # Go over each level and perform the necessary loading operations for it
        for level in levels:
            postfix = '' if level == 'video' else '_atomic'

            # load the annotation csvs
            split[level] = os.path.join('{}/panasonic/{}_split{}.csv'.format(
                os.environ['BASE_DIR'], self.mode, postfix))
            action_info[level] = pd.read_csv(split[level], header=None, names=list(range(20)))

            # Get the action_dict encoders and decoders
            action_file = os.path.join('../data/' + self.dataset, 'classInd{}.txt'.format(postfix))
            action_dict[level] = {}
            action_dict[level]['decode'], action_dict[level]['encode'] = self.get_action_idx(action_file)

            # Store info about the row
            row_info[level] = {}

            # Populate all action segments for each video
            dense_labels[level] = {}
            for idx, row in tqdm(action_info[level].iterrows(), total=len(action_info[level])):
                # Get vname, list((start, end, (atomic)action_id))
                num_views = int(len([i for i in np.array(row) if i == i]) / 4)
                # Hacky way to find out view 0 idx
                v0_idx = next(iter([i for i in range(num_views) if 'v000' in row[i * 4]]), None)
                if v0_idx is None:
                    # We don't have an ego view
                    continue
                vpath, start, end, action = row[4*v0_idx: 4*v0_idx + 4]
                if vpath not in dense_labels[level]:
                    dense_labels[level][vpath] = []
                dense_labels[level][vpath].append((int(start), int(end), action_dict[level]['encode'][action]))
                # Add info about the row
                row_info[level][vpath] = row

        return dense_labels, split, action_info, action_dict, row_info

    def sample_all_blocks(self, seq_len, num_seq, video, numFrames=None, maxBlocks=50):
        '''
        sample all indices from a video
        Returns T blocks of num_seq x seq_len indices
        '''

        # Amount of overlap between the blocks
        overlap = self.overlap
        downsample = self.downsample
        totFrames = int(num_seq * seq_len * downsample)
        if numFrames is None:
            numFrames = self.num_video_frames[video]

        jump = 1 - overlap
        # Increase the jump if it becomes too much
        if (numFrames / int(jump * totFrames)) >= maxBlocks:
            jump = (numFrames - 1) / (maxBlocks * totFrames * 1.0)

        sequences = []
        # Randomized initial start point from the first 10% of the video
        # start = random.randint(0, min(int(numFrames * 0.1), max(0, numFrames - totFrames - 1)))
        # Start from the first frame
        start = 0

        while (start + totFrames) <= numFrames:
            sequence_block = np.array(range(start, start + totFrames, downsample)).reshape(num_seq, seq_len)
            sequences.append(sequence_block)
            start += int(jump * totFrames)

        sequences = np.array(sequences)

        return sequences

    def sample_a_block(self, seq_len, num_seq, video, numFrames=None):
        '''
        sample a block indices from a video
        Returns 1 block of num_seq x seq_len indices
        '''

        # Amount of overlap between the blocks
        downsample = self.downsample
        totFrames = int(num_seq * seq_len * downsample)
        if numFrames is None:
            numFrames = self.num_video_frames[video]

        sequences = []

        # Randomized initial start point from the video
        start = random.randint(0, max(0, numFrames - totFrames))

        # Add dynamic sampling in case there are not enough frames
        if ((numFrames - start) - totFrames) <= 0:
            downsample = ((numFrames - start) / (num_seq * seq_len * 1.0)) * 0.95
            totFrames = int(num_seq * seq_len * downsample)

        sequence_block = np.array(np.arange(start, start + totFrames, downsample)).astype(int).reshape(num_seq, seq_len)
        sequences.append(sequence_block)

        sequences = np.array(sequences)

        return sequences

    def __len__(self):
        return len(self.video_names)

    def get_labels_for_sequences(self, sequences, video):
        '''
        Returns video and atomic action labels
        sequences: B x num_seq x seq_len
        return B x 1, B x num_atomic_classes
        '''
        B = sequences.shape[0]

        atomic_labels_list = [[] for i in range(self.num_video_frames[video])]
        for segment in self.dense_labels['atomic'].get(video, []):
            start, end, action = segment
            for j in range(start, end):
                atomic_labels_list[j].append(action)
        video_labels = torch.tensor([self.dense_labels['video'][video][0][-1] for _ in range(B)])

        atomic_labels = torch.zeros((B, self.num_classes['atomic'])).long()
        for idx in range(sequences.shape[0]):
            for frame in range(sequences[idx][0, 0], sequences[idx][-1, -1]):
                for action in atomic_labels_list[frame]:
                    atomic_labels[idx][action] = 1

        return video_labels, atomic_labels

    def get_video_name(self, index):
        return self.video_names[index]

    def get_ego_third_details(self, index):
        ego_path = self.get_video_name(index)
        row = self.row_info['video'][ego_path]
        # Sample a 3rd person view as well
        vpath0, _, vend0, vname, vpath1, _, vend1 = self.get_row_details(row)
        vend = min(vend0, vend1)
        assert ego_path == vpath0, "Mismatch in the path {}, {}".format(ego_path, vpath0)
        return vpath1, vend

    def __getitem__(self, index):
        video = self.get_video_name(index)
        third_person_video, nframes = self.get_ego_third_details(index)

        vals = {}
        sequences = self.sampler(self.seq_len, self.num_seq, video, nframes)

        # Init val with lists
        modes = ["imgs-0", "imgs-1", mu.AudioMode]
        for val in modes:
            if val in self.vals_to_return:
                vals[val] = []

        # Go over all the sequences
        for sequence in sequences:
            idx_block = sequence.reshape(self.seq_len * self.num_seq)
            # 0 stands for the ego-view while 1 stands for the third-person view
            if "imgs-0" in self.vals_to_return:
                img_t_seq0 = self.fetch_imgs_seq(video, self.seq_len, idx_block)
                vals['{}-0'.format(mu.ImgMode)].append(img_t_seq0)
            if "imgs-1" in self.vals_to_return:
                img_t_seq1 = self.fetch_imgs_seq(third_person_video, self.seq_len, idx_block)
                vals['{}-1'.format(mu.ImgMode)].append(img_t_seq1)
            if mu.AudioMode in self.vals_to_return:
                vals[mu.AudioMode].append(self.fetch_audio_seq(video, idx_block))

        for val in modes:
            if val in self.vals_to_return:
                try:
                    vals[val] = torch.stack(vals[val])
                    if self.sampling == 'single':
                        # Squeeze the first singular dim
                        vals[val] = vals[val].squeeze(0)
                except:
                    print(sequences.shape, video)

        if "labels" in self.vals_to_return:
            vals["video_labels"], vals["atomic_labels"] = self.get_labels_for_sequences(sequences, video)
            if self.sampling == 'single':
                for val in ["video_labels", "atomic_labels"]:
                    # Squeeze the first singular dim
                    vals[val] = vals[val].squeeze(0)

        # Add video index field
        vals["vnames"] = torch.LongTensor([index])

        return vals


import unittest


class TestHierarchicalPanasonic(unittest.TestCase):

    @classmethod
    def setUp(self):
        """
        This code code is ran once before all tests.
        """

        self.dataset = HierarchicalPanasonic(
            mode='test',
            transform=mu.get_test_transforms({"img_dim": 128}),
            seq_len=5,
            num_seq=8,
            downsample=3,
            which_split=1,
            vals_to_return=["imgs-0", "audio", "labels"]
        )

    def test_fetch_batch(self):
        for idx in range(25):
            batch = self.dataset[idx]
            B = batch['atomic_labels'].shape[0]
            vname = self.dataset.get_video_name(batch['vnames'][0]).split('/')[-1]
            #print([(k, v.shape) for k, v in batch.items()])
            print('Label counts:', vname, B, torch.unique(batch['atomic_labels'].sum(dim=-1), return_counts=True))


class TestHierarchicalPanasonicSingleSample(unittest.TestCase):

    @classmethod
    def setUp(self):
        """
        This code code is ran once before all tests.
        """

        self.dataset = HierarchicalPanasonic(
            mode='test',
            transform=mu.get_test_transforms({"img_dim": 128}),
            seq_len=5,
            num_seq=8,
            downsample=3,
            which_split=1,
            vals_to_return=["imgs-0", "imgs-1", "audio", "labels"],
            sampling='single',
            debug=True
        )

    def test_fetch_batch(self):
        for idx in range(5):
            batch = self.dataset[idx]
            vname = self.dataset.get_video_name(batch['vnames'][0]).split('/')[-1]
            print([(k, v.shape) for k, v in batch.items()])
            print('Video label counts:', vname, torch.unique(batch['video_labels'], return_counts=True))
            print('Atomic counts:', vname, torch.unique(batch['atomic_labels'].sum(dim=-1), return_counts=True))


if __name__ == '__main__':
    unittest.main()

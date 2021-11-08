import operator
import argparse
import torch
import os
import pickle
import data_utils

import matplotlib.pyplot as plt
import numpy as np
import model_utils as mu
import sim_utils as su
import model_trainer as mt
import dataset_3d as d3d

from tqdm import tqdm
from torch.utils import data

'''
Important components of hierarchical training
1. Collect per second block features for each video
- Checkpoint
2. Create list of classes to train on, etc to extend easily to few shot scenario
3. Add LSTM/Transformer i.e. recurrent net training to perform training easily
4. Add DataLoader to load necessary classes, instances into train, val, test
5. Wrap up training with video level and validation
- Checkpoint
6. Add multi task learning i.e. train for both atomic action type and video level action
'''


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='', type=str, help='save dir for model')
parser.add_argument('--prefix', required=True, type=str, help='prefix')
parser.add_argument('--notes', default='', type=str, help='additional notes')
parser.add_argument('--ckpt_path', required=True, type=str, help='Model ckpt path')
parser.add_argument('--img_dim', required=True, type=int)
parser.add_argument('--modality', required=True, type=str)
parser.add_argument('--num_workers', default=0, type=int)

eps = 1e-3
cuda = torch.device('cuda')
cosSimHandler = su.CosSimHandler()


def get_cross_cos_sim_score(list0, list1):
    return cosSimHandler.get_feature_cross_pair_score(
        cosSimHandler.l2NormedVec(list0), cosSimHandler.l2NormedVec(list1)
    )


def get_instances_for_class(fnames, featuresArr, className):
    idxs = [i for i in range(len(fnames)) if get_class(fnames[i]) == className.lower()]
    return featuresArr[idxs], idxs


def plot_histogram(scores, notes=''):
    scores = scores[scores < 1 - eps]
    plt.hist(scores.flatten().cpu(), bins=50, alpha=0.5)
    plt.ylabel('Cosine Similarity: {}'.format(notes))


def gen_and_plot_cossim_for_class(fnames, featuresArr, className):
    classFets, classFnames = get_instances_for_class(fnames, featuresArr, className)
    classScore = get_cross_cos_sim_score(classFets, classFets)
    plot_histogram(classScore, notes="Class - {}".format(className))


def get_context_representations(model, dataloader, modality):
    '''
    Returns a single context vector for some random sample of a video
    '''
    features = {}
    with torch.no_grad():
        tq = tqdm(dataloader, desc="Progress:")
        for idx, data in enumerate(tq):
            input_seqs = data[modality].to(cuda)
            for input_idx in range(input_seqs.shape[0]):
                video = dataloader.dataset.get_video_name(data['vnames'][0])
                input_seq = input_seqs[input_idx]
                contexts = model.get_representation(input_seq)[0]
                features[video] = {
                    'fets': contexts.cpu().detach(),
                    'video_labels': data['video_labels'],
                    'atomic_labels': data['atomic_labels']
                }
    return features


def get_class(fname):
    fname = fname.rstrip('/')
    return fname.split('/')[-2].lower()


def save_features(features, path_name):
    with open(path_name, 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_features(path_name):
    with open(path_name, 'rb') as handle:
        features = pickle.load(handle)
    return features


def setup_panasonic_model_args(save_dir, restore_ckpt, img_dim=128, modality="imgs-0"):

    dataset = "panasonic-atomic" if "panasonic-atomic" in restore_ckpt else "panasonic"

    parser = mu.get_multi_modal_model_train_args()
    args = parser.parse_args('')

    # Populate dataset and device
    args.dataset = dataset
    args.num_classes = mu.get_num_classes(args.dataset)
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    args.model = "super"
    args.batch_size = 8
    args.img_dim = img_dim
    args.ds = 3
    args.num_seq = 8
    args.seq_len = 5
    args.save_dir = save_dir

    # Populate modalities
    args.modalities = modality
    args.modes = mt.get_modality_list(args.modalities)

    args.losses = ["super"]
    args.num_workers = 6

    if modality == "imgs-0":
        args.imgs_0_restore_ckpt = restore_ckpt
    elif modality == "imgs-1":
        args.imgs_1_restore_ckpt = restore_ckpt
    elif modality == mu.AudioMode:
        args.audio_restore_ckpt = restore_ckpt

    args.restore_ckpts = mt.get_modality_restore_ckpts(args)

    return args


def get_hierarchical_panasonic_dataloader(args, split):
    dataset = d3d.HierarchicalPanasonic(
        mode=split,
        transform=mu.get_test_transforms(args),
        seq_len=args.seq_len,
        num_seq=args.num_seq,
        downsample=3,
        vals_to_return=args.modes + ["labels"],
    )

    data_loader = data.DataLoader(
        dataset,
        sampler=data.SequentialSampler(dataset),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_utils.individual_collate,
        pin_memory=True,
        drop_last=False
    )

    return data_loader


import time
import torch.nn as nn
import torch.optim as optim


class PredictorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device):
        super(PredictorRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.dropout = 0.5
        self.rnn = nn.LSTM(self.input_size, self.hidden_size)
        self.final_fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.num_classes),
        )

        self.device = device

    def forward(self, input: torch.tensor, hidden: torch.tensor):
        B, N, D = input.shape
        output, hidden = self.rnn(input, hidden)
        print(output.shape, hidden.shape)
        return output, hidden

    def initHidden(self, batch):
        return torch.zeros(1, batch, self.hidden_size, device=self.device)


class HierarchicalLearner(nn.Module):

    def __init__(self, args):
        super(HierarchicalLearner, self).__init__()

        self.device = args["device"]
        self.use_rep_loss = args["use_rep_loss"]

        self.predict = PredictorRNN(args["input_size"], args["hidden_size"], args["num_classes"], self.device)

        self.optimizer = optim.Adam(self.predict.parameters(), lr=args["lr"], weight_decay=args["wd"])
        self.teacher_forcing_ratio = args["teacher_forcing_ratio"]

        self.writer_train, self.writer_val = mu.get_writers(args["img_path"])

        self.feature_size = self.predict.hidden_size
        self.print_freq = args["print_freq"]
        self.iteration = 0

        self.rep_criterion_base = nn.CrossEntropyLoss()
        self.rep_criterion = lambda x, y: self.rep_criterion_base(x, y.float().argmax(dim=1))

        self.criterion = nn.MSELoss()

    def prep_data(self, input_seq, target_seq):
        batch, num_seq, seq_len, C, K = input_seq.shape

        input_seq = input_seq.view(batch * num_seq, seq_len, C, K).permute(1, 0, 2, 3)
        target_seq = target_seq.view(batch * num_seq, seq_len, C, K).permute(1, 0, 2, 3)

        encoder_hidden = self.predict.initHidden(batch * num_seq)
        encoder_outputs = torch.zeros(seq_len, batch * num_seq, self.predict.hidden_size, device=self.device)

        return input_seq, target_seq, encoder_hidden, encoder_outputs, batch, num_seq, seq_len

    def get_representation(self, input_seq):

        batch, num_seq, seq_len, C, K = input_seq.shape
        input_seq = input_seq.view(batch * num_seq, seq_len, C, K).permute(1, 0, 2, 3)
        encoder_hidden = self.predict.initHidden(batch * num_seq)

        for ei in range(seq_len):
            encoder_output, encoder_hidden = self.predict(input_seq[ei], encoder_hidden)

        return encoder_hidden.view(batch, num_seq, self.predict.hidden_size).detach()

    def train_step(self, input_seq, label):
        self.optimizer.zero_grad()

        B, N, D = input_seq.shape

        hidden = self.encoder.initHidden(B * N)
        output, hidden = self.predict(input_seq, hidden)

        loss = self.criterion(output, label)
        loss.backward()

        self.optimizer.step()

        return loss

    def val_step(self, input_seq, target_seq, ret_rep=False):

        input_seq, target_seq, encoder_hidden, encoder_outputs, batch, num_seq, seq_len = \
            self.prep_data(input_seq, target_seq)

        loss = 0

        for ei in range(seq_len):
            encoder_output, encoder_hidden = self.predict(input_seq[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output

        decoder_input = input_seq[-1].clone()

        decoder_hidden = encoder_hidden
        representation = encoder_hidden.reshape(batch * num_seq, -1).clone().detach()

        for di in range(seq_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output.detach()  # detach from history as input
            loss += self.criterion(decoder_output, target_seq[di])

        if ret_rep:
            return loss / seq_len, representation
        else:
            return loss / seq_len

    def train_epoch(self, epoch):

        self.train()

        trainX, trainY = [], []
        losses = AverageMeter()

        tq = tqdm(self.train_loader, desc="Training progress in epoch: {}".format(epoch))

        for idx, data in enumerate(tq):
            tic = time.time()

            input_seq = data["poses"]
            input_seq = input_seq.to(self.device)
            B = input_seq.size(0)
            NS = input_seq.size(1)

            target_seq = data["tgt_poses"]
            target_seq = target_seq.to(self.device)

            loss, X = self.train_step(input_seq, target_seq, ret_rep=True)
            losses.update(loss.item(), B)

            trainX.append(X)
            trainY.append(data["labels"].repeat(1, NS).reshape(-1))

            tq.set_postfix({
                "loss_val": losses.val,
                "loss_local_avg": losses.local_avg,
                "T": time.time()-tic
            })

            if idx % self.print_freq == 0:
                self.writer_train.add_scalar('local/loss', losses.val, self.iteration)
                self.iteration += 1

        trainX = torch.cat(trainX)
        trainY = torch.cat(trainY).reshape(-1)

        return losses.local_avg, {"X": trainX, "Y": trainY}

    def val_epoch(self):

        self.eval()

        valX, valY = [], []
        losses = AverageMeter()

        tq = tqdm(self.val_loader, desc="Val progress:")

        for idx, data in enumerate(tq):
            tic = time.time()

            input_seq = data["poses"]
            input_seq = input_seq.to(self.device)
            B = input_seq.size(0)
            NS = input_seq.size(1)

            target_seq = data["tgt_poses"]
            target_seq = target_seq.to(self.device)

            loss, X = self.val_step(input_seq, target_seq, ret_rep=True)
            losses.update(loss.item(), B)

            valX.append(X)
            valY.append(data["labels"].repeat(1, NS).reshape(-1))

            tq.set_postfix({
                "loss_val": losses.val,
                "loss_local_avg": losses.local_avg,
                "T": time.time()-tic
            })

            if idx % self.print_freq == 0:
                self.writer_val.add_scalar('local/loss', losses.val, self.iteration)
                self.iteration += 1

        valX = torch.cat(valX)
        valY = torch.cat(valY).reshape(-1)

        return losses.local_avg, {"X": valX, "Y": valY}


if __name__ == '__main__':
    script_args = parser.parse_args()
    args = setup_panasonic_model_args(
        save_dir=script_args.save_dir,
        restore_ckpt=script_args.ckpt_path,
        img_dim=script_args.img_dim,
        modality=script_args.modality,
    )

    args.data_sources = script_args.modality
    args.num_workers = script_args.num_workers

    datasets = {}

    # Create model and switch to eval mode
    model = mt.get_backbone_for_modality(args, script_args.modality)
    model.eval()

    splits = ['train', 'val', 'test']
    for split in splits:
        data_loader = get_hierarchical_panasonic_dataloader(args, split)
        features = get_context_representations(model, data_loader, script_args.modality)
        datasets[split] = features

    # Create parent file
    save_path = '/{}/nishantr/logs/{}/hierarchical_training_notes{}/features.pickle'.format(
        os.environ['BASE_DIR'], script_args.prefix, script_args.notes
    )
    parent_path = os.path.dirname(save_path)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    save_features(datasets, save_path)

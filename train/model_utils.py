import argparse
from collections import namedtuple

import data_utils
import os


from torch.utils import data
from tensorboardX import SummaryWriter
from torchvision import transforms
from copy import deepcopy
from collections import defaultdict

from dataset_3d import *

sys.path.append('../utils')
from utils import AverageMeter, calc_hamming_loss, calc_mAP

sys.path.append('../backbone')
from resnet_2d3d import neq_load_customized

import torch.nn as nn

# Constants for the framework
eps = 1e-7

CPCLoss = "cpc"
CooperativeLoss = "coop"
SupervisionLoss = "super"
DistillLoss = "distill"
HierarchicalLoss = "hierarchical"
WeighedHierarchicalLoss = "wgt-hier"

# Losses for mode sync
ModeSim = "sim"
AlignLoss = "align"
CosSimLoss = "cossim"
CorrLoss = "corr"
DenseCosSimLoss = "dcssim"
DenseCorrLoss = "dcrr"

# Sets of different losses
LossList = [CPCLoss, CosSimLoss, CorrLoss, DenseCorrLoss, DenseCosSimLoss,
            CooperativeLoss, SupervisionLoss, DistillLoss, HierarchicalLoss, WeighedHierarchicalLoss]
ModeSyncLossList = [CosSimLoss, CorrLoss, DenseCorrLoss, DenseCosSimLoss]

# Type of base model
ModelSSL = 'ssl'
ModelSupervised = 'super'

ImgMode = "imgs"
AudioMode = "audio"
FlowMode = "flow"
FnbFlowMode = "farne"
KeypointHeatmap = "kphm"
SegMask = "seg"
# FIXME: enable multiple views from the same modality
ModeList = [ImgMode, AudioMode, 'imgs-0', 'imgs-1']

ModeParams = namedtuple('ModeParams', ['mode', 'img_fet_dim', 'img_fet_segments', 'final_dim'])


def str2bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]


def str2list(s):
    """Convert string to list of strs, split on _"""
    return s.split('_')


def get_multi_modal_model_train_args():
    parser = argparse.ArgumentParser()

    # General global training parameters
    parser.add_argument('--save_dir', default='', type=str, help='dir to save intermediate results')
    parser.add_argument('--dataset', default='ucf101', type=str)
    parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
    parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
    parser.add_argument('--num_seq', default=8, type=int, help='number of video blocks')
    parser.add_argument('--pred_step', default=3, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--eval-freq', default=1, type=int, help='frequency of evaluation')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for dataloader')
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    parser.add_argument('--notes', default="", type=str, help='Additional notes')
    parser.add_argument('--vis_log_freq', default=100, type=int, help='Visualization frequency')

    # Evaluation specific flags
    parser.add_argument('--ft_freq', default=10, type=int, help='frequency to perform finetuning')

    # Global network and model details. Can be overriden using specific flags
    parser.add_argument('--model', default='super', type=str, help='Options: ssl, super')
    parser.add_argument('--net', default='resnet18', type=str)
    parser.add_argument('--train_what', default='all', type=str)
    parser.add_argument('--img_dim', default=128, type=int)
    parser.add_argument('--sampling', default="dynamic", type=str, help='sampling method (disjoint, random, dynamic)')
    parser.add_argument('--temp', default=0.07, type=float, help='Temperature to use with L2 normalization')
    parser.add_argument('--attention', default=False, type=str2bool, help='Whether to use attention')

    # Knowledge distillation, hierarchical flags
    parser.add_argument('--distill', default=False, type=str2bool, help='Whether to distill knowledge')
    parser.add_argument('--students', default="imgs-0", type=str2list, help='Modalities which are students')
    parser.add_argument('--hierarchical', default=False, type=str2bool, help='Whether to use hierarchical loss')

    # Training specific flags
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--losses', default="cpc", type=str2list, help='Losses to use (CPC, Align, Rep, Sim)')
    parser.add_argument('--dropout', default=0.3, type=float, help='Dropout to use for supervised training')
    parser.add_argument('--tune_bb', default=-1.0, type=float,
                        help='Fine-tune back-bone lr degradation. Useful for pretrained weights. (0.5, 0.1, 0.05)')

    # Hyper-parameters
    parser.add_argument('--msync_wt', default=1.0, type=float, help='Loss weight to use for mode sync loss')
    parser.add_argument('--dot_wt', default=1.0, type=float, help='Dot weight to use for cooperative loss')

    # Multi-modal related flags
    parser.add_argument('--data_sources', default='imgs', type=str2list, help='data sources separated by _')
    parser.add_argument('--modalities', default="imgs", type=str2list, help='Modalitiles to consider. Separate by _')

    # Checkpoint flags
    for m in ModeList:
        parser.add_argument('--{}_restore_ckpt'.format(m), default=None, type=str,
                            help='Restore checkpoint for {}'.format(m))

    # Flags which need not be touched
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
    parser.add_argument('--prefix', default='noprefix', type=str, help='prefix of checkpoint filename')

    # supervision categories
    parser.add_argument('--multilabel_supervision', action='store_true', help='allowing multiple categories in '
                                                                              'supervised training')
    # Extra arguments
    parser.add_argument('--debug', default=False, type=str2bool, help='Reduces latency for data ops')

    # Testing flags
    parser.add_argument('--test', action='store_true', help='Perform testing on the sample')
    parser.add_argument('--test_split', default='test', help='Which split to perform testing on (val, test)')

    # wandb
    parser.add_argument('--wandb_project_name', default="", type=str, help='wandb project name')

    return parser


def get_num_classes(dataset):
    if 'kinetics' in dataset:
        return 400
    elif dataset == 'ucf101':
        return 101
    elif dataset == 'jhmdb':
        return 21
    elif dataset == 'hmdb51':
        return 51
    elif dataset == 'panasonic':
        return 75
    elif dataset == 'panasonic-atomic':
        return 448
    elif dataset == 'hierarchical-panasonic':
        return 75
    else:
        return None


def get_transforms(args):
    return {
        ImgMode: get_imgs_transforms(args),
        AudioMode: get_audio_transforms(args),
    }


def get_test_transforms(args):
    return {
        ImgMode: get_imgs_test_transforms(args),
        AudioMode: get_audio_transforms(args),
    }


def convert_to_dict(args):
    if type(args) != dict:
        args_dict = vars(args)
    else:
        args_dict = args
    return args_dict


def get_imgs_test_transforms(args):
    args_dict = convert_to_dict(args)

    transform = transforms.Compose([
        CenterCrop(size=224, consistent=True),
        Scale(size=(args_dict["img_dim"], args_dict["img_dim"])),
        ToTensor(),
        Normalize()
    ])

    return transform


def get_imgs_transforms(args):

    args_dict = convert_to_dict(args)
    transform = None

    if args_dict["debug"]:
        return transforms.Compose([
            CenterCrop(size=224, consistent=True),
            Scale(size=(args_dict["img_dim"], args_dict["img_dim"])),
            ToTensor(),
            Normalize()
        ])

    if 'panasonic' in args_dict["dataset"]:
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            PadToSize(size=(256, 256)),
            RandomCrop(size=224, consistent=True),
            Scale(size=(args_dict["img_dim"], args_dict["img_dim"])),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])

    return transform


def get_audio_transforms(args):

    return transforms.Compose([transforms.ToTensor()])


def get_writers(img_path):

    try: # old version
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    except: # v1.7
        writer_val = SummaryWriter(logdir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))

    return writer_train, writer_val


def get_dataset_loaders(args, transform, mode='train', test_split=None):
    '''
    test_split is relevant in case of testing, either val or test
    '''

    print('Loading data for "%s" ...' % mode)

    if type(args) != dict:
        args_dict = deepcopy(vars(args))
    else:
        args_dict = args

    if args_dict['debug']:
        orig_mode = mode
        mode = 'train'

    if args_dict['test']:
        if test_split is None:
            test_split = mode
        # Only use the hierarchical panasonic for test
        dataset = HierarchicalPanasonic(
            mode=test_split,
            transform=transform,
            seq_len=args_dict["seq_len"],
            num_seq=args_dict["num_seq"],
            downsample=args_dict["ds"],
            vals_to_return=args_dict['data_sources'].split('_') + ["labels"],
            sampling='all',
        )
    elif args_dict["dataset"] == 'hierarchical-panasonic':
        dataset = HierarchicalPanasonic(
            mode=mode,
            transform=transform,
            seq_len=args_dict["seq_len"],
            num_seq=args_dict["num_seq"],
            downsample=args_dict["ds"],
            vals_to_return=args_dict["data_sources"].split('_'),
            debug=args_dict["debug"],
            sampling='single')
    elif args_dict["dataset"].startswith('panasonic'):
        dataset = Panasonic_3d(
                    mode=mode,
                    transform=transform,
                    seq_len=args_dict["seq_len"],
                    num_seq=args_dict["num_seq"],
                    downsample=args_dict["ds"],
                    vals_to_return=args_dict["data_sources"].split('_'),
                    debug=args_dict["debug"],
                    dataset=args_dict["dataset"].split('-')[0],
                    postfix='atomic' if args_dict["dataset"].endswith('atomic') else '',
                    multilabel_supervision=args_dict["multilabel_supervision"])
    else:
        raise ValueError('dataset not supported')

    val_sampler = data.SequentialSampler(dataset)
    train_sampler = data.RandomSampler(dataset)

    if args_dict["debug"]:
        if orig_mode == 'val':
            train_sampler = data.RandomSampler(dataset, replacement=True, num_samples=100)
        else:
            train_sampler = data.RandomSampler(dataset, replacement=True, num_samples=200)
        val_sampler = data.RandomSampler(dataset)

    data_loader = None
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args_dict["batch_size"],
                                      sampler=train_sampler,
                                      shuffle=False,
                                      num_workers=args_dict["num_workers"],
                                      collate_fn=data_utils.individual_collate,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      sampler=val_sampler,
                                      batch_size=args_dict["batch_size"],
                                      shuffle=False,
                                      num_workers=args_dict["num_workers"],
                                      collate_fn=data_utils.individual_collate,
                                      pin_memory=True,
                                      # Do not change drop last to false, integration issues
                                      drop_last=True)
    elif mode == 'test':
        test_sampler = val_sampler
        data_loader = data.DataLoader(dataset,
                                      sampler=test_sampler,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=args_dict["num_workers"],
                                      collate_fn=data_utils.individual_collate,
                                      pin_memory=True,
                                      # Do not change drop last to false, integration issues
                                      drop_last=True)

    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def set_multi_modal_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        args.modes_str = '_'.join(args.modes)
        args.model_str = 'pred{}'.format(args.pred_step)
        args.loss_str = 'loss_{}'.format('_'.join(args.losses))
        if args.model == ModelSupervised:
            args.model_str = 'supervised'
        exp_path = '{0}/logs/{args.prefix}/{args.dataset}-{args.img_dim}_{1}_' \
                   'bs{args.batch_size}_seq{args.num_seq}_len{args.seq_len}_{args.model_str}_loss-{args.loss_str}' \
                   '_ds{args.ds}_train-{args.train_what}{2}_modes-{args.modes_str}_multilabel_' \
                   '{args.multilabel_supervision}_attention-{args.attention}_hierarchical-{args.hierarchical}_' \
                   '{args.notes}'.format(
                        os.environ['BASE_DIR'],
                        'r%s' % args.net[6::],
                        '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '',
                        args=args
                    )
        exp_path = os.path.join(args.save_dir, exp_path)

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path


def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)


def check_name_to_be_avoided(k):
    # modules_to_avoid = ['.agg.', '.network_pred.']
    modules_to_avoid = []
    for m in modules_to_avoid:
        if m in k:
            return True
    return False


def load_model(model, model_path):
    if os.path.isfile(model_path):
        print("=> loading resumed checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model = neq_load_customized(model, checkpoint['state_dict'])
        print("=> loaded resumed checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
    else:
        print("[WARNING] no checkpoint found at '{}'".format(model_path))
    return model


def get_stats_dict(losses_dict, stats):
    postfix_dict = {}

    def get_short_key(a, b, c):
        return '{}/{}/{}'.format(a, b, c)

    # Populate accuracies
    for loss in stats.keys():
        for mode in stats[loss].keys():
            for stat, meter in stats[loss][mode].items():
                key = get_short_key(loss, mode, str(stat))
                # FIXME: temporary fix
                if stat == 'multilabel_counts':
                    # FIXME: Removing this as it pollutes logs rn. Move this to test instead of val
                    pass
                    # val = meter
                    # postfix_dict[key] = val
                else:
                    val = meter.avg if eval else meter.local_avg
                    postfix_dict[key] = round(val, 3)

    # Populate losses
    for loss in losses_dict.keys():
        for key, meter in losses_dict[loss].items():
            val = meter.avg if eval else meter.local_avg
            postfix_dict[get_short_key('loss', loss, key)] = round(val, 3)

    return postfix_dict


def compute_val_metrics(stats, prefix=None):
    val_stats = dict()
    keys_to_remove = []
    for key in stats.keys():
        if 'multilabel_counts' in key:
            k = '_'.join(key.split('_')[:2]) if prefix is None else prefix
            val_stats['{}_multilabel_hamming_loss'.format(k)] = calc_hamming_loss(stats[key]['tp'], stats[key]['tn'], stats[key]['fp'], stats[key]['fn'])
            val_stats['{}_multilabel_mAP'.format(k)] = calc_mAP(stats[key]['tp'], stats[key]['fp'])
            val_stats['{}_multilabel_acc1'.format(k)] = stats[key]['top1_single'].cpu().data.numpy() / stats[key]['all_single'].cpu().data.numpy()
            val_stats['{}_multilabel_acc3'.format(k)] = stats[key]['top3_single'].cpu().data.numpy() / stats[key]['all_single'].cpu().data.numpy()
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del stats[key]
    stats.update(val_stats)


def shorten_stats(overall_stats):

    def get_short_key(a, b, c):
        return '{}{}{}'.format(a[0], b[0] + b[-1], c[0] + c[-1])

    shortened_stats = {}
    for k, v in overall_stats.items():
        a, b, c = k.split('/')
        # Don't include accuracy 3 stats or align
        shouldInclude = (c[-1] != '3') and (a != 'align')
        if shouldInclude:
            shortened_stats[get_short_key(a, b, c)] = round(v, 2)

    return shortened_stats


def init_loggers(losses):
    losses_dict = defaultdict(lambda: defaultdict(lambda: AverageMeter()))
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: AverageMeter())))
    return losses_dict, stats


class BinaryFocalLossWithLogits(nn.Module):

    def __init__(self, gamma=2, alpha_ratio=10, eps=1e-7):
        super(BinaryFocalLossWithLogits, self).__init__()
        self.gamma = gamma
        self.alpha_ratio = alpha_ratio
        self.eps = eps

    def forward(self, input, y):
        logit = torch.sigmoid(input)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -(y * torch.log(logit) + (1 - y) * torch.log(1 - logit)) # cross entropy
        alpha = self.alpha_ratio * y + (1 - y)
        p_t = logit * y + (1 - logit) * (1 - y)
        loss = alpha * loss * (1 - p_t) ** self.gamma # focal loss

        return loss.mean()
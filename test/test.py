import os
import sys
import time
import argparse
import pickle
import re
import numpy as np
import transform_utils as tu

from tqdm import tqdm
from tensorboardX import SummaryWriter

sys.path.append('../utils')
sys.path.append('../backbone')
from dataset_3d_lc import UCF101_3d, HMDB51_3d
from model_3d_lc import *
from resnet_2d3d import neq_load_customized
from augmentation import *
from utils import AverageMeter, AccuracyTable, ConfusionMeter, save_checkpoint, write_log, calc_topk_accuracy, denorm, calc_accuracy

import torch
import torch.optim as optim
from torch.utils import data
import torch.nn as nn  
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='/data/nishantr/svl/', type=str, help='dir to save intermediate results')
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='lc', type=str)
parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument('--modality', required=True, type=str, help="Modality to use")
parser.add_argument('--split', default=1, type=int)
parser.add_argument('--seq_len', default=5, type=int)
parser.add_argument('--num_seq', default=8, type=int)
parser.add_argument('--num_class', default=101, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--ds', default=3, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--pretrain', default='random', type=str)
parser.add_argument('--test', default='', type=str)
parser.add_argument('--extensive', default=0, type=int)
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', default=5, type=int)
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--train_what', default='last', type=str, help='Train what parameters?')
parser.add_argument('--prefix', default='tmp', type=str)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--full_eval_freq', default=10, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--notes', default='', type=str)

parser.add_argument('--ensemble', default=0, type=int)
parser.add_argument('--prob_imgs', default='', type=str)
parser.add_argument('--prob_flow', default='', type=str)
parser.add_argument('--prob_seg', default='', type=str)
parser.add_argument('--prob_kphm', default='', type=str)


def get_data_loader(args, mode='train'):
    print("Getting data loader for:", args.modality)
    transform = None
    if mode == 'train':
        transform = tu.get_train_transforms(args)
    elif mode == 'val':
        transform = tu.get_val_transforms(args)
    elif mode == 'test':
        transform = tu.get_test_transforms(args)
    loader = get_data(transform, mode)
    return loader


def get_num_channels(modality):
    if modality == mu.ImgMode:
        return 3
    elif modality == mu.FlowMode:
        return 2
    elif modality == mu.FnbFlowMode:
        return 2
    elif modality == mu.KeypointHeatmap:
        return 17
    elif modality == mu.SegMask:
        return 1
    else:
        assert False, "Invalid modality: {}".format(modality)


def freeze_backbone(model):
    print('Freezing the backbone...')
    for name, param in model.module.named_parameters():
        if ('resnet' in name) or ('rnn' in name) or ('agg' in name):
            param.requires_grad = False
    return model


def unfreeze_backbone(model):
    print('Unfreezing the backbone...')
    for name, param in model.module.named_parameters():
        if ('resnet' in name) or ('rnn' in name):
            param.requires_grad = True
    return model


def main():
    global args; args = parser.parse_args()
    global device; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'ucf101': args.num_class = 101
    elif args.dataset == 'hmdb51': args.num_class = 51

    if args.ensemble:
        def read_pkl(fname):
            if fname == '':
                return None
            with open(fname, 'rb') as f:
                prob = pickle.load(f)
            return prob
        ensemble(read_pkl(args.prob_imgs), read_pkl(args.prob_flow), read_pkl(args.prob_seg), read_pkl(args.prob_kphm))
        sys.exit()

    args.in_channels = get_num_channels(args.modality)

    ### classifier model ###
    if args.model == 'lc':
        model = LC(sample_size=args.img_dim, 
                   num_seq=args.num_seq, 
                   seq_len=args.seq_len, 
                   in_channels=args.in_channels,
                   network=args.net,
                   num_class=args.num_class,
                   dropout=args.dropout)
    else:
        raise ValueError('wrong model!')

    model = nn.DataParallel(model)
    model = model.to(device)
    global criterion; criterion = nn.CrossEntropyLoss()
    
    ### optimizer ### 
    params = None
    if args.train_what == 'ft':
        print('=> finetune backbone with smaller lr')
        params = []
        for name, param in model.module.named_parameters():
            if ('resnet' in name) or ('rnn' in name):
                params.append({'params': param, 'lr': args.lr/10})
            else:
                params.append({'params': param})
    elif args.train_what == 'freeze':
        print('=> Freeze backbone')
        params = []
        for name, param in model.module.named_parameters():
            param.requires_grad = False
    else:
        pass # train all layers
    
    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            print(name, param.requires_grad)
    print('=================================\n')

    if params is None:
        params = model.parameters()

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    # Old version
    # if args.dataset == 'hmdb51':
    #     lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[50,70,90], repeat=1)
    # elif args.dataset == 'ucf101':
    #     if args.img_dim == 224: lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[90,140,180], repeat=1)
    #     else: lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[50, 70, 90], repeat=1)
    if args.img_dim == 224:
        lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[60,120,180], repeat=1)
    else:
        lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[50, 70, 90], repeat=1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    args.old_lr = None
    best_acc = 0
    global iteration; iteration = 0
    global num_epoch; num_epoch = 0

    ### restart training ###
    if args.test:
        if os.path.isfile(args.test):
            print("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test)
            try: model.load_state_dict(checkpoint['state_dict'])
            except:
                print('=> [Warning]: weight structure is not equal to test model; Use non-equal load ==')
                model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded testing checkpoint '{}' (epoch {})".format(args.test, checkpoint['epoch']))
            num_epoch = checkpoint['epoch']
        elif args.test == 'random':
            print("=> [Warning] loaded random weights")
        else: 
            raise ValueError()

        test_loader = get_data_loader(args, 'test')
        test_loss, test_acc = test(test_loader, model, extensive=args.extensive)
        sys.exit()
    else: # not test
        torch.backends.cudnn.benchmark = True

    if args.resume:
        if os.path.isfile(args.resume):
            # args.old_lr = float(re.search('_lr(.+?)_', args.resume).group(1))
            args.old_lr = 1e-3
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.reset_lr: # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
            else: print('==== Change lr from %f to %f ====' % (args.old_lr, args.lr))
            iteration = checkpoint['iteration']
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if (not args.resume) and args.pretrain:
        if args.pretrain == 'random':
            print('=> using random weights')
        elif os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    ### load data ###
    train_loader = get_data_loader(args, 'train')
    val_loader = get_data_loader(args, 'val')
    test_loader = get_data_loader(args, 'test')

    # setup tools
    global de_normalize; de_normalize = denorm()
    global img_path; img_path, model_path = set_path(args)
    global writer_train
    try: # old version
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    except: # v1.7
        writer_val = SummaryWriter(logdir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))

    args.test = model_path
    print("Model path:", model_path)

    # Freeze the model backbone initially
    model = freeze_backbone(model)
    cooldown = 0

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        num_epoch = epoch

        train_loss, train_acc = train(train_loader, model, optimizer, epoch)
        val_loss, val_acc = validate(val_loader, model)
        scheduler.step(epoch)

        writer_train.add_scalar('global/loss', train_loss, epoch)
        writer_train.add_scalar('global/accuracy', train_acc, epoch)
        writer_val.add_scalar('global/loss', val_loss, epoch)
        writer_val.add_scalar('global/accuracy', val_acc, epoch)

        # save check_point
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        # Perform testing if either the frequency is hit or the model is the best after a few epochs
        if (epoch + 1) % args.full_eval_freq == 0:
            test(test_loader, model)
        elif (epoch > 70) and (cooldown >= 5) and is_best:
            test(test_loader, model)
            cooldown = 0
        else:
            cooldown += 1

        save_checkpoint(
            state={
                'epoch': epoch+1,
                'net': args.net,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'iteration': iteration
            },
            mode=args.modality,
            is_best=is_best,
            gap=5,
            filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch+1)),
            keep_all=False)

        # Unfreeze the model backbone after the first run
        if epoch == (args.start_epoch):
            model = unfreeze_backbone(model)
    
    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
    print("Model path:", model_path)


def train(data_loader, model, optimizer, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    model.train()
    global iteration

    tq = tqdm(data_loader, desc="Train: Ep {}".format(epoch))

    for idx, (input_seq, target, _) in enumerate(tq):
        tic = time.time()
        input_seq = input_seq.to(device)
        target = target.to(device)
        B = input_seq.size(0)
        output, _ = model(input_seq)

        # visualize
        if (iteration == 0) or (iteration == args.print_freq):
            if B > 2: input_seq = input_seq[0:2,:]
            writer_train.add_image('input_seq', 
                                   de_normalize(vutils.make_grid(
                                       input_seq[:, :3, ...].transpose(2,3).contiguous().view(-1,3,args.img_dim,args.img_dim),
                                       nrow=args.num_seq*args.seq_len)), 
                                   iteration)
        del input_seq

        [_, N, D] = output.size()
        output = output.view(B*N, D)
        target = target.repeat(1, N).view(-1)

        loss = criterion(output, target)
        acc = calc_accuracy(output, target)

        del target 

        losses.update(loss.item(), B)
        accuracy.update(acc.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_weight = 0.0
        decay_weight = 0.0
        for m in model.parameters():
            if m.requires_grad: decay_weight += m.norm(2).data
            total_weight += m.norm(2).data

        tq_stats = {
            'loss': losses.local_avg,
            'acc': accuracy.local_avg,
            'decay_wt': decay_weight.item(),
            'total_wt': total_weight.item(),
        }

        tq.set_postfix(tq_stats)

        if idx % args.print_freq == 0:
            writer_train.add_scalar('local/loss', losses.val, iteration)
            writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

            iteration += 1

    return losses.local_avg, accuracy.local_avg


def validate(data_loader, model):
    losses = AverageMeter()
    accuracy = AverageMeter()
    model.eval()
    with torch.no_grad():
        tq = tqdm(data_loader,  desc="Val: ")
        for idx, (input_seq, target, _) in enumerate(tq):
            input_seq = input_seq.to(device)
            target = target.to(device)
            B = input_seq.size(0)
            output, _ = model(input_seq)

            [_, N, D] = output.size()
            output = output.view(B*N, D)
            target = target.repeat(1, N).view(-1)

            loss = criterion(output, target)
            acc = calc_accuracy(output, target)

            losses.update(loss.item(), B)
            accuracy.update(acc.item(), B)

            tq.set_postfix({
                'loss': losses.avg,
                'acc': accuracy.avg,
            })
                
    print('Val - Loss {loss.avg:.4f}\t'
          'Acc: {acc.avg:.4f} \t'.format(loss=losses, acc=accuracy))
    return losses.avg, accuracy.avg


def test(data_loader, model, extensive=False):
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    acc_table = AccuracyTable(data_loader.dataset.action_dict_decode)
    confusion_mat = ConfusionMeter(args.num_class)
    probs = {}

    model.eval()
    with torch.no_grad():
        tq = tqdm(data_loader,  desc="Test: ")
        for idx, (input_seq, target, index) in enumerate(tq):
            input_seq = input_seq.to(device)
            target = target.to(device)
            B = input_seq.size(0)
            input_seq = input_seq.squeeze(0) # squeeze the '1' batch dim
            output, _ = model(input_seq)
            del input_seq

            prob = torch.mean(torch.mean(nn.functional.softmax(output, 2), 0), 0, keepdim=True)
            top1, top5 = calc_topk_accuracy(prob, target, (1,5))
            acc_top1.update(top1.item(), B)
            acc_top5.update(top5.item(), B)
            del top1, top5

            output = torch.mean(torch.mean(output, 0), 0, keepdim=True)
            loss = criterion(output, target.squeeze(-1))

            losses.update(loss.item(), B)
            del loss

            _, pred = torch.max(output, 1)
            confusion_mat.update(pred, target.view(-1).byte())
            acc_table.update(pred, target)
            probs[index] = {'prob': prob.detach().cpu(), 'target': target.detach().cpu()}

            tq.set_postfix({
                'loss': losses.avg,
                'acc1': acc_top1.avg,
                'acc5': acc_top5.avg,
            })

    print('Test - Loss {loss.avg:.4f}\t'
          'Acc top1: {top1.avg:.4f} Acc top5: {top5.avg:.4f} \t'.format(loss=losses, top1=acc_top1, top5=acc_top5))
    confusion_mat.plot_mat(args.test+'.svg')
    write_log(content='Loss {loss.avg:.4f}\t Acc top1: {top1.avg:.4f} Acc top5: {top5.avg:.4f} \t'.format(loss=losses, top1=acc_top1, top5=acc_top5, args=args),
              epoch=num_epoch,
              filename=os.path.join(os.path.dirname(args.test), 'test_log_{}.md').format(args.notes))
    with open(os.path.join(os.path.dirname(args.test), 'test_probs_{}.pkl').format(args.notes), 'wb') as f:
        pickle.dump(probs, f)

    if extensive:
        acc_table.print_table()
        acc_table.print_dict()

    # import ipdb; ipdb.set_trace()
    return losses.avg, [acc_top1.avg, acc_top5.avg]


def ensemble(prob_imgs=None, prob_flow=None, prob_seg=None, prob_kphm=None):
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    probs = [prob_imgs, prob_flow, prob_seg, prob_kphm]
    for idx in range(len(probs)):
        if probs[idx] is not None:
            probs[idx] = {k[0][0].data: v for k, v in probs[idx].items()}
    valid_probs = [x for x in probs if x is not None]
    weights = [2, 2, 1, 1]

    ovr_probs = {}
    for k in valid_probs[0].keys():
        ovr_probs[k] = valid_probs[0][k]['prob'] * 0.0
        total = 0
        for idx in range(len(probs)):
            p = probs[idx]
            if p is not None:
                total += weights[idx]
                ovr_probs[k] += p[k]['prob'] * weights[idx]
        ovr_probs[k] /= total

        top1, top5 = calc_topk_accuracy(ovr_probs[k], valid_probs[0][k]['target'], (1, 5))
        acc_top1.update(top1.item(), 1)
        acc_top5.update(top5.item(), 1)

    print('Test - Acc top1: {top1.avg:.4f} Acc top5: {top5.avg:.4f} \t'.format(top1=acc_top1, top5=acc_top5))


def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)
    global dataset
    if args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode, 
                         transform=transform, 
                         seq_len=args.seq_len,
                         num_seq=args.num_seq,
                         downsample=args.ds,
                         which_split=args.split,
                         modality=args.modality
                    )
    elif args.dataset == 'hmdb51':
        dataset = HMDB51_3d(mode=mode, 
                         transform=transform, 
                         seq_len=args.seq_len,
                         num_seq=args.num_seq,
                         downsample=args.ds,
                         which_split=args.split,
                         modality=args.modality
                    )
    else:
        raise ValueError('dataset not supported')
    my_sampler = data.RandomSampler(dataset)
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log/{args.prefix}/ft_{args.dataset}-{args.img_dim}_mode-{args.modality}_' \
                   'sp{args.split}_{0}_{args.model}_bs{args.batch_size}_' \
                   'lr{1}_wd{args.wd}_ds{args.ds}_seq{args.num_seq}_len{args.seq_len}_' \
                   'dp{args.dropout}_train-{args.train_what}{2}'.format(
                        'r%s' % args.net[6::],
                        args.old_lr if args.old_lr is not None else args.lr,
                        '_'+args.notes,
                        args=args)
        exp_path = os.path.join(args.save_dir, exp_path)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path


def MultiStepLR_Restart_Multiplier(epoch, gamma=0.1, step=[10,15,20], repeat=3):
    '''return the multipier for LambdaLR, 
    0  <= ep < 10: gamma^0
    10 <= ep < 15: gamma^1 
    15 <= ep < 20: gamma^2
    20 <= ep < 30: gamma^0 ... repeat 3 cycles and then keep gamma^2'''
    max_step = max(step)
    effective_epoch = epoch % max_step
    if epoch // max_step >= repeat:
        exp = len(step) - 1
    else:
        exp = len([i for i in step if effective_epoch>=i])
    return gamma ** exp


if __name__ == '__main__':
    main()

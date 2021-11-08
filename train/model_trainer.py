import sys
import time
import os
import torch
import random
import sklearn

# Ignore sklearn related warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import torch.nn as nn
import torchvision.utils as vutils
import torch.optim as optim
import finetune_utils as ftu
import model_utils as mu
import model_3d as m3d
import sim_utils as su
import mask_utils as masku
import wandb
os.environ['WANDB_DIR'] = os.environ['BASE_DIR']

sys.path.append('../backbone')
from model_3d import DpcRnn, SupervisedDpcRnn
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

sys.path.append('../utils')
from utils import AverageMeter, AccuracyTable, ConfusionMeter, save_checkpoint, \
                  denorm, calc_topk_accuracy, calc_per_class_multilabel_counts


def get_modality_list(modalities):
    modes = []
    for m in mu.ModeList:
        if m in modalities:
            modes.append(m)
    return modes


def get_modality_restore_ckpts(args):
    ckpts = {}
    for m in args.modes:
        # Replacing - with _ because of the way parser works
        ckpt = getattr(args, m.replace('-', '_') + "_restore_ckpt")
        ckpts[m] = ckpt
        if ckpt is not None:
            print("Mode: {} is being restored from: {}".format(m, ckpt))
    return ckpts


class ModalitySyncer(nn.Module):

    def get_feature_extractor_based_on_mode(self, mode_params):
        if mode_params.mode.split('-')[0] in [mu.ImgMode, mu.AudioMode]:
            return m3d.ImageFetCombiner(mode_params.img_fet_dim, mode_params.img_fet_segments)
        else:
            assert False, "Invalid mode provided: {}".format(mode_params)

    def __init__(self, args):
        super(ModalitySyncer, self).__init__()

        self.losses = args["losses"]

        self.mode0_dim = args["mode_0_params"].final_dim
        self.mode1_dim = args["mode_1_params"].final_dim
        self.mode0_fet_extractor = self.get_feature_extractor_based_on_mode(args["mode_0_params"])
        self.mode1_fet_extractor = self.get_feature_extractor_based_on_mode(args["mode_1_params"])

        self.instance_mask = args["instance_mask"]

        self.common_dim = min(self.mode0_dim, self.mode1_dim) // 2
        # input is B x dim0, B x dim1
        self.mode1_to_common = nn.Sequential()
        self.mode0_to_common = nn.Sequential()

        self.mode_losses = [mu.DenseCosSimLoss]

        self.simHandler = nn.ModuleDict(
            {
                mu.AlignLoss: su.AlignSimHandler(self.instance_mask),
                mu.CosSimLoss: su.CosSimHandler(),
                mu.CorrLoss: su.CorrSimHandler(),
                mu.DenseCorrLoss: su.DenseCorrSimHandler(self.instance_mask),
                mu.DenseCosSimLoss: su.DenseCosSimHandler(self.instance_mask),
            }
        )

        # Perform initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_total_loss(self, mode0, mode1):
        loss = torch.tensor(0.0).to(mode0.device)

        stats = {}
        for lossKey in self.mode_losses:
            lossVal0, stat = self.simHandler[lossKey](mode0, mode1)
            lossVal1, _ = self.simHandler[lossKey](mode1, mode0)
            lossVal = (lossVal0 + lossVal1) * 0.5
            loss += lossVal
            stats[lossKey] = stat

        return loss, stats

    def forward(self, input0, input1):
        assert len(input0.shape) == 5, "{}".format(input0.shape)
        # inputs are B, N, D, S, S
        y0 = self.mode0_fet_extractor(input0)
        y1 = self.mode1_fet_extractor(input1)

        # outputs are B * N, dim
        B, N, _ = y1.shape
        y0_in_space_common = self.mode0_to_common(y0.view(B * N, -1)).view(B, N, -1)
        y1_in_space_common = self.mode1_to_common(y1.view(B * N, -1)).view(B, N, -1)

        return self.get_total_loss(y0_in_space_common, y1_in_space_common)


class SupervisedModalitySyncer(ModalitySyncer):

    def __init__(self, args):
        super(SupervisedModalitySyncer, self).__init__(args)

        # input is B x dim0, B x dim1
        self.mode1_to_common = m3d.NonLinearProjection(args["mode_0_params"].img_fet_dim, self.common_dim)
        self.mode0_to_common = m3d.NonLinearProjection(args["mode_1_params"].img_fet_dim, self.common_dim)

        self.mode_losses = [mu.AlignLoss]

    def forward(self, input0, input1):
        # input is B, N, D or B, 1, D
        assert len(input0.shape) == 3, "{}".format(input0.shape)

        B, _, D = input0.shape
        # outputs are B, dim
        y0_in_space_common = self.mode0_to_common(input0.view(B, -1)).view(B, 1, -1)
        y1_in_space_common = self.mode1_to_common(input1.view(B, -1)).view(B, 1, -1)

        loss, stats = self.get_total_loss(y0_in_space_common, y1_in_space_common)

        return loss, stats, None


class AttentionModalitySyncer(SupervisedModalitySyncer):

    def __init__(self, args):
        super(AttentionModalitySyncer, self).__init__(args)

        # input is B x N x dim0 x s x s, B x 1 x dim1
        self.mode0_attention_fn = m3d.AttentionProjection(
            args["mode_0_params"].img_fet_dim,
            (args["mode_0_params"].img_fet_segments, args["mode_0_params"].img_fet_segments))
        self.mode1_attention_fn = m3d.AttentionProjection(
            args["mode_1_params"].img_fet_dim,
            (args["mode_1_params"].img_fet_segments, args["mode_1_params"].img_fet_segments))

        self.mode_losses = [mu.AlignLoss]

    def forward(self, input0, input1):
        # input0, input1 is B, N, D/D', S, S
        assert len(input0.shape) == 5, "{}".format(input0.shape)
        assert len(input1.shape) == 5, "{}".format(input1.shape)

        B, N, D, S, S = input0.shape
        y0, attn0 = self.mode0_attention_fn.applyAttention(input0)
        y1, attn1 = self.mode1_attention_fn.applyAttention(input1)

        # outputs are B, dim
        y0_in_space_common = self.mode0_to_common(y0.view(B, -1)).view(B, 1, -1)
        y1_in_space_common = self.mode1_to_common(y1.view(B, -1)).view(B, 1, -1)

        loss, stats = self.get_total_loss(y0_in_space_common, y1_in_space_common)

        return loss, stats, {'0': attn0, '1': attn1}


class MultiModalModelTrainer(nn.Module):

    def addImgGrid(self, data):
        '''
        Plots image frames in different subsections
        '''
        # shape of data[mode] batch, num_seq, seq_len, IC, IH, IW
        for mode in self.modes:
            IC = data[mode].shape[3]
            if IC == 3:
                images = data[mode].detach().cpu()[:, 0, 0, ...]
            else:
                # Plot the summation instead
                images = data[mode].detach().cpu()[:, 0, 0, ...].mean(dim=1, keepdim=True)
            # Shape is batch, IC, IH, IW
            grid = vutils.make_grid(images, nrow=int(np.sqrt(images.shape[0])))
            self.writer_train.add_image('images/frames/{}'.format(mode), grid, 0)
            if self.use_wandb:
                wandb.log({'images/frames/{}'.format(mode): [wandb.Image(grid)]}, step=self.iteration)

    def addAttnGrid(self, attn):
        '''
        Plots attention frames in different modalities; can be directly compared to added imgs
        '''
        # shape of data[mode], batch, 1, 1, S0, S1
        for key in attn.keys():
            images = attn[key].detach().cpu()[:, 0, 0, ...].unsqueeze(1)
            # Shape is batch, 1, S, S
            grid = vutils.make_grid(images, nrow=int(np.sqrt(images.shape[0])))
            self.writer_train.add_image('attn/map/{}'.format(key), grid, 0)
            if self.use_wandb:
                wandb.log({'attn/map/{}'.format(key): [wandb.Image(grid)]}, step=self.iteration)

    def get_modality_feature_extractor(self, final_feature_size, last_size, mode):
        if mode.split('-')[0] in [mu.ImgMode, mu.AudioMode]:
            return m3d.ImageFetCombiner(final_feature_size, last_size)
        else:
            assert False, "Invalid mode provided: {}".format(mode)

    def __init__(self, args):
        super(MultiModalModelTrainer, self).__init__()

        self.args = args

        self.modes = args["modalities"]
        self.model_type = args["model"]
        self.models = nn.ModuleDict(args["models"])
        self.losses = args["losses"]
        self.num_classes = args["num_classes"]
        self.data_sources = args["data_sources"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention = args['attention']
        # Gradient accumulation step interval
        self.grad_step_interval = 4

        self.vis_log_freq = args["vis_log_freq"]

        # Model log writers
        self.img_path = args["img_path"]
        self.model_path = args["model_path"]
        self.model_name = self.model_path.split('/')[-2]
        self.writer_train, self.writer_val = mu.get_writers(self.img_path)

        # wandb
        self.use_wandb = args["wandb_project_name"] != ""
        if self.use_wandb:
            wandb.init(project=args["wandb_project_name"])
            wandb.run.name = self.model_name
            wandb.run.save()
            wandb.watch(list(self.models.values()))

        print("Model path is:", self.model_path, self.img_path)

        # multilabel training for supervision
        self.multilabel_supervision = args["multilabel_supervision"]

        self.shouldFinetune = args["ft_freq"] > 0
        self.finetuneSkip = 1 if not self.multilabel_supervision else 5

        transform = mu.get_transforms(args)
        if not args['test']:
            self.train_loader = mu.get_dataset_loaders(args, transform, 'train')
            self.val_loader = mu.get_dataset_loaders(args, transform, 'val')

        if args['test']:
            test_transform = mu.get_test_transforms(args)
            self.test_loader = mu.get_dataset_loaders(args, test_transform, 'test', test_split=args['test_split'])

        self.num_classes = args["num_classes"]
        self.l2_norm = True
        self.temp = args["temp"] if self.l2_norm else 1.0

        self.common_dim = 128
        self.cpc_projections = nn.ModuleDict({m: nn.Sequential() for m in self.modes})

        self.denormalize = denorm()

        self.val_criterion_base = nn.CrossEntropyLoss()
        self.val_criterion = lambda x, y: self.val_criterion_base(x, y.float().argmax(dim=1))

        self.criteria = {
            mu.CPCLoss: self.val_criterion,
            mu.CooperativeLoss: nn.L1Loss(),
            mu.SupervisionLoss: nn.CrossEntropyLoss() if not self.multilabel_supervision else
            mu.BinaryFocalLossWithLogits(),
            mu.HierarchicalLoss: mu.BinaryFocalLossWithLogits(),
            mu.WeighedHierarchicalLoss: {m: m3d.WeighedLoss(num_losses=2, device=self.device) for m in self.modes},
        }

        self.CooperativeLossLabel = mu.CooperativeLoss

        print('Modes being used:', self.modes)
        self.compiled_features = {m: self.get_modality_feature_extractor(
            self.models[m].final_feature_size, self.models[m].last_size, m) for m in self.modes}
        self.interModeDotHandler = su.InterModeDotHandler(last_size=None)

        self.B0 = self.B1 = self.args["batch_size"]

        self.standard_grid_mask = {
            m: masku.process_mask(
                masku.get_standard_grid_mask(
                    self.B0,
                    self.B1,
                    self.args["pred_step"],
                    self.models[m].last_size,
                    device=self.device
                )
            ) for m in self.modes
        }
        self.standard_instance_mask = masku.process_mask(
            masku.get_standard_instance_mask(self.B0, self.B1, self.args["pred_step"], device=self.device)
        )
        self.standard_video_level_mask = masku.process_mask(
            torch.eye(self.B0, device=self.device).reshape(self.B0, 1, self.B0, 1)
        )

        self.modeSyncers = nn.ModuleDict()

        self.mode_pairs = [(m0, m1) for m0 in self.modes for m1 in self.modes if m0 < m1]

        self.sync_wt = self.args["msync_wt"]

        for (m0, m1) in self.mode_pairs:
            num_seq = self.args["num_seq"]
            mode_align_args = {
                "losses": self.losses,
                "mode_0_params": self.get_mode_params(m0),
                "mode_1_params": self.get_mode_params(m1),
                "dim_layer_1": 64,
                "instance_mask": None,
            }
            # Have to explicitly send these to the GPU as they're present in a dict
            modality_syncer = ModalitySyncer
            if self.model_type == mu.ModelSupervised:
                mode_align_args['instance_mask'] = self.standard_video_level_mask
                if self.attention:
                    modality_syncer = AttentionModalitySyncer
                else:
                    modality_syncer = SupervisedModalitySyncer
            else:
                instance_mask_m0_m1 = masku.process_mask(
                    masku.get_standard_instance_mask(self.B0, self.B1, num_seq, self.device)
                )
                mode_align_args['instance_mask'] = instance_mask_m0_m1
            self.modeSyncers[self.get_tuple_name(m0, m1)] = modality_syncer(mode_align_args).to(self.device)

        print("[NOTE] Losses used: ", self.losses)

        self.cosSimHandler = su.CosSimHandler()
        self.dot_wt = args["dot_wt"]

        # Use a smaller learning rate if the backbone is already trained
        degradeBackboneLR = 1.0
        if args["tune_bb"] > 0:
            degradeBackboneLR = args["tune_bb"]

        backboneLr = {
            'params':
                [p for k in self.models.keys() for p in self.models[k].parameters()],
            'lr': args["lr"] * degradeBackboneLR
        }
        miscLr = {
            'params': [p for model in list(self.modeSyncers.values()) for p in model.parameters()],
            'lr': args["lr"]
        }

        self.optimizer = optim.Adam(
            [miscLr, backboneLr],
            lr=args["lr"],
            weight_decay=args["wd"]
        )

        patience = 10
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, verbose=True, patience=patience, min_lr=1e-5
        )

        self.model_finetuner = ftu.QuickSupervisedModelTrainer(self.num_classes, self.modes)

        self.iteration = 0
        self.accuracyKList = [1, 3]

        self.init_knowledge_distillation()
        self.init_hierarchical()

    def init_hierarchical(self):
        self.hierarchical = self.args['hierarchical']

    def init_knowledge_distillation(self):
        self.kd_weight = 0.1
        self.temperature = 2.5
        self.distill = self.args['distill']
        if self.distill:
            self.student_modes = self.args['student_modes']
        else:
            self.student_modes = self.modes

        for mode in self.modes:
            if mode not in self.student_modes:
                self.models[mode].eval()
                # Freeze the models if they are not students
                for param in self.models[mode].parameters():
                    param.requires_grad = False

    @staticmethod
    def get_tuple_name(m0, m1):
        return "{}|{}".format(m0[0] + m0[-1], m1[0] + m1[-1])

    def get_mode_params(self, mode):
        if mode.split('-')[0] in [mu.ImgMode, mu.AudioMode]:
            return mu.ModeParams(
                mode,
                self.models[mode].param['feature_size'],
                self.models[mode].last_size,
                self.models[mode].param['feature_size']
            )
        else:
            assert False, "Incorrect mode: {}".format(mode)

    def get_feature_pair_score(self, pred_features, gt_features):
        """
            (pred/gt)features: [B, N, D, S, S]
            Special case for a few instances would be with S=1
            Returns 6D pair score tensor
        """
        B1, N1, D1, S1, S1 = pred_features.shape
        B2, N2, D2, S2, S2 = gt_features.shape
        assert (D1, S1) == (D2, S2), \
            "Mismatch between pred and gt features: {}, {}".format(pred_features.shape, gt_features.shape)

        # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT.
        preds = pred_features.permute(0, 1, 3, 4, 2).contiguous().view(B1 * N1 * S1 * S1, D1) / self.temp
        gts = gt_features.permute(0, 1, 3, 4, 2).contiguous().view(B2 * N2 * S2 * S2, D2).transpose(0, 1) / self.temp

        # Get the corresponding scores of each region in the matrix with each other region i.e.
        # total last_size ** 4 combinations. Note that we have pred_step ** 2 such pairs as well
        score = torch.matmul(preds, gts).view(B1, N1, S1*S1, B2, N2, S2*S2)

        return score

    def log_visuals(self, input_seq, mode):
        if input_seq.size(0) > 5:
            input_seq = input_seq[:5]

        ic = 3
        if mode.split('-')[0] in [mu.AudioMode]:
            denormed_img = vutils.make_grid(input_seq)
        else:
            if mode.split('-')[0] not in [mu.ImgMode, mu.AudioMode]:
                assert input_seq.shape[2] in [1, 2, 3, 17], "Invalid shape: {}".format(input_seq.shape)
                input_seq = torch.abs(input_seq)
                input_seq = input_seq.sum(dim=2, keepdim=True)
                input_seq = input_seq / 0.25
                input_seq[input_seq > 1] = 1.0
                input_seq[input_seq != input_seq] = 0.0
                ic = 1

            img_dim = input_seq.shape[-1]
            assert img_dim in [64, 128, 224], "imgs_dim: {}, input_seq: {}".format(img_dim, input_seq.shape)
            grid_img = vutils.make_grid(
                input_seq.transpose(2, 3).contiguous().view(-1, ic, img_dim, img_dim),
                nrow=self.args["num_seq"] * self.args["seq_len"]
            )

            if mode.startswith("imgs"):
                denormed_img = self.denormalize(grid_img)
            else:
                denormed_img = grid_img

        self.writer_train.add_image('input_seq/{}'.format(mode), denormed_img, self.iteration)
        if self.use_wandb:
            wandb.log({'input_seq/{}'.format(mode): [wandb.Image(denormed_img)]}, step=self.iteration)

    def log_metrics(self, losses_dict, stats, writer, prefix):
        for loss in self.losses:
            for mode in losses_dict[loss].keys():
                val = losses_dict[loss][mode].val if hasattr(losses_dict[loss][mode], 'val') else losses_dict[loss][mode]
                writer.add_scalar(
                    prefix + '/losses/' + loss + '/' + str(mode),
                    val,
                    self.iteration
                )
                if self.use_wandb:
                    wandb.log({prefix + '/losses/' + loss + '/' + str(mode): val}, step=self.iteration)
        for loss in stats.keys():
            for mode in stats[loss].keys():
                for stat in stats[loss][mode].keys():
                    val = stats[loss][mode][stat].val if hasattr(stats[loss][mode][stat], 'val') else stats[loss][mode][stat]
                    writer.add_scalar(
                        prefix + '/stats/' + loss + '/' + str(mode) + '/' + str(stat),
                        val,
                        self.iteration
                    )
                    if self.use_wandb:
                        wandb.log({prefix + '/stats/' + loss + '/' + str(mode) + '/' + str(stat): val}, step=self.iteration)

    def perform_self_supervised_forward_passes(self, data, feature_dict):

        NS = self.args["pred_step"]
        pred_features, gt_all_features, agg_features, probabilities = {}, {}, {}, {}
        flat_scores = {}

        if self.shouldFinetune:
            feature_dict['Y'].append(data['labels'][::self.finetuneSkip])

        for mode in self.modes:
            if not mode in data.keys():
                continue

            B = data[mode].shape[0]
            input_seq = data[mode].to(self.device)
            # assert input_seq.shape[0] == self.args["batch_size"]
            SQ = self.models[mode].last_size ** 2

            pred_features[mode], gt_features, gt_all_features[mode], probs, X = \
                self.models[mode](input_seq, ret_rep=True)

            gt_all_features[mode] = self.cpc_projections[mode](gt_all_features[mode])
            pred_features[mode] = self.cpc_projections[mode](pred_features[mode])
            probabilities[mode] = probs

            # score is a 6d tensor: [B, P, SQ, B', N', SQ]
            score_ = self.get_feature_pair_score(pred_features[mode], gt_features)
            flat_scores[mode] = score_.view(B * NS * SQ, -1)

            if self.shouldFinetune:
                feature_dict['X'][mode].append(X.reshape(X.shape[0], -1)[::self.finetuneSkip].detach().cpu())

            del input_seq

        return pred_features, gt_all_features, agg_features, flat_scores, probabilities, feature_dict

    def update_supervised_loss_stats(self, logit, supervised_target, stats, loss_dict, mode, eval=False):

        B = supervised_target.shape[0]

        if self.multilabel_supervision:
            if eval:
                # Intermediate counters for per-class TP, TN, FP, FN
                tp, tn, fp, fn, top1_single, top3_single, all_single = calc_per_class_multilabel_counts(torch.sigmoid(logit), supervised_target)
                counter = dict(tp=tp, tn=tn, fp=fp, fn=fn, top1_single=top1_single, top3_single=top3_single, all_single=all_single)
                if not 'multilabel_counts' in stats[mu.SupervisionLoss][mode].keys():
                    stats[mu.SupervisionLoss][mode]['multilabel_counts'] = counter
                else:
                    stats[mu.SupervisionLoss][mode]['multilabel_counts']['tp'] += counter['tp']
                    stats[mu.SupervisionLoss][mode]['multilabel_counts']['tn'] += counter['tn']
                    stats[mu.SupervisionLoss][mode]['multilabel_counts']['fp'] += counter['fp']
                    stats[mu.SupervisionLoss][mode]['multilabel_counts']['fn'] += counter['fn']
                    stats[mu.SupervisionLoss][mode]['multilabel_counts']['top1_single'] += counter['top1_single']
                    stats[mu.SupervisionLoss][mode]['multilabel_counts']['top3_single'] += counter['top3_single']
                    stats[mu.SupervisionLoss][mode]['multilabel_counts']['all_single'] += counter['all_single']
        else:
            # Compute and log top-k accuracy
            topKs = calc_topk_accuracy(logit, supervised_target, self.accuracyKList)
            for i in range(len(self.accuracyKList)):
                stats[mu.SupervisionLoss][mode]["acc" + str(self.accuracyKList[i])].update(topKs[i].item(), B)

        # Compute supervision loss
        if not self.multilabel_supervision:
            supervised_target = supervised_target.view(-1)
        else:
            supervised_target = supervised_target.float()

        loss_dict[mu.SupervisionLoss][mode] = \
            self.criteria[mu.SupervisionLoss](logit, supervised_target)

    def update_supervised_hierarchical_loss_stats(self, logits, targets, stats, loss_dict, mode, eval=False):
        B = targets['video'].shape[0]

        # Compute and log top-k accuracy for video level
        topKs = calc_topk_accuracy(logits['main'], targets['video'], self.accuracyKList)
        for i in range(len(self.accuracyKList)):
            stats[mu.SupervisionLoss][mode]["acc" + str(self.accuracyKList[i])].update(topKs[i].item(), B)

        # Compute supervision hierarchical loss
        loss_dict[mu.SupervisionLoss][mode] = \
            self.criteria[mu.SupervisionLoss](logits['main'], targets['video'].view(-1))

        loss_dict[mu.HierarchicalLoss][mode] = \
            self.criteria[mu.HierarchicalLoss](logits['side'], targets['atomic'].float()) * 10.0

        if mu.WeighedHierarchicalLoss in self.losses:
            loss_dict[mu.WeighedHierarchicalLoss][mode] = self.criteria[mu.WeighedHierarchicalLoss][mode](
                [loss_dict[mu.SupervisionLoss][mode], loss_dict[mu.HierarchicalLoss][mode]]
            )

    def update_self_supervised_metrics(self, gt_all_features, flat_scores, probabilities, stats, data, eval=False):

        NS, N = self.args["pred_step"], self.args["num_seq"]
        loss_dict = {k: {} for k in self.losses}

        for mode in flat_scores.keys():
            B = data[mode].shape[0]
            SQ = self.models[mode].last_size ** 2

            target_flattened = self.standard_grid_mask[mode].view(self.B0 * NS * SQ, self.B1 * NS * SQ)[:B * NS * SQ, :B * NS * SQ]

            # CPC loss
            if mu.CPCLoss in self.losses:

                score_flat = flat_scores[mode]
                target = target_flattened

                target_lbl = target.float().argmax(dim=1)

                # Compute and log performance metrics
                topKs = calc_topk_accuracy(score_flat, target_lbl, self.accuracyKList)
                for i in range(len(self.accuracyKList)):
                    stats[mu.CPCLoss][mode]["acc" + str(self.accuracyKList[i])].update(topKs[i].item(), B)

                # Compute CPC loss for independent model training
                loss_dict[mu.CPCLoss][mode] = self.criteria[mu.CPCLoss](score_flat, target)

            # Supervision loss
            if mu.SupervisionLoss in self.losses:
                probability = probabilities[mode]
                supervised_target = data['labels'].to(self.device)
                self.update_supervised_loss_stats(probability, supervised_target, stats, loss_dict, mode, eval)

        if mu.CooperativeLoss in self.losses:
            for (m0, m1) in self.mode_pairs:
                if (m0 not in flat_scores.keys()) or (m1 not in flat_scores.keys()):
                    continue

                tupName = self.get_tuple_name(m0, m1)

                # Cdot related losses
                comp_gt_all0 = self.compiled_features[m0](gt_all_features[m0]).unsqueeze(3).unsqueeze(3)
                comp_gt_all1 = self.compiled_features[m1](gt_all_features[m1]).unsqueeze(3).unsqueeze(3)
                cdot0 = self.interModeDotHandler.get_cluster_dots(comp_gt_all0)
                cdot1 = self.interModeDotHandler.get_cluster_dots(comp_gt_all1)

                B, NS, B2, NS = cdot0.shape

                assert cdot0.shape == cdot1.shape == (B, NS, B2, NS), \
                    "Invalid shapes: {}, {}, {}".format(cdot0.shape, cdot1.shape, (B, NS, B2, NS))

                cos_sim_dot_loss = self.criteria[self.CooperativeLossLabel](cdot0, cdot1)
                loss_dict[self.CooperativeLossLabel][tupName] = self.dot_wt * cos_sim_dot_loss

                # Modality sync loss
                sync_loss, mode_stats = self.modeSyncers[tupName](gt_all_features[m0], gt_all_features[m1])

                # stats: dict modeLoss -> specificStat
                for modeLoss in mode_stats.keys():
                    for stat in mode_stats[modeLoss].keys():
                        stats[modeLoss][tupName][stat].update(mode_stats[modeLoss][stat].item(), B)

                loss_dict[mu.ModeSim][tupName] = self.sync_wt * sync_loss

        return loss_dict, stats

    @staticmethod
    def categorical_cross_entropy(logits, b):
        la = torch.nn.functional.log_softmax(logits, dim=1)
        return -(b * la).sum(dim=1).mean()

    def kd_loss_criterion(self, results, labels):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        :param results: Expect the dictionary returned by forward()
        :param labels: GT label for the batch
        """

        w = self.kd_weight
        t = self.temperature

        gt_loss = self.criteria[mu.SupervisionLoss](results['outs'], labels)

        kd_loss = 0.0

        for v in results["teacher_outs"]:
            kd_loss += self.categorical_cross_entropy(
                results["outs"] / t,
                torch.nn.functional.softmax(v / t, dim=1)
            )

        kd_loss /= len(results["teacher_outs"])

        # Weigh kd_loss with t^2 to preserve scale of gradients
        final_loss = (kd_loss * w * t * t) + (gt_loss * (1 - w))

        return {
            "loss": final_loss,
            "gt_loss": gt_loss,
            "kd_loss": kd_loss
        }

    def update_distillation_loss_stats(self, logits, supervised_target, loss_dict):
        for mode in self.student_modes:
            results = {
                'outs': logits[mode]['main'],
                'teacher_outs': [logits[m]['main'] for m in self.modes if m != mode]
            }

            losses = self.kd_loss_criterion(results, supervised_target.view(-1))
            loss_dict[mu.DistillLoss][mode] = losses['kd_loss']

    def get_supervised_forward_pass_results(self, data):
        results = {'feature': {}, 'logits': {}, 'grid': {}, 'attn': {}}
        for mode in self.modes:
            input_seq = data[mode].to(self.device)
            feature, main_logit, side_logit, grid, attn = self.models[mode](input_seq)
            results['feature'][mode], results['grid'][mode], results['attn'][mode] = \
                feature, grid, attn
            results['logits'][mode] = {'main': main_logit, 'side': side_logit}
            del input_seq
        return results

    def perform_supervised_forward_passes(self, data, feature_dict):

        if self.shouldFinetune:
            key = 'video_labels' if 'video_labels' in data else 'labels'
            feature_dict['Y'].append(data[key][::self.finetuneSkip])

        results = self.get_supervised_forward_pass_results(data)
        features, attentions, logits = results['grid'], results['attn'], results['logits']

        for mode in self.modes:
            if self.shouldFinetune:
                feature = results['feature'][mode]
                feature_dict['X'][mode].append(
                    feature.reshape(feature.shape[0], -1)[::self.finetuneSkip].detach().cpu())

        # features: B x D; clip level feature
        return features, logits, attentions, feature_dict

    def update_supervised_metrics(self, features, logits, stats, attentions, data, eval=False):

        B, NS, N = self.args["batch_size"], self.args["pred_step"], self.args["num_seq"]
        loss_dict = defaultdict(lambda: {})

        for mode in self.modes:
            # Hierarchical loss - normal or auto-weighed
            if self.hierarchical:
                logit = logits[mode]
                targets = {
                    'video': data['video_labels'].to(self.device),
                    'atomic': data['atomic_labels'].to(self.device),
                }
                self.update_supervised_hierarchical_loss_stats(logit, targets, stats, loss_dict, mode, eval)
            elif mu.SupervisionLoss in self.losses:
                # Supervision loss only if hierarchical loss is not present
                logit = logits[mode]['main']
                supervised_target = data['labels'].to(self.device)
                self.update_supervised_loss_stats(logit, supervised_target, stats, loss_dict, mode, eval)

        if mu.DistillLoss in self.losses:
            supervised_target = data['labels'].to(self.device)
            self.update_distillation_loss_stats(logits, supervised_target, loss_dict)

        if mu.AlignLoss in self.losses:
            for (m0, m1) in self.mode_pairs:
                # In the supervised setting, we receive B x N x D and B x D' features (block, clip level respectively)
                # Currently we receive only B x D' x s x s i.e. clip level features
                tupName = self.get_tuple_name(m0, m1)

                # Modality sync loss
                sync_loss, mode_stats, attns = self.modeSyncers[tupName](features[m0], features[m1])

                if attns:
                    attentions['{}_in_{}'.format(m0, tupName)] = attns['0']
                    attentions['{}_in_{}'.format(m1, tupName)] = attns['1']

                # stats: dict modeLoss -> specificStat
                for modeLoss in mode_stats.keys():
                    for stat in mode_stats[modeLoss].keys():
                        stats[modeLoss][tupName][stat].update(mode_stats[modeLoss][stat].item(), B)

                loss_dict[mu.AlignLoss][tupName] = self.sync_wt * sync_loss

        return loss_dict, stats

    def pretty_print_stats(self, stats):
        grouped_stats = {}
        for k, v in stats.items():
            a, b = k.split('/')[:2]
            middle = (a, b)
            if middle not in grouped_stats:
                grouped_stats[middle] = []
            grouped_stats[middle].append((k, v))
        for v in grouped_stats.values():
            print(sorted(v))

    def perform_self_supervised_pass(self, data, feature_dict, stats, eval):
        _, gt_all_features, agg_features, flat_scores, probabilities, feature_dict = \
            self.perform_self_supervised_forward_passes(data, feature_dict)
        loss_dict, stats = \
            self.update_self_supervised_metrics(gt_all_features, flat_scores, probabilities, stats, data, eval)
        return loss_dict, stats, feature_dict, {}

    def perform_supervised_pass(self, data, feature_dict, stats, eval):
        features, logits, attentions, feature_dict = self.perform_supervised_forward_passes(data, feature_dict)
        loss_dict, stats = self.update_supervised_metrics(features, logits, stats, attentions, data, eval)

        etc = {'attn': attentions}
        if self.hierarchical:
            etc['atomic_labels'] = data['atomic_labels'].cpu().detach()
            for m, v in logits.items():
                etc['atomic_preds_{}'.format(m)] = v['side'].cpu().detach()

        video_labels = 'video_labels' if self.hierarchical else 'labels'
        if eval:
            etc['video_labels'] = data[video_labels].cpu().detach()
            for m, v in logits.items():
                etc['video_preds_{}'.format(m)] = v['main'].cpu().detach()

        return loss_dict, stats, feature_dict, etc

    def perform_pass(self, data, feature_dict, stats, eval=False):
        if self.model_type == mu.ModelSSL:
            return self.perform_self_supervised_pass(data, feature_dict, stats, eval)
        elif self.model_type == mu.ModelSupervised:
            return self.perform_supervised_pass(data, feature_dict, stats, eval)

    def aggregate_finetune_set(self, feature_dict):
        if self.shouldFinetune:
            feature_dict['X'] = {k: torch.cat(v) for k, v in feature_dict['X'].items()}
            feature_dict['Y'] = torch.cat(feature_dict['Y']).reshape(-1).detach()
        return feature_dict

    def compute_atomic_action_stats(self, etcs, stats):

        def torch_to_numpy(y_pred, y_true):
            y_pred = torch.sigmoid(y_pred)
            return y_pred.cpu().detach().numpy(), y_true.cpu().detach().long().numpy()

        def map_score_avg(aps):
            return aps[aps == aps].mean()

        def map_score_weighted(aps, y_true):
            aps[aps != aps] = 0
            unique, support = np.unique(np.concatenate([np.nonzero(t)[0] for t in y_true]), return_counts=True)
            counts = np.zeros(448)
            counts[unique] = support
            print('Num missing classes:', (counts/counts.sum() < 1e-8).sum())
            return np.average(aps, weights=counts)

        def map_score_all(y_pred, y_true):
            y_pred, y_true = torch_to_numpy(y_pred, y_true)
            aps = sklearn.metrics.average_precision_score(y_true, y_pred, average=None)
            return map_score_avg(aps), map_score_weighted(aps, y_true)

        for m in self.modes:
            avg, wgt = map_score_all(etcs['atomic_preds_{}'.format(m)], etcs['atomic_labels'])
            stats['map/avg/{}'.format(m[0] + m[-1])] = avg
            stats['map/weighted/{}'.format(m[0] + m[-1])] = wgt

    def train_epoch(self, epoch):

        self.train()
        print("Model path is:", self.model_path)

        for mode in self.models.keys():
            if mode in self.student_modes:
                self.models[mode].train()
        for mode in self.modeSyncers.keys():
            self.modeSyncers[mode].train()

        losses_dict, stats = mu.init_loggers(self.losses)

        B = self.args["batch_size"]
        train = {'X': {m: [] for m in self.modes}, 'Y': []}

        tq = tqdm(self.train_loader, desc="Train: Ep {}".format(epoch), position=0)
        self.optimizer.zero_grad()

        etcs, overall_stats = {}, {}

        for idx, data in enumerate(tq):
            loss_dict, stats, train, etc = self.perform_pass(data, train, stats)

            loss = torch.tensor(0.0).to(self.device)
            # Only include losses which are part of self.losses i.e. not super, hierarchical if we're using weighed loss
            for l in loss_dict.keys():
                for v in loss_dict[l].keys():
                    losses_dict[l][v].update(loss_dict[l][v].item(), B)
                    if l in self.losses:
                        loss += loss_dict[l][v]

            loss.backward()

            if idx % self.grad_step_interval:
                self.optimizer.step()
                self.optimizer.zero_grad()

            overall_stats = mu.get_stats_dict(losses_dict, stats)
            tq_stats = mu.shorten_stats(overall_stats)
            tq.set_postfix(tq_stats)

            del loss

            # Perform logging
            if self.iteration % self.vis_log_freq == 0:
                # Log attention
                if 'attn' in etc:
                    self.addAttnGrid(etc['attn'])
                # Log images
                self.addImgGrid(data)
                for mode in self.modes:
                    if mode in data.keys():
                        # Log visuals of the complete input seq
                        self.log_visuals(data[mode], mode)

            # Don't need attn anymore
            if 'attn' in etc:
                del etc['attn']

            for k, v in etc.items():
                if k not in etcs:
                    etcs[k] = []
                etcs[k].append(v)

            if idx % self.args["print_freq"] == 0:
                self.log_metrics(losses_dict, stats, self.writer_train, prefix='local')

            self.iteration += 1

        for k in etcs.keys():
            etcs[k] = torch.cat(etcs[k])

        if self.hierarchical:
            self.compute_atomic_action_stats(etcs, overall_stats)

        print("Overall train stats:")
        self.pretty_print_stats(overall_stats)
        # log to wandb
        if self.use_wandb:
            wandb.log({'train/' + k: v for k, v in overall_stats.items()}, step=self.iteration)

        train = self.aggregate_finetune_set(train)

        return losses_dict, stats, train

    def eval_mode(self):
        self.eval()
        for mode in self.models.keys():
            if mode in self.student_modes:
                self.models[mode].eval()
                self.models[mode].eval_mode()
        for mode in self.modeSyncers.keys():
            self.modeSyncers[mode].eval()

    def validate_epoch(self, epoch):
        self.eval_mode()

        losses_dict, stats = mu.init_loggers(self.losses)
        overall_loss = AverageMeter()

        B = self.args["batch_size"]
        tq_stats = {}
        val = {'X': {m: [] for m in self.modes}, 'Y': []}

        etcs, overall_stats = {}, {}

        with torch.no_grad():
            tq = tqdm(self.val_loader, desc="Val: Ep {}".format(epoch), position=0)

            for idx, data in enumerate(tq):
                loss_dict, stats, val, etc = self.perform_pass(data, val, stats, eval=True)

                # Perform logging - Handle val separately
                # if self.iteration % self.vis_log_freq == 0:
                #     # Log attention
                #     if 'attn' in etc:
                #         self.addAttnGrid(etc['attn'])
                #     # Log images
                #     self.addImgGrid(data)
                #     for mode in self.modes:
                #         self.log_visuals(data[mode], mode)

                loss = torch.tensor(0.0).to(self.device)
                for l in loss_dict.keys():
                    for v in loss_dict[l].keys():
                        losses_dict[l][v].update(loss_dict[l][v].item(), B)
                        if l in self.losses:
                            loss += loss_dict[l][v]

                overall_loss.update(loss.item(), B)

                # Don't need attn anymore
                if 'attn' in etc:
                    del etc['attn']

                for k, v in etc.items():
                    if k not in etcs:
                        etcs[k] = []
                    etcs[k].append(v)

                overall_stats = mu.get_stats_dict(losses_dict, stats)
                tq_stats = mu.shorten_stats(overall_stats)
                tq.set_postfix(tq_stats)

        for k in etcs.keys():
            etcs[k] = torch.cat(etcs[k])

        if self.hierarchical:
            self.compute_atomic_action_stats(etcs, overall_stats)

        # compute validation metrics
        mu.compute_val_metrics(overall_stats)

        for loss in stats.keys():
            for mode in stats[loss].keys():
                mu.compute_val_metrics(stats[loss][mode], prefix='{}_{}'.format(loss, mode))

        # log to wandb
        if self.use_wandb:
            wandb.log({'val/' + k: v for k, v in overall_stats.items()}, step=self.iteration)

        print("Overall val stats:")
        self.pretty_print_stats(overall_stats)

        val = self.aggregate_finetune_set(val)

        return overall_loss, losses_dict, stats, val

    def test(self):
        self.eval_mode()

        losses_dict, stats = mu.init_loggers(self.losses)
        overall_loss = AverageMeter()

        B = self.args["batch_size"]
        tq_stats = {}
        val = {'X': {m: [] for m in self.modes}, 'Y': []}

        etcs, overall_stats = {}, {}

        with torch.no_grad():
            tq = tqdm(self.test_loader, desc="Test:", position=0)
            for idx, data in enumerate(tq):
                for k in data.keys():
                    data[k] = data[k].squeeze(0)

                loss_dict, stats, val, etc = self.perform_pass(data, val, stats, eval=True)

                loss = torch.tensor(0.0).to(self.device)
                for l in loss_dict.keys():
                    for v in loss_dict[l].keys():
                        losses_dict[l][v].update(loss_dict[l][v].item(), B)
                        if l in self.losses:
                            loss += loss_dict[l][v]

                overall_loss.update(loss.item(), B)

                # Don't need attn anymore
                if 'attn' in etc:
                    del etc['attn']

                for k, v in etc.items():
                    if k not in etcs:
                        etcs[k] = []
                    etcs[k].append(v)

                overall_stats = mu.get_stats_dict(losses_dict, stats)
                tq_stats = mu.shorten_stats(overall_stats)
                tq.set_postfix(tq_stats)

        for k in etcs.keys():
            etcs[k] = torch.cat(etcs[k])

        if self.hierarchical:
            self.compute_atomic_action_stats(etcs, overall_stats)

        # compute validation metrics
        mu.compute_val_metrics(overall_stats)

        for loss in stats.keys():
            for mode in stats[loss].keys():
                mu.compute_val_metrics(stats[loss][mode], prefix='{}_{}'.format(loss, mode))

        # log to wandb
        if self.use_wandb:
            wandb.log({'val/' + k: v for k, v in overall_stats.items()}, step=self.iteration)

        print("Overall val stats:")
        self.pretty_print_stats(overall_stats)

        val = self.aggregate_finetune_set(val)

        return overall_loss, losses_dict, stats, val

    def train_module(self):

        best_acc = {m: 0.0 for m in self.modes}

        for epoch in range(self.args["start_epoch"], self.args["epochs"]):
            train_losses, train_stats, trainD = self.train_epoch(epoch)
            eval_epoch = epoch % self.args["eval_freq"] == 0
            if eval_epoch:
                ovr_loss, val_losses, val_stats, valD = self.validate_epoch(epoch)
                self.lr_scheduler.step(ovr_loss.avg)

            # Log fine-tune performance
            # FIXME (haofeng): This below doesn't work with multi label supervision
            if self.shouldFinetune and not self.multilabel_supervision:
                if (epoch % self.args["ft_freq"] == 0) and eval_epoch:
                    self.model_finetuner.evaluate_classification(trainD, valD)
                    if not self.args["debug"]:
                        train_clustering_dict = self.model_finetuner.evaluate_clustering(trainD, tag='train')
                        # val_clustering_dict = self.model_finetuner.evaluate_clustering(valD, tag='val')
                        if self.use_wandb:
                            for n, d in zip(['train'], [train_clustering_dict]): # , val_clustering_dict])
                                for k0, d0 in d.items():
                                    wandb.log({'cluster/{}_{}_{}'.format(n, k0, k1): v for k1, v in d0.items()}, step=self.iteration)

            # save curve
            self.log_metrics(train_losses, train_stats, self.writer_train, prefix='global')
            if eval_epoch:
                self.log_metrics(val_losses, val_stats, self.writer_val, prefix='global')

                # save check_point for each mode individually
                for modality in self.modes:
                    loss = mu.CPCLoss
                    if loss not in self.losses:
                        loss = mu.SupervisionLoss

                    is_best = val_stats[loss][modality][1].avg > best_acc[modality]
                    best_acc[modality] = max(val_stats[loss][modality][1].avg, best_acc[modality])

                    state = {
                        'epoch': epoch + 1,
                        'mode': modality,
                        'net': self.args["net"],
                        'state_dict': self.models[modality].state_dict(),
                        'best_acc': best_acc[modality],
                        'optimizer': self.optimizer.state_dict(),
                        'iteration': self.iteration
                    }

                    for (m0, m1) in self.mode_pairs:
                        if modality not in [m0, m1]:
                            tupName = self.get_tuple_name(m0, m1)
                            state['mode_syncer_{}'.format(tupName)] = self.modeSyncers[tupName].state_dict()

                    save_checkpoint(
                        state=state,
                        mode=modality,
                        is_best=is_best,
                        filename=os.path.join(
                            self.model_path, 'mode_' + modality + '_epoch%s.pth.tar' % str(epoch + 1)
                        ),
                        gap=3,
                        keep_all=False
                    )

        print('Training from ep %d to ep %d finished' % (self.args["start_epoch"], self.args["epochs"]))

        total_stats = {
            'train': {
                'losses': train_losses,
                'stats': train_stats,
            },
        }

        if eval_epoch:
            total_stats['val'] = {
                'losses': val_losses,
                'stats': val_stats,
            }
        return total_stats


def get_backbone_for_modality(args, mode):
    tmp_args = deepcopy(args)
    tmp_args.mode = mode
    tmp_args_dict = deepcopy(vars(tmp_args))

    model = None
    print('Current model being used is: {}'.format(args.model))
    if mode == mu.AudioMode:
        model = m3d.AudioVGGEncoder(tmp_args_dict)
    else:
        if args.model == mu.ModelSSL:
            model = DpcRnn(tmp_args_dict)
        elif args.model == mu.ModelSupervised:
            model = SupervisedDpcRnn(tmp_args_dict)
        else:
            assert False, "Invalid model type: {}".format(args.model)

    if args.restore_ckpts[mode] is not None:
        print(mode, args.restore_ckpts[mode])
        # First try to load it hoping it's stored without the dataParallel
        print('Model saved in dataParallel form')
        model = m3d.get_parallel_model(model)
        model = mu.load_model(model, args.restore_ckpts[mode])
    else:
        model = m3d.get_parallel_model(model)

    # Freeze the required layers
    if args.train_what == 'last':
        for name, param in model.resnet.named_parameters():
            param.requires_grad = False

    if args.test:
        for param in model.parameters():
            param.requires_grad = False

    print('\n=========Check Grad: {}============'.format(mode))
    param_list = ['-'.join(name.split('.')[:2])
                  for name, param in model.named_parameters()
                  if not param.requires_grad]
    print(set(param_list))
    print('=================================\n')

    return model.to(args.device)


def run_multi_modal_training(args):

    torch.manual_seed(0)
    np.random.seed(0)

    # Update the batch size according to the number of GPUs
    if torch.cuda.is_available():
        args.batch_size *= torch.cuda.device_count()
        args.num_workers *= int(np.sqrt(2 * torch.cuda.device_count()))

    args.num_classes = mu.get_num_classes(args.dataset)
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    # FIXME: support multiple views from the same modality
    args.modes = get_modality_list(args.modalities)
    args.student_modes = get_modality_list(args.students)
    args.restore_ckpts = get_modality_restore_ckpts(args)
    args.old_lr = None

    args.img_path, args.model_path = mu.set_multi_modal_path(args)

    models = {}
    for mode in args.modes:
        models[mode] = get_backbone_for_modality(args, mode)

    # Restore from an earlier checkpoint
    args_dict = deepcopy(vars(args))

    args_dict["models"] = models
    args_dict["data_sources"] = '_'.join(args_dict["modes"]) + "_labels"

    model_trainer = MultiModalModelTrainer(args_dict)
    model_trainer = model_trainer.to(args.device)

    stats = None
    if args.test:
        stats = model_trainer.test()
    else:
        stats = model_trainer.train_module()

    return model_trainer, stats


if __name__ == '__main__':

    parser = mu.get_multi_modal_model_train_args()
    args = parser.parse_args()

    run_multi_modal_training(args)

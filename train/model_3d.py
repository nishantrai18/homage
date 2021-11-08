import sys
import math
import torch
import random

import numpy as np
import torch.nn as nn
import sim_utils as su
import model_utils as mu
import torch.nn.functional as F

from dataset_3d import get_spectrogram_window_length

sys.path.append('../backbone')

from select_backbone import select_resnet
from convrnn import ConvGRU


eps = 1e-7
INF = 25.0


class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_parallel_model(model):
    if torch.cuda.is_available():
        dev_count = torch.cuda.device_count()
        print("Using {} GPUs".format(dev_count))
        model = MyDataParallel(model, device_ids=list(range(dev_count)))
    return model


def get_num_channels(modality):
    if modality.startswith(mu.ImgMode):
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


class NonLinearProjection(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(NonLinearProjection, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 128
        self.projection = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, features):
        is_grid_format = (len(features.shape) == 5) and (features.shape[-1] <= 32)
        # In case the last two dimension are grids
        if is_grid_format:
            assert features.shape[-1] == features.shape[-2], "Invalid shape: {}".format(features.shape)
            features = features.permute(0, 1, 3, 4, 2)
        projected_features = self.projection(features)
        # projected_features = self.cosSimHandler.l2NormedVec(projected_features, dim=-1)
        if is_grid_format:
            projected_features = features.permute(0, 1, 4, 2, 3)
        return projected_features


class AttentionProjection(nn.Module):

    def __init__(self, input_dim, grid_shape):
        super(AttentionProjection, self).__init__()

        self.input_dim = input_dim
        self.grid_shape = grid_shape
        self.output_dim = 32

        # Projection layer to generate small attention maps
        self.hidden_dim = 32
        self.projection = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        # Generate attention map by squashing the grids
        self.total_cells = self.grid_shape[0] * self.grid_shape[1]
        self.attention = nn.Sequential(
            nn.Linear(self.total_cells * self.output_dim, self.total_cells),
            nn.ReLU(),
            nn.Linear(self.total_cells, self.total_cells),
            nn.Softmax(dim=-1),
        )

    def forward(self, features):
        # input is B, N, D, s, s
        B, N, D, s, s = features.shape
        assert features.shape[-1] == features.shape[-2], "Invalid shape: {}".format(features.shape)
        features = features.permute(0, 1, 3, 4, 2)

        projected_features = self.projection(features.reshape(B * N * s * s, D))
        # projected_features is -1, output_dim
        projected_features = projected_features.view(B, N, s, s, self.output_dim)
        # projected_features is B, N, s, s, output_dim
        attention_map = self.attention(projected_features.view(B * N, -1))
        # attention_map is B * N, total_cells
        attention_map = attention_map.view(B, N, 1, self.grid_shape[0], self.grid_shape[1])

        return attention_map

    def applyAttention(self, features):
        # features is B, 1, D, s, s
        attention_map = self.forward(features)
        # attention_map is B, 1, 1, s, s
        context = features * attention_map
        context = context.sum(-1).sum(-1)
        return context, attention_map


class ImageFetCombiner(nn.Module):

    def __init__(self, img_fet_dim, img_segments):
        super(ImageFetCombiner, self).__init__()

        # Input feature dimension is [B, dim, s, s]
        self.dim = img_fet_dim
        self.s = img_segments
        self.flat_dim = self.dim * self.s * self.s

        layers = []
        if self.s == 7:
            layers.append(nn.MaxPool2d(2, 2, padding=1))
            layers.append(nn.MaxPool2d(2, 2))
            layers.append(nn.AvgPool2d(2, 2))
        if self.s == 4:
            layers.append(nn.MaxPool2d(2, 2))
            layers.append(nn.AvgPool2d(2, 2))
        elif self.s == 2:
            layers.append(nn.AvgPool2d(2, 2))

        # input is B x dim x s x s
        self.feature = nn.Sequential(*layers)
        # TODO: Normalize
        # Output is B x dim

    def forward(self, input: torch.Tensor):
        # input is B, N, D, s, s
        B, N, D, s, s = input.shape
        input = input.view(B * N, D, s, s)
        y = self.feature(input)
        y = y.reshape(B, N, -1)
        return y


class WeighedLoss(nn.Module):
    """
    Class that implements automatically weighed loss from: https://arxiv.org/pdf/1705.07115.pdf
    """

    def __init__(self, num_losses, device):

        super(WeighedLoss, self).__init__()
        self.device = device
        self.coeffs = []
        for i in range(num_losses):
            init_value = random.random()
            param = nn.Parameter(torch.tensor(init_value))
            name = "auto_param_" + str(i)
            self.register_parameter(name, param)
            self.coeffs.append(param)

    def forward(self, losses=[]):
        """
        Forward pass
        Keyword Arguments:
            losses {list} -- List of tensors of losses
        Returns:
            torch.Tensor -- 0-dimensional tensor with final loss. Can backpropagate it.
        """

        assert len(losses) == len(self.coeffs), \
            "Loss mismatch, check how many losses are passed"

        net_loss = torch.tensor(0.0).to(self.device)

        for i, loss in enumerate(losses):
            net_loss += torch.exp(-self.coeffs[i]) * loss
            net_loss += 0.5 * self.coeffs[i]

        return net_loss


class IdentityFlatten(nn.Module):

    def __init__(self):
        super(IdentityFlatten, self).__init__()

    def forward(self, input: torch.Tensor):
        # input is B, N, D, s, s
        B, N, D, s, s = input.shape
        return input.reshape(B, N, -1)


class BaseDpcRnn(nn.Module):

    def get_modality_feature_extractor(self):
        if self.mode.split('-')[0] in [mu.ImgMode]:
            assert self.last_size0 == self.last_size1
            return ImageFetCombiner(self.final_feature_size, self.last_size0)
        else:
            assert False, "Invalid mode provided: {}".format(self.mode)

    '''DPC with RNN'''
    def __init__(self, args):
        super(BaseDpcRnn, self).__init__()

        torch.cuda.manual_seed(233)

        self.debug = args['debug']

        self.mode = args["mode"]

        self.num_seq = args["num_seq"]
        self.seq_len = args["seq_len"]
        self.sample_size = args["img_dim"]
        self.last_duration = int(math.ceil(self.seq_len / 4))

        self.last_size1 = int(math.ceil(self.sample_size / 32))
        self.last_size0 = self.last_size1
        if self.mode == mu.AudioMode:
            # Assume each audio image is 32x128
            self.last_size0 = 1

        self.last_size = None
        if self.last_size0 == self.last_size1:
            self.last_size = self.last_size0

        print('final feature map has size %dx%d' % (self.last_size0, self.last_size1))

        self.in_channels = get_num_channels(self.mode)
        self.l2_norm = True
        self.num_classes = args['num_classes']
        self.dropout = 0.75

        if self.debug:
            self.dropout = 0.0

        track_running_stats = True
        print("Track running stats: {}".format(track_running_stats))
        self.backbone, self.param = select_resnet(
            args["net"], track_running_stats=track_running_stats, in_channels=self.in_channels
        )

        # params for GRU
        self.param['num_layers'] = 1
        self.param['hidden_size'] = self.param['feature_size']

        # param for current model
        self.final_feature_size = self.param["feature_size"]
        self.total_feature_size = self.param['hidden_size'] * (self.last_size0 * self.last_size1)

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=self.param['hidden_size'],
                               kernel_size=1,
                               num_layers=self.param['num_layers'])

        self.compiled_features = self.get_modality_feature_extractor()
        self.cosSimHandler = su.CosSimHandler()

        self.mask = None
        # self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)

        self.panasonic_num_classes = {'video': 75, 'atomic': 448}

    def initialize_supervised_inference_layers(self):
        self.final_bn = nn.BatchNorm1d(self.param['feature_size'])
        self.final_bn.weight.data.fill_(1)
        self.final_bn.bias.data.zero_()

        self.final_fc = self.init_classification_fc_layer(self.num_classes)

    def init_classification_fc_layer(self, num_classes):
        final_fc = nn.Sequential(
            nn.Linear(self.param['feature_size'], self.param['feature_size']),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.param['feature_size'], num_classes),
        )

        self._initialize_weights(final_fc)

        return final_fc

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None


class DpcRnn(BaseDpcRnn):

    '''DPC with RNN'''
    def __init__(self, args):

        print('Using DPC-RNN model for mode: {}'.format(args["mode"]))

        super(DpcRnn, self).__init__(args)

        self.pred_step = args["pred_step"]
        self.network_pred = nn.Sequential(
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
                        )
        self._initialize_weights(self.network_pred)

        self.is_supervision_enabled = mu.SupervisionLoss in args["losses"]
        if self.is_supervision_enabled:
            self.initialize_supervised_inference_layers()

    def get_representation(self, block, detach=False):

        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        feature = self.backbone(block)

        del block
        feature = F.relu(feature)

        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=1)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size0, self.last_size1)
        # [B*N,D,last_size,last_size]
        context, _ = self.agg(feature)
        context = context[:,-1,:].unsqueeze(1)
        context = F.max_pool3d(context, (1, self.last_size0, self.last_size1), stride=1).squeeze(-1).squeeze(-1)
        del feature

        if self.l2_norm:
            context = self.cosSimHandler.l2NormedVec(context, dim=2)

        # Return detached version if required
        if detach:
            return context.detach()
        else:
            return context

    def forward(self, block, ret_rep=False):
        # ret_cdot values: [c, z, zt]

        # block: [B, N, C, SL, W, H]
        # B: Batch, N: Number of sequences per instance, C: Channels, SL: Sequence Length, W, H: Dims

        ### extract feature ###
        (B, N, C, SL, H, W) = block.shape

        block = block.view(B*N, C, SL, H, W)
        feature = self.backbone(block)

        del block

        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))

        if self.l2_norm:
            feature = self.cosSimHandler.l2NormedVec(feature, dim=1)

        # before ReLU, (-inf, +inf)
        feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size0, self.last_size1)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size0, self.last_size1)

        # Generate feature for future frames
        feature_inf = feature_inf_all[:, N - self.pred_step::, :].contiguous()

        del feature_inf_all

        # Random assignment to serve as placeholder value
        probabilities = torch.tensor(0)
        # aggregate and predict overall context
        probabilities = None
        if self.is_supervision_enabled:
            context, _ = self.agg(feature)
            context = context[:, -1, :].unsqueeze(1)
            context = F.max_pool3d(context, (1, self.last_size0, self.last_size1), stride=1).squeeze(-1).squeeze(-1)

            # [B,N,C] -> [B,C,N] -> BN() -> [B,N,C], because BN operates on id=1 channel.
            context = self.final_bn(context.transpose(-1, -2)).transpose(-1,-2)
            probabilities = self.final_fc(context).view(B, self.num_classes)

        ### aggregate, predict future ###
        # Generate inferred future (stored in feature_inf) through the initial frames
        _, hidden = self.agg(feature[:, 0:N-self.pred_step, :].contiguous())

        if self.l2_norm:
            hidden = self.cosSimHandler.l2NormedVec(hidden, dim=2)

        # Get the last hidden state, this gives us the predicted representation
        # after tanh, (-1,1). get the hidden state of last layer, last time step
        hidden = hidden[:, -1, :]

        # Predict next pred_step time steps for this instance
        pred = []
        for i in range(self.pred_step):
            # sequentially pred future based on the hidden states
            p_tmp = self.network_pred(hidden)

            if self.l2_norm:
                p_tmp = self.cosSimHandler.l2NormedVec(p_tmp, dim=1)

            pred.append(p_tmp)
            _, hidden = self.agg(p_tmp.unsqueeze(1), hidden.unsqueeze(0))

            if self.l2_norm:
                hidden = self.cosSimHandler.l2NormedVec(hidden, dim=2)

            hidden = hidden[:, -1, :]
        # Contains the representations for each of the next pred steps
        pred = torch.stack(pred, 1) # B, pred_step, xxx

        # Both are of the form [B, pred_step, D, s, s]
        return pred, feature_inf, feature, probabilities, hidden


# Vals to return
ContextVal = 'ctx'
GridCtxVal = 'grid'
AttentionVal = 'attn'


class SupervisedDpcRnn(BaseDpcRnn):

    '''
    DPC with RNN for supervision
    '''

    def __init__(self, args):
        print('Using Supervised DPC-RNN model for mode: {}'.format(args["mode"]))
        super(SupervisedDpcRnn, self).__init__(args)

        self.hierarchical = args['hierarchical']
        self.initialize_supervised_inference_layers()
        if self.hierarchical:
            self.init_multihead_layers()

        # Initialize attention based layers
        self.attention = args['attention']
        self.attention_fn = AttentionProjection(self.final_feature_size, (self.last_size0, self.last_size1))
        print('Using attention: {}'.format(self.attention))

    def eval_mode(self):
        self.backbone.eval()
        self.final_bn.eval()
        for m in list(self.backbone.modules()) + list(self.modules()):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    def init_multihead_layers(self):
        self.final_fc_video = self.init_classification_fc_layer(self.panasonic_num_classes['video'])
        self.final_fc_atomic = self.init_classification_fc_layer(self.panasonic_num_classes['atomic'])

    def get_representation(self, block, returns=[ContextVal]):

        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        feature = self.backbone(block)

        del block
        feature = F.relu(feature)

        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=1)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size0, self.last_size1)
        # feature is [B,N,D,last_size, last_size] now

        context, _ = self.agg(feature)
        del feature

        # Get the context at the final timestep i.e. remove N
        gridCtx = context[:,-1,:].unsqueeze(1)

        attention_map = None
        # context is B, 1, D, s, s
        if self.attention:
            attention_map = self.attention_fn(gridCtx)
            # attention_map is B, 1, 1, s, s
        else:
            attention_map = torch.ones((B, 1, 1, self.last_size0, self.last_size1), device=gridCtx.device)
            # Normalize attention_map
            attention_map = attention_map / attention_map.sum(-1, keepdim=True).sum(-2, keepdim=True)

        context = gridCtx * attention_map
        context = context.sum(-1).sum(-1)

        valsToReturn = []
        if ContextVal in returns:
            valsToReturn.append(context)
        if GridCtxVal in returns:
            valsToReturn.append(gridCtx)
        if AttentionVal in returns:
            valsToReturn.append(attention_map)

        return tuple(valsToReturn)

    def forward(self, block):
        context, grid, attn = self.get_representation(block, returns=[ContextVal, GridCtxVal, AttentionVal])

        # [B,N,C] -> [B,C,N] -> BN() -> [B,N,C], because BN operates on id=1 channel.
        context = self.final_bn(context.transpose(-1, -2)).transpose(-1, -2)

        # for m in self.backbone.modules():
        #     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        #         print(m, torch.unique(m.running_mean)[:10], torch.unique(m.running_var)[:10])

        main_logits, atomic_logits = None, None
        if self.hierarchical:
            main_logits = self.final_fc_video(context).contiguous().reshape(-1, self.panasonic_num_classes['video'])
            atomic_logits = self.final_fc_atomic(context).contiguous().reshape(-1, self.panasonic_num_classes['atomic'])
        else:
            main_logits = self.final_fc(context).view(-1, self.num_classes)
            # Dummy result, doesn't matter in the non hierarchical case
            atomic_logits = main_logits.detach()

        return context, main_logits, atomic_logits, grid, attn


import math
import vgg


class AudioVGGEncoder(nn.Module):
    '''
    VGG model with feature outputs
    '''
    def __init__(self, args):
        super(AudioVGGEncoder, self).__init__()

        torch.cuda.manual_seed(233)

        self.dim = (128, get_spectrogram_window_length(args['seq_len'], args['num_seq'], args['ds']))
        self.args = args

        self.dropout = 0.75
        self.num_classes = args['num_classes']

        self.debug = args['debug']
        if self.debug:
            self.dropout = 0.0

        # cfg['E'] refers to VGG_19 with batchnorm
        self.custom_cfg = list(vgg.cfg['E'])
        # We add a convolution and maxpooling to reduce resolution
        self.reducedDim = (self.dim[0] // 32, self.dim[1] // 32)
        self.numReductions = int(math.log(min(self.reducedDim), 2))
        for _ in range(self.numReductions):
            self.custom_cfg.extend([512, 'M'])

        # Gives the final shape after the new reductions
        self.numResidualElements = int(np.prod(self.reducedDim) / 2 ** (len(self.reducedDim) * self.numReductions))
        self.param = {'mid_feature_size': 512 * self.numResidualElements, 'feature_size': 256}

        self.features = vgg.make_layers(self.custom_cfg, batch_norm=True)
        self.flatten = nn.Sequential(
            nn.Linear(self.param['mid_feature_size'], self.param['feature_size']),
        )

        self.final_feature_size = self.param['feature_size']
        self.last_size = 1

        self.is_supervised = args['model'] == mu.ModelSupervised
        if self.is_supervised:
            self.final_fc = self.init_classification_fc_layer(self.num_classes)

        self.panasonic_num_classes = {'video': 75, 'atomic': 448}

        self.hierarchical = args['hierarchical']
        if self.hierarchical:
            self.init_multihead_layers()

    def eval_mode(self):
        self.features.eval()
        for m in list(self.features.modules()) + list(self.modules()):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    def init_multihead_layers(self):
        self.final_fc_video = self.init_classification_fc_layer(self.panasonic_num_classes['video'])
        self.final_fc_atomic = self.init_classification_fc_layer(self.panasonic_num_classes['atomic'])

    def init_classification_fc_layer(self, num_classes):
        final_fc = nn.Sequential(
            nn.Linear(self.param['feature_size'], self.param['feature_size']),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.param['feature_size'], num_classes),
        )
        self._initialize_weights(final_fc)
        return final_fc

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.flatten(x)
        B, D = x.shape

        if self.is_supervised:
            x = x.view(B, 1, D)
            main_logits, atomic_logits = None, None
            if self.hierarchical:
                main_logits = self.final_fc_video(x.view(B, D)).contiguous().view(-1, self.panasonic_num_classes['video'])
                atomic_logits = self.final_fc_atomic(x.view(B, D)).contiguous().view(-1, self.panasonic_num_classes['atomic'])
            else:
                main_logits = self.final_fc(x.view(B, D)).view(-1, self.num_classes)
                # Dummy result, doesn't matter in the non hierarchical case
                atomic_logits = main_logits.detach()
            return x, main_logits, atomic_logits, x.unsqueeze(-1).unsqueeze(-1), x
        else:
            x = x.view(B, 1, D, 1, 1)
            return x


import unittest
from tqdm import tqdm


class TestForwardPass(unittest.TestCase):
    @classmethod
    def setUp(self):
        """
        This code code is ran once before all tests.
        """

        parser = mu.get_multi_modal_model_train_args()
        self.args = vars(parser.parse_args(''))
        self.args["mode"] = self.args["modalities"][0]
        self.args["img_dim"] = 64
        self.args["num_classes"] = 10

        self.args["num_seq"], self.args["seq_len"] = 2, 4

        self.B, self.N, self.SL, self.H, self.W, self.D, self.IC = \
            self.args["batch_size"], self.args["num_seq"], self.args["seq_len"], self.args["img_dim"], self.args["img_dim"], 256, 3

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)

        self.cosSimHandler = su.CosSimHandler().to(self.device)

    def test_forward_pass_for_attention(self):
        # Run training for each loss
        self.args["attention"] = True
        self.model = SupervisedDpcRnn(self.args)
        block = torch.rand(self.B, self.N, self.IC, self.SL, self.H, self.W)
        self.model(block)
        self.model.get_representation(block)

    def test_forward_pass_without_attention(self):
        self.args["attention"] = False
        self.model = SupervisedDpcRnn(self.args)
        # Run training for each loss
        block = torch.rand(self.B, self.N, self.IC, self.SL, self.H, self.W)
        self.model(block)
        self.model.get_representation(block)


if __name__ == '__main__':
    unittest.main()

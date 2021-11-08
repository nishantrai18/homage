import torch
import sys

sys.path.append('../utils')
from augmentation import *

sys.path.append('../train')
import model_utils as mu


def get_train_transforms(args):
    if args.modality == mu.ImgMode:
        return get_imgs_train_transforms(args)
    elif args.modality == mu.FlowMode:
        return get_flow_transforms(args)
    elif args.modality == mu.KeypointHeatmap:
        return get_heatmap_transforms(args)
    elif args.modality == mu.SegMask:
        return get_segmask_transforms(args)


def get_val_transforms(args):
    if args.modality == mu.ImgMode:
        return get_imgs_val_transforms(args)
    elif args.modality == mu.FlowMode:
        return get_flow_transforms(args)
    elif args.modality == mu.KeypointHeatmap:
        return get_heatmap_transforms(args)
    elif args.modality == mu.SegMask:
        return get_segmask_transforms(args)


def get_test_transforms(args):
    if args.modality == mu.ImgMode:
        return get_imgs_test_transforms(args)
    elif args.modality == mu.FlowMode:
        return get_flow_test_transforms(args)
    elif args.modality == mu.KeypointHeatmap:
        return get_heatmap_test_transforms(args)
    elif args.modality == mu.SegMask:
        return get_segmask_test_transforms(args)


def get_imgs_test_transforms(args):

    transform = transforms.Compose([
        RandomSizedCrop(consistent=True, size=224, p=0.0),
        Scale(size=(args.img_dim, args.img_dim)),
        ToTensor(),
        Normalize()
    ])

    return transform


def get_flow_test_transforms(args):
    center_crop_size = 224
    if args.dataset == 'kinetics':
        center_crop_size = 128

    transform = transforms.Compose([
        CenterCrop(size=center_crop_size, consistent=True),
        Scale(size=(args.img_dim, args.img_dim)),
        ToTensor(),
    ])

    return transform


def get_heatmap_test_transforms(_):
    transform = transforms.Compose([
        CenterCropForTensors(size=192),
        ScaleForTensors(size=(64, 64)),
    ])
    return transform


def get_segmask_test_transforms(_):
    transform = transforms.Compose([
        CenterCropForTensors(size=192),
        ScaleForTensors(size=(64, 64)),
    ])
    return transform


def get_imgs_train_transforms(args):
    transform = None

    # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
    if args.dataset == 'ucf101':
        transform = transforms.Compose([
            RandomSizedCrop(consistent=True, size=224, p=1.0),
            Scale(size=(args.img_dim, args.img_dim)),
            RandomHorizontalFlip(consistent=True),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.3, consistent=True),
            ToTensor(),
            Normalize()
        ])
    elif (args.dataset == 'jhmdb') or (args.dataset == 'hmdb51'):
        transform = transforms.Compose([
            RandomSizedCrop(consistent=True, size=224, p=1.0),
            Scale(size=(args.img_dim, args.img_dim)),
            RandomHorizontalFlip(consistent=True),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.3, consistent=True),
            ToTensor(),
            Normalize()
        ])
    # designed for kinetics400, short size=150, rand crop to 128x128
    elif args.dataset == 'kinetics':
        transform = transforms.Compose([
            RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            RandomHorizontalFlip(consistent=True),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.3, consistent=True),
            ToTensor(),
            Normalize()
        ])

    return transform


def get_imgs_val_transforms(args):
    transform = None

    # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
    if args.dataset == 'ucf101':
        transform = transforms.Compose([
            RandomSizedCrop(consistent=True, size=224, p=0.3),
            Scale(size=(args.img_dim, args.img_dim)),
            RandomHorizontalFlip(consistent=True),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3, consistent=True),
            ToTensor(),
            Normalize()
        ])
    elif (args.dataset == 'jhmdb') or (args.dataset == 'hmdb51'):
        transform = transforms.Compose([
            RandomSizedCrop(consistent=True, size=224, p=0.3),
            Scale(size=(args.img_dim, args.img_dim)),
            RandomHorizontalFlip(consistent=True),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3, consistent=True),
            ToTensor(),
            Normalize()
        ])
    # designed for kinetics400, short size=150, rand crop to 128x128
    elif args.dataset == 'kinetics':
        transform = transforms.Compose([
            RandomSizedCrop(consistent=True, size=224, p=0.3),
            RandomHorizontalFlip(consistent=True),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3, consistent=True),
            ToTensor(),
            Normalize()
        ])

    return transform


def get_flow_transforms(args):
    transform = None

    # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
    if (args.dataset == 'ucf101') or (args.dataset == 'jhmdb') or (args.dataset == 'hmdb51'):
        transform = transforms.Compose([
            RandomIntensityCropForFlow(size=224),
            Scale(size=(args.img_dim, args.img_dim)),
            ToTensor(),
        ])
    # designed for kinetics400, short size=150, rand crop to 128x128
    elif args.dataset == 'kinetics':
        transform = transforms.Compose([
            RandomIntensityCropForFlow(size=args.img_dim),
            ToTensor(),
        ])

    return transform


def get_heatmap_transforms(_):
    crop_size = int(192 * 0.8)
    transform = transforms.Compose([
        RandomIntensityCropForTensors(size=crop_size),
        ScaleForTensors(size=(64, 64)),
    ])
    return transform


def get_segmask_transforms(_):
    crop_size = int(192 * 0.8)
    transform = transforms.Compose([
        RandomIntensityCropForTensors(size=crop_size),
        ScaleForTensors(size=(64, 64)),
    ])
    return transform

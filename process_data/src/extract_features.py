import argparse
import numpy as np
import cv2
import os
import glob
import torch

cv2.setNumThreads(0)

from tqdm import tqdm
from torch.utils import data
from typing import Dict, List, Union

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.layers import interpolate, cat
from detectron2.utils.logger import setup_logger
setup_logger()


def str2bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]


imgShape = None

from typing import Dict, List, Optional, Tuple, Union
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.layers import interpolate, cat


@torch.no_grad()
def process_heatmaps(maps, rois, img_shapes):
    """
    Extract predicted keypoint locations from heatmaps.
    Args:
        maps (Tensor): (#ROIs, #keypoints, POOL_H, POOL_W). The predicted heatmap of logits for
            each ROI and each keypoint.
        rois (Tensor): (#ROIs, 4). The box of each ROI.
    Returns:
        Tensor of shape (#ROIs, #keypoints, POOL_H, POOL_W) representing confidence scores
    """

    offset_i = (rois[:, 1]).int()
    offset_j = (rois[:, 0]).int()

    widths = (rois[:, 2] - rois[:, 0]).clamp(min=1)
    heights = (rois[:, 3] - rois[:, 1]).clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    # roi_map_scores = torch.zeros((maps.shape[0], maps.shape[1], imgShape[0], imgShape[1]))
    roi_map_scores = [torch.zeros((maps.shape[1], img_shapes[i][0], img_shapes[i][1])) for i in range(maps.shape[0])]
    num_rois, num_keypoints = maps.shape[:2]

    for i in range(num_rois):
        outsize = (int(heights_ceil[i]), int(widths_ceil[i]))
        # #keypoints x H x W
        roi_map = interpolate(maps[[i]], size=outsize, mode="bicubic", align_corners=False).squeeze(0)

        # softmax over the spatial region
        max_score, _ = roi_map.view(num_keypoints, -1).max(1)
        max_score = max_score.view(num_keypoints, 1, 1)
        tmp_full_resolution = (roi_map - max_score).exp_()
        tmp_pool_resolution = (maps[i] - max_score).exp_()

        norm_score = ((tmp_full_resolution / tmp_pool_resolution.sum((1, 2), keepdim=True)) * 255.0).to(torch.uint8)

        # Produce scores over the region H x W, but normalize with POOL_H x POOL_W,
        # so that the scores of objects of different absolute sizes will be more comparable
        for idx in range(num_keypoints):
            roi_map_scores[i][idx, offset_i[i]:(offset_i[i] + outsize[0]), offset_j[i]:(offset_j[i] + outsize[1])] = \
                norm_score[idx, ...].float()

    return roi_map_scores


def heatmap_rcnn_inference(pred_keypoint_logits, pred_instances):
    bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)

    num_instances_per_image = [len(i) for i in pred_instances]
    img_shapes = [instance._image_size for instance in pred_instances for _ in range(len(instance))]
    hm_results = process_heatmaps(pred_keypoint_logits.detach(), bboxes_flat.detach(), img_shapes)

    hm_logits = []
    cumsum_idx = np.cumsum(num_instances_per_image)

    assert len(hm_results) == cumsum_idx[-1], \
        "Invalid sizes: {}, {}, {}".format(len(hm_results), cumsum_idx[-1], cumsum_idx)

    for idx in range(len(num_instances_per_image)):
        l = 0 if idx == 0 else cumsum_idx[idx - 1]
        if num_instances_per_image[idx] == 0:
            hm_logits.append(torch.zeros((0, 17, 0, 0)))
        else:
            hm_logits.append(torch.stack(hm_results[l:l + num_instances_per_image[idx]]))

    for idx in range(min(len(pred_instances), len(hm_logits))):
        pred_instances[idx].heat_maps = hm_logits[idx]


@ROI_HEADS_REGISTRY.register()
class HeatmapROIHeads(StandardROIHeads):
    """
    A Standard ROIHeads which contains returns HeatMaps instead of keypoints.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

    def _forward_keypoint(
            self, features: List[torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        if not self.keypoint_on:
            return {} if self.training else instances

        if self.training:
            assert False, "Not implemented yet!"
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)
            heatmap_rcnn_inference(keypoint_logits, instances)
            return instances


def get_heatmap_detection_module():
    # Inference with a keypoint detection module
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NAME = "HeatmapROIHeads"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
    predictor = build_model(cfg)
    print("heatmap head:", cfg.MODEL.ROI_HEADS.NAME)
    DetectionCheckpointer(predictor).load(cfg.MODEL.WEIGHTS)
    predictor.eval()
    return cfg, predictor


def get_panoptic_segmentation_module():
    # Inference with a segmentation module
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
    predictor = build_model(cfg)
    print("segmask head:", cfg.MODEL.ROI_HEADS.NAME)
    DetectionCheckpointer(predictor).load(cfg.MODEL.WEIGHTS)
    predictor.eval()
    return cfg, predictor


def individual_collate(batch):
    """
    Custom collation function for collate with new implementation of individual samples in data pipeline
    """

    data = batch

    # Assuming there's at least one instance in the batch
    add_data_keys = data[0].keys()
    collected_data = {k: [] for k in add_data_keys}

    for i in range(len(list(data))):
        for k in add_data_keys:
            collected_data[k].extend(data[i][k])

    return collected_data


def resize_dim(w, h, target):
    '''resize (w, h), such that the smaller side is target, keep the aspect ratio'''
    if w >= h:
        return (int(target * w / h), int(target))
    else:
        return (int(target), int(target * h / w))


class VideoDataset(data.Dataset):

    def __init__(self, v_root, vid_range, save_path, skip_len=2):
        super(VideoDataset, self).__init__()

        self.v_root = v_root
        self.vid_range = vid_range
        self.save_path = save_path

        self.init_videos()

        self.max_idx = len(self.v_names)
        self.skip = skip_len

        self.width, self.height = 320, 240
        self.dim = 192

    def num_frames_in_vid(self, v_path):
        vidcap = cv2.VideoCapture(v_path)
        nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vidcap.release()
        return nb_frames

    def extract_video_opencv(self, v_path):

        global imgShape

        v_class = v_path.split('/')[-2]
        v_name = os.path.basename(v_path)[0:-4]

        vidcap = cv2.VideoCapture(v_path)
        nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

        if (width == 0) or (height == 0):
            print(v_path, 'not successfully loaded, drop ..')
            return

        new_dim = resize_dim(width, height, self.dim)

        fnames, imgs = [], []

        success, image = vidcap.read()
        count = 1
        while success:
            image = cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)
            if (count % self.skip == 0):
                fnames.append((v_class, v_name, count))
                imgs.append(image)

            success, image = vidcap.read()
            count += 1

        if int(nb_frames * 0.8) > count:
            print(v_path, 'NOT extracted successfully: %df/%df' % (count, nb_frames))

        vidcap.release()

        return imgs, fnames

    def vid_already_processed(self, v_path):
        v_class = v_path.split('/')[-2]
        # Remove avi extension
        v_name = os.path.basename(v_path)[0:-4]

        out_dir = os.path.join(self.save_path, v_class, v_name)
        num_frames = self.num_frames_in_vid(v_path)
        for count in range(max(0, num_frames - 10), num_frames):
            fpath = os.path.join(out_dir, 'segmask_%05d.npz' % count)
            if os.path.exists(fpath):
                return True

        return False

    def init_videos(self):
        print('processing videos from %s' % self.v_root)

        self.v_names = []

        v_act_root = sorted(glob.glob(os.path.join(self.v_root, '*/')))

        num_skip, tot_files = 0, 0
        for vid_dir in v_act_root:
            v_class = vid_dir.split('/')[-2]

            if (v_class[0].lower() >= self.vid_range[0]) and (v_class[0].lower() <= self.vid_range[1]):
                v_paths = glob.glob(os.path.join(vid_dir, '*.avi'))
                v_paths = sorted(v_paths)

                for v_path in v_paths:
                    tot_files += 1
                    if self.vid_already_processed(v_path):
                        num_skip += 1
                        continue
                    self.v_names.append(v_path)

        print('Processing: {} files. Skipped: {}/{} files.'.format(len(self.v_names), num_skip, tot_files))

    def __getitem__(self, idx):
        vname = self.v_names[idx]
        imgs, fnames = self.extract_video_opencv(vname)
        return {"img": imgs, "filename": fnames}

    def __len__(self):
        return self.max_idx


def get_video_data_loader(path, vid_range, save_path, batch_size=2):
    dataset = VideoDataset(path, vid_range, save_path)
    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=data.SequentialSampler(dataset),
        shuffle=False,
        num_workers=2,
        collate_fn=individual_collate,
        pin_memory=True,
        drop_last=True
    )
    return data_loader


def write_heatmap_to_file(root, fname, heatmap):
    # fname is a list of (class, vname, count)
    v_class, v_name, count = fname
    out_dir = os.path.join(root, v_class, v_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.savez_compressed(os.path.join(out_dir, 'heatmap_%05d.npz' % count), hm=heatmap)


def write_segmask_to_file(root, fname, segmask):
    # fname is a list of (class, vname, count)
    v_class, v_name, count = fname
    out_dir = os.path.join(root, v_class, v_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.savez_compressed(os.path.join(out_dir, 'segmask_%05d.npz' % count), seg=segmask)


def convert_to_uint8(x):
    x[x < 0.0] = 0.0
    x[x > 255.0] = 255.0
    nx = x.to(torch.uint8).numpy()
    return nx


def process_videos(root, vid_provider, args, batch_size=32, debug=False):

    _, modelKP = get_heatmap_detection_module()
    _, modelPS = get_panoptic_segmentation_module()

    for batch in tqdm(vid_provider):
        imgsTot, fnamesTot = batch['img'], batch['filename']

        for idx in range(0, len(imgsTot), batch_size):

            imgs, fnames = imgsTot[idx: idx + batch_size], fnamesTot[idx: idx + batch_size]

            imgsDict = [{'image': torch.Tensor(img).float().permute(2, 0, 1)} for img in imgs]

            with torch.no_grad():
                if args.heatmap:
                    outputsKP = modelKP(imgsDict)
                if args.segmask:
                    outputsPS = modelPS(imgsDict)

            for i in range(len(imgs)):
                if args.heatmap:
                    # Process the keypoints
                    try:
                        heatmap = outputsKP[i]['instances'].heat_maps.cpu()
                        scores = outputsKP[i]['instances'].scores.cpu()
                        avgHeatmap = (heatmap * scores.view(-1, 1, 1, 1)).sum(dim=0)
                        # Clamp the max values
                        avgHeatmap = convert_to_uint8(avgHeatmap)
                    except:
                        print("Heatmap generation:", fnames[i])
                        print(outputsKP[i])
                    else:
                        assert avgHeatmap.shape[0] == 17, "Invalid size: {}".format(heatmap.shape)
                        if not debug:
                            write_heatmap_to_file(root, fnames[i], avgHeatmap)

                if args.segmask:
                    # Process the segmentation mask
                    try:
                        semantic_map = torch.softmax(outputsPS[i]['sem_seg'].detach(), dim=0)[0].cpu() * 255.0
                        semantic_map = convert_to_uint8(semantic_map)
                    except:
                        print("Segmask generation:", fnames[i])
                        print(outputsPS[i])
                    else:
                        if not debug:
                            write_segmask_to_file(root, fnames[i], semantic_map)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='/scr/data/ucf101/features/', type=str)
    parser.add_argument('--dataset', default='/scr/data/ucf101/videos', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--vid_range', default='az', type=str)
    parser.add_argument('--debug', default=0, type=int)
    parser.add_argument('--heatmap', default=0, type=int)
    parser.add_argument('--segmask', default=0, type=int)
    args = parser.parse_args()

    vid_provider = get_video_data_loader(args.dataset, args.vid_range, args.save_path)

    process_videos(args.save_path, vid_provider, batch_size=args.batch_size, debug=args.debug, args=args)

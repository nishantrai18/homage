from joblib import delayed, Parallel
import os 
import sys 
import glob 
from tqdm import tqdm 
import cv2
import argparse

import matplotlib.pyplot as plt
plt.switch_backend('agg')


def str2bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]


def extract_video_opencv(v_path, f_root, dim=240):
    '''v_path: single video path;
       f_root: root to store frames'''
    v_class = v_path.split('/')[-2]
    v_name = os.path.basename(v_path)[0:-4]
    out_dir = os.path.join(f_root, v_class, v_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vidcap = cv2.VideoCapture(v_path)
    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    if (width == 0) or (height == 0):
        print(v_path, 'not successfully loaded, drop ..'); return
    new_dim = resize_dim(width, height, dim)

    success, image = vidcap.read()
    count = 1
    while success:
        image = cv2.resize(image, new_dim, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(out_dir, 'image_%05d.jpg' % count), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])# quality from 0-100, 95 is default, high is good
        success, image = vidcap.read()
        count += 1

    # Correct the amount of frames
    if (count * 30) < nb_frames:
        nb_frames = int(nb_frames * 30 / 1000)

    if nb_frames > count:
        print('/'.join(out_dir.split('/')[-2::]), 'NOT extracted successfully: %df/%df' % (count, nb_frames))

    vidcap.release()


def resize_dim(w, h, target):
    '''resize (w, h), such that the smaller side is target, keep the aspect ratio'''
    if w >= h:
        return (int(target * w / h), int(target))
    else:
        return (int(target), int(target * h / w)) 


def main_UCF101(v_root, f_root):
    print('extracting UCF101 ... ')
    print('extracting videos from %s' % v_root)
    print('frame save to %s' % f_root)
    
    if not os.path.exists(f_root): os.makedirs(f_root)
    v_act_root = glob.glob(os.path.join(v_root, '*/'))
    print(len(v_act_root))
    for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
        v_paths = glob.glob(os.path.join(j, '*.avi'))
        v_paths = sorted(v_paths)
        Parallel(n_jobs=32)(delayed(extract_video_opencv)(p, f_root) for p in tqdm(v_paths, total=len(v_paths)))


def main_HMDB51(v_root, f_root):
    print('extracting HMDB51 ... ')
    print('extracting videos from %s' % v_root)
    print('frame save to %s' % f_root)
    
    if not os.path.exists(f_root): os.makedirs(f_root)
    v_act_root = glob.glob(os.path.join(v_root, '*/'))
    for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
        v_paths = glob.glob(os.path.join(j, '*.avi'))
        v_paths = sorted(v_paths)
        Parallel(n_jobs=32)(delayed(extract_video_opencv)(p, f_root) for p in tqdm(v_paths, total=len(v_paths)))


def main_JHMDB(v_root, f_root):
    print('extracting JHMDB ... ')
    print('extracting videos from %s' % v_root)
    print('frame save to %s' % f_root)

    if not os.path.exists(f_root): os.makedirs(f_root)
    v_act_root = glob.glob(os.path.join(v_root, '*/'))
    for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
        v_paths = glob.glob(os.path.join(j, '*.avi'))
        v_paths = sorted(v_paths)
        Parallel(n_jobs=32)(delayed(extract_video_opencv)(p, f_root) for p in tqdm(v_paths, total=len(v_paths)))


def main_kinetics400(v_root, f_root, dim=128):
    print('extracting Kinetics400 ... ')
    for basename in ['train', 'val']:
        v_root_real = v_root + '/' + basename
        if not os.path.exists(v_root_real):
            print('Wrong v_root'); sys.exit()

        f_root_real = f_root + '/' + basename
        print('Extract to: \nframe: %s' % f_root_real)
        if not os.path.exists(f_root_real):
            os.makedirs(f_root_real)
        v_act_root = glob.glob(os.path.join(v_root_real, '*/'))
        v_act_root = sorted(v_act_root)

        # if resume, remember to delete the last video folder
        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            v_paths = glob.glob(os.path.join(j, '*.mp4'))
            v_paths = sorted(v_paths)
            # for resume:
            v_class = j.split('/')[-2]
            out_dir = os.path.join(f_root_real, v_class)
            if os.path.exists(out_dir): print(out_dir, 'exists!'); continue
            print('extracting: %s' % v_class)
            # dim = 150 (crop to 128 later) or 256 (crop to 224 later)
            Parallel(n_jobs=32)(delayed(extract_video_opencv)(p, f_root_real, dim=dim) for p in tqdm(v_paths, total=len(v_paths)))


def main_Panasonic(v_root, f_root, dim):
    print('extracting Panasonic ... ')
    print('extracting videos from %s' % v_root)
    print('frame save to %s' % f_root)

    if not os.path.exists(f_root): os.makedirs(f_root)
    v_act_root = glob.glob(os.path.join(v_root, '*/'))
    print(len(v_act_root))
    for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
        v_paths = glob.glob(os.path.join(j, '*.mkv'))
        v_paths = sorted(v_paths)
        Parallel(n_jobs=32)(delayed(extract_video_opencv)(p, f_root, dim) for p in tqdm(v_paths, total=len(v_paths)))


if __name__ == '__main__':
    # v_root is the video source path, f_root is where to store frames
    # edit 'your_path' here:

    parser = argparse.ArgumentParser()
    parser.add_argument('--ucf101', default=False, type=str2bool)
    parser.add_argument('--jhmdb', default=False, type=str2bool)
    parser.add_argument('--hmdb51', default=False, type=str2bool)
    parser.add_argument('--kinetics', default=False, type=str2bool)
    parser.add_argument('--panasonic', default=False, type=str2bool)
    parser.add_argument('--dataset_path', default='/scr/data', type=str)
    parser.add_argument('--dim', default=128, type=int)
    args = parser.parse_args()

    dataset_path = args.dataset_path

    if args.ucf101:
        main_UCF101(v_root=dataset_path + '/ucf101/videos/', f_root=dataset_path + '/ucf101/frame/')

    if args.jhmdb:
        main_JHMDB(v_root=dataset_path + '/jhmdb/videos/', f_root=dataset_path + '/jhmdb/frame/')

    if args.hmdb51:
        main_HMDB51(v_root=dataset_path+'/hmdb/videos', f_root=dataset_path+'/hmdb/frame')

    if args.panasonic:
        main_Panasonic(v_root=dataset_path+'/action_split_data/V1.0', f_root=dataset_path+'/frame', dim=192)

    if args.kinetics:
        if args.dim == 256:
            main_kinetics400(
                v_root=dataset_path + '/kinetics/video', f_root=dataset_path + '/kinetics/frame256', dim=args.dim
            )
        else:
            assert args.dim == 128, "Invalid dim: {}".format(args.dim)
            main_kinetics400(v_root=dataset_path+'/kinetics/video', f_root=dataset_path+'/kinetics/frame', dim=128)

    # main_kinetics400(v_root='your_path/Kinetics400_256/videos',
    #                  f_root='your_path/Kinetics400_256/frame', dim=256)

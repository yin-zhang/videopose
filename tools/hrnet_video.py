'''
通过最新的deep-high-resolution作为2D关键点的获取, 实现高精度的端到端3D姿态重建
'''

import os
import sys
path = os.path.split(os.path.realpath(__file__))[0]
main_path = os.path.join(path, '..')
sys.path.insert(0, main_path)

import numpy as np
import ipdb;pdb = ipdb.set_trace

from common.arguments import parse_args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
import time
from tools.utils import interp_keypoints, scale_keypoints
from tools.save_utils import save_prediction

metadata={'layout_name': 'coco','num_joints': 17,'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15],[2, 4, 6, 8, 10, 12, 14, 16]]}

# record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()

time0 = ckpt_time()

class skeleton():
    def parents(self):
        return np.array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
    def joints_right(self):
        return [1, 2, 3, 9, 10]

def evaluate(test_generator, model_pos, action=None, return_predictions=False):

    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
    with torch.no_grad():
        model_pos.eval()

        N = 0

        for _, batch, batch_2d in test_generator.next_epoch():

            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

def main():
    args = parse_args()

    fps = 25
    cam_w = 1000
    cam_h = 1002
    
    # 2D kpts loads or generate
    if not args.input_npz:
        # 通过hrnet生成关键点
        from joints_detectors.hrnet.pose_estimation.video import generate_kpts
        video_name = args.viz_video
        print('generat keypoints by hrnet...')
        keypoints, fps, cam_w, cam_h = generate_kpts(video_name, no_nan=False)
    else:
        npz = np.load(args.input_npz)
        keypoints = npz['kpts'] #(N, 17, 2)
        fps = npz['fps']
        cam_w = npz['cam_w']
        cam_h = npz['cam_h']

    input_keypoints = interp_keypoints(keypoints)
    keypoints = scale_keypoints(input_keypoints, w=1000, h=1002)

    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    # normlization keypoints  假设use the camera parameter
    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=1000, h=1002)

    model_pos = TemporalModel(17, 2, 17,filter_widths=[3, 3, 3, 3, 3], causal=args.causal, dropout=args.dropout, channels=args.channels,
                                dense=args.dense)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    ckpt, time1 = ckpt_time(time0)
    print('------- load data spends {:.2f} seconds'.format(ckpt))


    # load trained model
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', os.path.join(main_path,chk_filename))
    checkpoint = torch.load(os.path.join(main_path,chk_filename), map_location=lambda storage, loc: storage)# 把loc映射到storage
    model_pos.load_state_dict(checkpoint['model_pos'])


    ckpt, time2 = ckpt_time(time1)
    print('------- load 3D model spends {:.2f} seconds'.format(ckpt))

    #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2 # Padding on each side
    causal_shift = 0

    print('Rendering...')
    gen = UnchunkedGenerator(None, None, [keypoints],
                                pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen,model_pos, return_predictions=True)

    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        save_prediction(args.viz_export, prediction=prediction, input_keypoints=input_keypoints, cam_w=cam_w, cam_h=cam_h)
        
    rot = np.array([ 0.14070565, -0.15007018, -0.7552408 ,  0.62232804], dtype=np.float32)
    prediction = camera_to_world(prediction, R=rot, t=0)

    # We don't have the trajectory, but at least we can rebase the height
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    anim_output = {'Reconstruction': prediction}

    ckpt, time3 = ckpt_time(time2)
    print('------- generate reconstruction 3D data spends {:.2f} seconds'.format(ckpt))

    if args.viz_output is not None:
        from common.visualization import render_animation
        render_animation(input_keypoints, anim_output,
                         skeleton(), fps, args.viz_bitrate, np.array(70., dtype=np.float32), args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(1000, 1002),
                         input_video_skip=args.viz_skip)

    ckpt, time4 = ckpt_time(time3)
    print('total spend {:2f} second'.format(ckpt))
if __name__ == '__main__':
    main()

# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.data
from predict.annot.coco_minival import Mscoco_minival
from predict.p_poseNMS import pose_nms, write_json
import numpy as np
from predict.opt import opt
from tqdm import tqdm
from utils.img import flip_v, shuffleLR_v, vis_frame
from utils.eval import getPrediction, getIntegral7x7Joints
from utils.eval import getmap
import os
import cv2
from models.FastPose import createModel
import argparse
import json

def gaussian(size):
    '''
    Generate a 2D gaussian array
    '''
    sigma = 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = size / 4.0
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g = g[np.newaxis, :]
    return g

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  max(b1_x1, b2_x1)
    inter_rect_y1 =  max(b1_y1, b2_y1)
    inter_rect_x2 =  min(b1_x2, b2_x2)
    inter_rect_y2 =  min(b1_y2, b2_y2)
    
    #Intersection area
    
    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1,0)*max(inter_rect_y2 - inter_rect_y1 + 1, 0)
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

gaussian_kernel = nn.Conv2d(17, 17, kernel_size=4 * 1 + 1,
                            stride=1, padding=2, groups=17, bias=False)

g = torch.from_numpy(gaussian(4 * 1 + 1)).clone()
g = torch.unsqueeze(g, 1)
g = g.repeat(17, 1, 1, 1)
gaussian_kernel.weight.data = g.float()
gaussian_kernel.cuda()

def add_candidate_joints(result, hms, pt1, pt2, inpH, inpW, oupH, oupW):
    for res in result:
        pick = res['pick_idx']
        dist = res['dist']
        can_kps = []
        can_sco = []
        for k in range(hms[pick].shape[0]):
            kps, sco = getIntegral7x7Joints(hms[pick][k], pt1[pick], pt2[pick], inpH, inpW, oupH, oupW)
            min_dst = np.finfo(np.float32).max
            min_idx = 0
            for i in range(kps.shape[0]):
                d = torch.norm(res['keypoints'][k] - kps[i]).item()
                if min_dst > d:
                    min_dst = d
                    min_idx = i                    
                if sco[i] > 0.05:
                    can_kps.append(kps[i])
                    can_sco.append(sco[i])
            if min_dst > dist:
                print('bad match kps {} can {}, min_dist {}, dist {}'.format(res['keypoints'][k], kps[min_idx], min_dst, dist))
            
        res['can_kps'] = can_kps
        res['can_sco'] = can_sco

def select_best_candidate(gt_json, final_result):
    mirror_map = {
            1:2, 2:1, 3:4, 4:3, 5:6, 6:5, 7:8, 8:7, 9:10, 10:9, 11:12, 12:11, 13:14, 14:13, 15:16, 16:15
        }
    
    with open(gt_json, 'r') as f:
        gt = json.load(f)['annotations']
    gt_map = {}
    for g in gt:
        if g['image_id'] in gt_map:
            gt_map[g['image_id']].append(g['keypoints'])
        else:
            gt_map[g['image_id']] = [g['keypoints']]
    gt_image_num = len(gt_map.keys())
    print('Number of groundtruth images:', gt_image_num)

    def compute_pose_dist(pose, pose_gt):
        n = len(pose_gt) // 3
        d = 0
        v = 0
        for i in range(n):
            if pose_gt[i*3+2] > 0:
                d += np.linalg.norm(np.array(pose[i*2:i*2+2]) - np.array(pose_gt[i*3:i*3+2]))
                v += 1
        return d / v if v > 0 else -1

    def match_pose(pose, gt_list):
        mp = gt_list[0]
        if len(gt_list) > 1:
            min_idx = 0
            min_dst = np.finfo(np.float32).max
            for i, gt in enumerate(gt_list):
                d = compute_pose_dist(pose, gt)
                if d > 0 and d < min_dst:
                    min_dst = d
                    min_idx = i
            mp = gt_list[min_idx]
        if check_pose(pose, mp):
            return mp
        else:
            return None

    def get_min_joint(joint, can_joints):
        min_d = np.finfo(np.float32).max
        min_j = None
        min_i = 0
        for i, j in enumerate(can_joints):
            d = np.linalg.norm(joint-j)
            if d < min_d:
                min_d = d
                min_j = j
                min_i = i
        return min_j, min_i

    def select_best_candidate(kps, can_kps, gt_kps):
        n = len(gt_kps) // 3
        assert n == kps.shape[0]
        change = False
        for i in range(n):
            if gt_kps[i*3+2] > 0:                
                if i in mirror_map:
                    if i < mirror_map[i]:
                        can_joints = [kps[i], kps[mirror_map[i]]]
                        if can_kps[i] is not None:
                            can_joints += [c.numpy() for c in can_kps[i]]
                        if can_kps[mirror_map[i]] is not None:
                            can_joints += [c.numpy() for c in can_kps[mirror_map[i]]]]

                        lj, l_idx = get_min_joint(np.array(gt_kps[i*3:i*3+2]), can_joints)
                        rj, r_idx = get_min_joint(np.array(gt_kps[mirror_map[i]*3:mirror_map[i]*3+2]), can_joints)
                        if np.linalg.norm(lj-rj) > 2:
                            kps[i] = lj
                            kps[mirror_map[i]] = rj
                            change |= l_idx != 0 or r_idx != 1
                else:                    
                    if len(can_kps[i]) > 0:
                        can_joints = [kps[i]]
                        can_joints += [c.numpy() for c in can_kps[i]]
                        kps[i], idx = get_min_joint(np.array(gt_kps[i*3:i*3+2]), can_joints)
                        change |= idx != 0

        return kps, change

    def check_pose(pose0, pose_gt, threshold=0.5):
        pose1 = np.array(pose_gt).reshape(-1,3)
        pose_idx = (pose1[:,2] > 0)
        if not np.any(pose_idx):
            return False
        pose0 = pose0[pose_idx]
        pose1 = pose1[:,:2][pose_idx]

        box0_lt = np.min(pose0, axis=0)
        box0_rb = np.max(pose0, axis=0)
        
        box1_lt = np.min(pose1, axis=0)
        box1_rb = np.max(pose1, axis=0)
        
        box0 = np.array([box0_lt[0], box0_lt[1], box0_rb[0], box0_rb[1]])
        box1 = np.array([box1_lt[0], box1_lt[1], box1_rb[0], box1_rb[1]])

        if bbox_iou(box0, box1) < threshold:
            return False
        return True

    rs_image_num = len(final_result)
    print('Number of detected images:', rs_image_num)

    change_num = 0
    all_detected = 0
    for result in final_result:
        image_id = int(result['imgname'].split('/')[-1].split('.')[0].split('_')[-1])
        if image_id not in gt_map: 
            print('image', image_id, 'doesnt have groundtruth')
            continue

        for pose in result['result']:
            all_detected += 1
            kps = pose['keypoints'].numpy()
            best_match_pose = match_pose(kps.reshape(-1), gt_map[image_id])
            if best_match_pose is not None:
                refine_pose, change = select_best_candidate(kps, pose['can_kps'], best_match_pose)
                if change:
                    change_num += 1
                pose['keypoints'] = torch.from_numpy(refine_pose)
            else:
                print('check pose not pass!', best_match_pose)
    print('Number of refined pose: {:d}/{:d}'.format(change_num, all_detected))

def prediction(model, img_folder, boxh5, imglist):
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    dataset = Mscoco_minival(img_folder, boxh5, imglist)
    minival_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=20, pin_memory=True)
    minival_loader_desc = tqdm(minival_loader)

    final_result = []
    tmp_inp = {}
    for i, (inp, box, im_name, metaData) in enumerate(minival_loader_desc):
        #inp = torch.autograd.Variable(inp.cuda(), volatile=True)
        pt1, pt2, ori_inp = metaData
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        if im_name[0] in tmp_inp.keys():
            inps = tmp_inp[im_name[0]]['inps']
            ori_inps = tmp_inp[im_name[0]]['ori_inps']
            boxes = tmp_inp[im_name[0]]['boxes']
            pt1s = tmp_inp[im_name[0]]['pt1s']
            pt2s = tmp_inp[im_name[0]]['pt2s']
            tmp_inp[im_name[0]]['inps'] = torch.cat((inps, inp), dim=0)
            tmp_inp[im_name[0]]['pt1s'] = torch.cat((pt1s, pt1), dim=0)
            tmp_inp[im_name[0]]['pt2s'] = torch.cat((pt2s, pt2), dim=0)
            tmp_inp[im_name[0]]['ori_inps'] = torch.cat(
                (ori_inps, ori_inp), dim=0)
            tmp_inp[im_name[0]]['boxes'] = torch.cat((boxes, box), dim=0)
        else:
            tmp_inp[im_name[0]] = {
                'inps': inp,
                'ori_inps': ori_inp,
                'boxes': box,
                'pt1s': pt1,
                'pt2s': pt2
            }

    for im_name, item in tqdm(tmp_inp.items()):
        inp = item['inps']
        pt1 = item['pt1s']
        pt2 = item['pt2s']
        box = item['boxes']
        ori_inp = item['ori_inps']

        with torch.no_grad():
            try:
                if torch.cuda.is_available():
                    inp = inp.cuda()
                kp_preds = model(inp)
                kp_preds = kp_preds.data[:, :17, :]
            except RuntimeError as e:
                '''
                Divide inputs into two batches
                '''
                # assert str(e) == 'CUDA error: out of memory'
                bn = inp.shape[0]
                inp1 = inp[: bn // 2]
                inp2 = inp[bn // 2:]
                kp_preds1 = model(inp1)
                kp_preds2 = model(inp2)
                kp_preds = torch.cat((kp_preds1, kp_preds2), dim=0)
                kp_preds = kp_preds.data[:, :17, :]

            # kp_preds = gaussian_kernel(F.relu(kp_preds))

            # Get predictions
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)

            preds, preds_img, preds_scores = getPrediction(
                kp_preds.cpu().data, pt1, pt2,
                opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW
            )

            result = pose_nms(box, preds_img, preds_scores)
            
            add_candidate_joints(result, kp_preds.cpu().numpy(), pt1.numpy(), pt2.numpy(), opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
            
            result = {
                'imgname': im_name,
                'result': result
            }
        #img = display_frame(orig_img, result, opt.outputpath)
        #ori_inp = np.transpose(
        #    ori_inp[0][:3].clone().numpy(), (1, 2, 0)) * 255
        #img = vis_frame(ori_inp, result)
        #cv2.imwrite(os.path.join(
        #    './val', 'vis', im_name), img)
        final_result.append(result)
    # np.savez('../../examples/coco_val/final_result.npz', result=result)
    select_best_candidate('../../examples/coco_val/person_keypoints_val2017.json', final_result)
    write_json(final_result, '../../examples/coco_val', for_eval=True)
    return getmap()

if __name__ == '__main__':

    m = createModel()
    assert os.path.exists(opt.loadModel), 'model file {} not exsit'.format(opt.loadModel)

    print('Loading Model from {}'.format(opt.loadModel))
    m.load_state_dict(torch.load(opt.loadModel))
    prediction(m, opt.inputpath, opt.boxh5, opt.inputlist)

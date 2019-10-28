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
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    
    inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1,torch.zeros(inter_rect_x2.shape).cuda())*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
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
            if torch.norm(res['keypoints'][k] - kps[0]).item() > dist:
                print('bad match kps {} can {}'.format(res['keypoints'][k], kps[0]))
            if len(sco) > 1 and sco[1] > 0.05:
                can_kps.append(kps[1])
                can_sco.append(sco[1])
            else:
                can_kps.append(None)
                can_sco.append(0)
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
    
    def compute_pose_dist(pose, pose_gt):
        n = len(pose_gt) // 3
        d = 0
        v = 0
        for i in range(n):
            if pose_gt[i*3+2] > 0:
                d += np.linalg.norm(np.array(pose[i*2:i*2+2]) - np.array(pose_gt[i*3:i*3+2]))
                v += 1
        return d / v

    def match_pose(pose, gt_list):
        if len(gt_list) == 1:
            return gt_list[0]
        else:
            min_idx = 0
            min_dst = np.finfo(np.float32).max
            for i, gt in enumerate(gt_list):
                d = compute_pose_dist(pose, gt)
                if d < min_dst:
                    min_dst = d
                    min_idx = i
            return gt_list[min_idx]

    def get_min_joint(joint, can_joints):
        min_d = np.finfo(np.float32).max
        min_j = None
        for j in can_joints:
            d = np.linalg.norm(joint, j)
            if d < min_d:
                min_d = d
                min_j = j
        return min_j

    def select_best_candidate(kps, can_kps, gt_kps):
        n = len(gt_kps) // 3
        assert n == kps.shape[0]
        for i in range(n):
            if gt_kps[i*3+2] > 0:                
                if i in mirror_map:
                    if i < mirror_map[i]:
                        can_joints = [kps[i], kps[mirror_map[i]]]
                        if can_kps[i] is not None: can_joints.append(can_kps[i])
                        if can_kps[mirror_map[i]] is not None: can_joints.append(can_kps[mirror_map[i]])
                        lj = get_min_joint(np.array(pose_gt[i*3:i*3+2]), can_kps)
                        rj = get_min_joint(np.array(pose_gt[mirror_map[i]*3:mirror_map[i]*3+2]), can_kps)
                        if np.linalg.norm(lj, rj) > 2:
                            kps[i] = lj
                            kps[mirror_map[i]] = rj
                else:
                    if can_kps[i] is not None:
                        min_d = np.linalg.norm(kps[i] - np.array(pose_gt[i*3:i*3+2]))
                        can_d = np.linalg.norm(can_kps[i].numpy() - np.array(pose_gt[i*3:i*3+2]))
                        if can_d < min_d:
                            kps[i] = can_kps[i].numpy()
        return kps

    def check_pose(pose0, pose1, threshold=0.5):
        box0_lt = np.min(pose0, axis=0)
        box0_rb = np.max(pose0, axis=0)
        
        box1_lt = np.min(pose1, axis=0)
        box1_rb = np.max(pose1, axis=0)

        box0 = np.array([box0_lt[0], box0_lt[1], box0_rb[0], box0_rb[1]])
        box1 = np.array([box1_lt[0], box1_lt[1], box1_rb[0], box1_rb[1]])

        if bbox_iou(box0, box1) < threshold:
            return False
        return True

    for result in final_result:
        image_id = int(result['imgname'].split('/')[-1].split('.')[0].split('_')[-1])
        for pose in result['result']:
            kps = pose['keypoints'].numpy()
            best_match_pose = match_pose(kps.reshape(-1), gt_map[image_id])
            if check_pose(kps, np.array(best_match_pose).reshape(-1,3)[:,2]):
                refine_pose = select_best_candidate(kps, pose['can_kps'], best_match_pose)
                pose['keypoints'] = torch.from_numpy(refine_pose)
            else:
                print('check pose not pass!')
    

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
        
    select_best_candidate('../../examples/coco_val/person_keypoints_val2017.json', final_result)
    write_json(final_result, '../../examples/coco_val', for_eval=True)
    return getmap()

if __name__ == '__main__':

    m = createModel()
    assert os.path.exists(opt.loadModel), 'model file {} not exsit'.format(opt.loadModel)

    print('Loading Model from {}'.format(opt.loadModel))
    m.load_state_dict(torch.load(opt.loadModel))
    prediction(m, opt.inputpath, opt.boxh5, opt.inputlist)

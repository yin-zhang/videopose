# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import os
import h5py
from functools import reduce

import torch.utils.data as data
from ..pose import generateSampleBox
from opt import opt


class H36M(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/h36m/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints = 17

        '''
        "0 hips", "1 rightHip", "2 rightKnee", "3 rightFoot"
        "4 leftHip", "5 leftKnee", "6 leftFoot",
        "7 spine", "8 thorax", "9 nose", "10 head"
        "11 leftShoulder", "12 leftElbow", "13 leftWrist",
        "14 rightShoulder", "15 rightElbow", "16 rightWrist"
        '''
        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2,5), (3,6), (4,7), (12,15),(13,16),(14,17))
        self.joint_names = {
            "0 hips", "1 rightHip", "2 rightKnee", "3 rightFoot"
            "4 leftHip", "5 leftKnee", "6 leftFoot",
            "7 spine", "8 thorax", "9 nose", "10 head"
            "11 leftShoulder", "12 leftElbow", "13 leftWrist",
            "14 rightShoulder", "15 rightElbow", "16 rightWrist"}
        
        # create train/val split
        with h5py.File('../data/h36m/annot_h36m.h5', 'r') as annot:
            # train
            self.imgname_coco_train = annot['imgname'][:1000983]
            self.bndbox_coco_train = annot['bndbox'][:1000983]
            self.part_coco_train = annot['part'][:1000983]
            # val
            '''
            self.imgname_coco_val = annot['imgname'][1000983:]
            self.bndbox_coco_val = annot['bndbox'][1000983:]
            self.part_coco_val = annot['part'][1000983:]
            '''
            self.imgname_coco_val = annot['imgname'][:10000]
            self.bndbox_coco_val = annot['bndbox'][:10000]                                                                                               
            self.part_coco_val = annot['part'][:10000]

        self.size_train = self.imgname_coco_train.shape[0]
        self.size_val = self.imgname_coco_val.shape[0]

    def __getitem__(self, index):
        sf = self.scale_factor

        if self.is_train:
            part = self.part_coco_train[index]
            bndbox = self.bndbox_coco_train[index]
            imgname = self.imgname_coco_train[index]
        else:
            part = self.part_coco_val[index]
            bndbox = self.bndbox_coco_val[index]
            imgname = self.imgname_coco_val[index]
        imgname = imgname.decode().split('/')[-2:]
        imgname[1] = str(int(imgname[1].split('.')[0].split('_')[-1])) + '.' +  imgname[1].split('.')[1]
        img_path = os.path.join(self.img_folder, imgname[0], imgname[1])

        metaData = generateSampleBox(img_path, bndbox.reshape(1,-1), part, self.nJoints,
                                     'h36m', sf, self, train=self.is_train)

        inp, out, setMask = metaData

        return inp, out, setMask, 'h36m'

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val

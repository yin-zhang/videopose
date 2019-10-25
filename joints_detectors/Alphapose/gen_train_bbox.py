import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader import ImageLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

import os
import sys
from tqdm import tqdm
import time
from fn import getTime

from pPose_nms import pose_nms, write_json

args = opt

if __name__ == "__main__":
    inputpath = args.inputpath
    inputlist = args.inputlist
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    if len(inputlist):
        im_names = open(inputlist, 'r').readlines()
    elif len(inputpath) and inputpath != '/':
        for root, dirs, files in os.walk(inputpath):
            im_names = files
    else:
        raise IOError('Error: must contain either --indir/--list')

    # Load input images
    data_loader = ImageLoader(im_names, batchSize=args.detbatch, format='yolo').start()

    # Load detection loader 先找到人
    print('Loading YOLO model..')
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
    det_processor = DetectionProcessor(det_loader, output_cropimg=False).start()
        
    data_len = data_loader.length()
    im_names_desc = tqdm(range(data_len))

    imgname_list, boxes_list = [], []
    batchSize = args.posebatch
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
            imgname_list.append(im_name)
            boxes_list.append(boxes.cpu().numpy() if boxes is not None else boxes)

    np.savez(os.path.join(args.outputpath, 'bbox.npz'), images=imgname_list, boxes=boxes_list)

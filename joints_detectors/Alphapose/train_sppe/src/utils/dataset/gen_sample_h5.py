import h5py
import json
import logging
import os
import numpy as np
import argparse

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def cvt_bbox2evalformat(bbox_path, out_image_path, out_h5):
    '''
    Convert output npz file to an image list txt and a bndbox h5 for evaluation (train_sppe/src/evaluation.py)
    '''
    bboxes = np.load(bbox_path, allow_pickle=True)
    b_images = bboxes['images']
    b_boxes = bboxes['boxes']
    assert len(b_images) == len(b_boxes), 'boxes length: {} images length {}'.format(len(b_boxes), len(b_images))

    imglist_file = open(out_image_path, 'w')
    bb_h5 = h5py.File(out_h5, 'w')

    bb_map = {'xmin':[], 'ymin':[], 'xmax':[], 'ymax':[]}
    for i in range(len(b_images)):
        if b_boxes[i] is not None:
            for box in b_boxes[i]:
                bb['xmin'].append(box[0])
                bb['ymin'].append(box[1])
                bb['xmax'].append(box[2])
                bb['ymax'].append(box[3])
                imglist_file.write(b_images[i] + '\n')
    bb_h5['xmin'] = bb['xmin']
    bb_h5['ymin'] = bb['ymin']
    bb_h5['xmax'] = bb['xmax']
    bb_h5['ymax'] = bb['ymax']

    imglist_file.close()
    bb_h5.close()

def gen_h36m_h5(train_sample_path, val_sample_path, output_path):
    '''
    Convert h36m annotation file to input h5 format for detection
    '''
    def append_samples(path, imgname_list, part_list, bndbox_list):
        logger.info('Process {}'.format(path))
        with open(path, 'r') as f:
            ann = json.load(f)
            num = len(ann['images'])
            for i in range(num):
                imgname = ann['images'][i]['file_name']
                part = np.array(ann['annotations'][i]['keypoints_img'])

                part_vis = ann['annotations'][i]['keypoints_vis']
                for j in range(len(part)):
                    if not part_vis[j]:
                        part[j] *= -1
                bndbox = np.array(ann['annotations'][i]['bbox'])

                imgname_list.append(imgname.encode())
                part_list.append(part)
                bndbox_list.append(bndbox)
    
    imgname_list, part_list, bndbox_list = [], [], []
    for path in train_sample_path:
        append_samples(path, imgname_list, part_list, bndbox_list)
    logger.info('Number of train samples {}'.format(len(imgname_list)))
    for path in val_sample_path:
        append_samples(path, imgname_list, part_list, bndbox_list)
    logger.info('Number of all samples {}'.format(len(imgname_list)))
    assert len(imgname_list) == len(part_list) and len(imgname_list) == len(bndbox_list)

    for i in range(len(part_list)-1):
        assert part_list[i].shape == part_list[i+1].shape, 'shape_{:d} {} shape_{:d} {}'.format(i, part_list[i].shape, i+1, part_list[i+1].shape)

    f = h5py.File(output_path, 'w')
    f['imgname'] = imgname_list
    f['part'] = part_list
    f['bndbox'] = bndbox_list
    f.close()

def merge_dtbox_gtjoints(box_npz, joint_h5, output_path):
    '''
    Merge bounding box output file with joints' groundtruth for training sppe.
    '''
    boxes = np.load(box_npz, allow_pickle=True)
    jinfo = h5py.File(joint_h5, 'r')
    b_images = boxes['images']
    b_boxes = boxes['boxes']
    j_images = jinfo['imgname']
    j_joints = jinfo['part']
    j_bndbox = jinfo['bndbox']

    assert len(b_images) == len(j_images), 'detected images count {:d} groundtruth images count {:d}'.format(len(b_images), len(j_images))
    images_num = len(b_images)

    out_images, out_part, out_bndbox = [], [], []
    for i in range(images_num):
        if b_boxes[i] is not None:
            for b in range(len(b_boxes[i])):
                i_lt = np.min(b_boxes[i][b][:2], j_bndbox[i][:2])
                i_rb = np.max(b_boxes[i][b][2:], j_bndbox[i][2:])
                it_area = (i_rb[0] - i_lt[0]) * (i_rb[1] - i_lt[1])
                if it_area > 0:
                    a_wh = b_boxes[i][b][2:] - b_boxes[i][b][:2]
                    b_wh = j_bndbox[i][2:] - j_bndbox[i][:2]
                    if it_area / (a_wh[0] * a_wh[1] + b_wh[0] * b_wh[1]) > 0.6:
                        out_images.append(j_images[i])
                        out_bndbox.append(b_boxes[i][b])
                        out_part.append(j_joints[i])
    f = h5py.File(output_path, 'w')
    f['imgname'] = out_images
    f['part'] = out_part
    f['bndbox'] = out_bndbox
    f.close()

h36m_config = {
    'train_list':['Human36M_subject1.json', 'Human36M_subject5.json','Human36M_subject6.json', 'Human36M_subject7.json', 'Human36M_subject8.json'],
    'val_list':['Human36M_subject9.json', 'Human36M_subject11.json'],    
}

def parse_args():
    parser = argparse.ArgumentParser(description='Preparation of data')
    parser.add_argument('-o', '--output', type=str, help='output file path')
    parser.add_argument('--detect-box-path', type=str, help='bounding box npz file outputed by gen_train_bbox.py')
    parser.add_argument('--gt-joint-path', type=str, help='joints groundtruth file path')
    parser.add_argument('-m', '--mode', type=str, help='mode, h36m2h5|merge_box_gt|box2eval')
    parser.add_argument('-d', '--dir', type=str, help='data directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger.info(args)
    
    if args.mode == 'h36m2h5':
        train_list = [os.path.join(args.dir, p) for p in h36m_config['train_list']]
        val_list = [os.path.join(args.dir, p) for p in h36m_config['val_list']]
        gen_h36m_h5(train_list, val_list, args.output)
    elif args.mode == 'merge_box_gt':
        merge_dtbox_gtjoints(args.detect_box_path, args.gt_joint_path, args.output)
    elif args.mode == 'box2eval':
        imglist_path, bbh5_path = args.output.split(',')
        cvt_bbox2evalformat(args.detect_box_path, imglist_path, bbh5_path)
    
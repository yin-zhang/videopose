import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import json
import math
from functools import partial

from config import cfg
from tfflat.base import ModelDesc

from nets.basemodel import resnet50, resnet101, resnet152, resnet_arg_scope, resnet_v1
resnet_arg_scope = partial(resnet_arg_scope, bn_trainable=cfg.bn_train)

class Model(ModelDesc):
    
    def head_net(self, blocks, is_training, trainable=True, add_paf_output=False):
        
        normal_initializer = tf.truncated_normal_initializer(0, 0.01)
        msra_initializer = tf.contrib.layers.variance_scaling_initializer()
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        
        with slim.arg_scope(resnet_arg_scope(bn_is_training=is_training)):
            
            out = slim.conv2d_transpose(blocks[-1], 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up1')
            out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up2')
            out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up3')

            if add_paf_output:
                hms_out = slim.conv2d(out, cfg.num_kps, [1, 1],
                    trainable=trainable, weights_initializer=msra_initializer,
                    padding='SAME', normalizer_fn=None, activation_fn=None,
                    scope='out')
                paf_out = slim.conv2d(out, len(cfg.kps_lines)*2, [1, 1],
                    trainable=trainable, weights_initializer=msra_initializer,
                    padding='SAME', normalizer_fn=None, activation_fn=None,
                    scope='paf')
                
                out = (hms_out, paf_out)
            else:
                out = slim.conv2d(out, cfg.num_kps, [1, 1],
                    trainable=trainable, weights_initializer=msra_initializer,
                    padding='SAME', normalizer_fn=None, activation_fn=None,
                    scope='out')
        return out

    def find_peaks(hm, threshold=0.05):
        w, h = hm.shape[1], hm.shape[0]
        peak = (hm > threshold)[1:-1,1:-1]
        peak = peak & (hm[1:-1,1:-1] > hm[:-2,  1:-1])
        peak = peak & (hm[1:-1,1:-1] > hm[:-2,  :-2])
        peak = peak & (hm[1:-1,1:-1] > hm[1:-1:,:-2])
        peak = peak & (hm[1:-1,1:-1] > hm[2:,   :-2]) 
        peak = peak & (hm[1:-1,1:-1] > hm[2:,   1:-1])
        peak = peak & (hm[1:-1,1:-1] > hm[2:,   2:])
        peak = peak & (hm[1:-1,1:-1] > hm[1:-1, 2:])
        peak = peak & (hm[1:-1,1:-1] > hm[:-2,  2:])
        peak = np.pad(peak, ((1,1),(1,1)), mode='constant', constant_values=False)

        peak_val = hm[peak]
        peak_idx = np.array(peak.nonzero()).astype(np.float32)
        
        # average 7x7 around peak as final peak
        for i in range(peak_idx.shape[1]):
            pY, pX = int(peak_idx[0,i]), int(peak_idx[1,i])
            f = 7
            sum_score, sum_x, sum_y = 0, 0, 0
            for x in range(f):
                for y in range(f):
                    tx = pX - f // 2 + x
                    ty = pY - f // 2 + y
                    if tx < 0 or tx >= w or ty <= 0 or ty >= h: continue
                    score = hm[ty,tx]
                    sum_score += score
                    sum_x += tx * score
                    sum_y += ty * score
            peak_idx[0,i] = sum_y / sum_score
            peak_idx[1,i] = sum_x / sum_score

        
        return peak_idx[::-1,:].T, peak_val

    def extract_coordinate_paf(self, heatmap_outs, paf_outs):
        can_idx_list, can_val_list = [], []
        for i in range(len(heatmap_outs)):
            peak_idx, peak_val = self.find_peaks(heatmap_outs[i])
            can_idx_list.append(peak_idx)
            can_val_list.append(peak_val)

        height = heatmap_outs[0].shape[0]
        subset_size = len(heatmap_outs) + 2
        subset = []
        for line_idx in range(len(cfg.kps_lines)):
            indexA, indexB = cfg.kps_lines[line_idx]
            candA = can_idx_list[indexA]
            candB = can_idx_list[indexB]
            nA = candA.shape[0]
            nB = candB.shape[0]

            if nA == 0 or nB == 0:
                if nA == 0:
                    for i in range(nB):
                        num = False
                        for j in range(len(subset)):
                            if subset[j][indexB] == i:
                                num = True
                                break
                        if not num:
                            subset.append([-1]*subset_size)
                            subset[-1][indexB] = i
                            subset[-1][-1] = 1
                            subset[-1][-2] = can_val_list[indexB][i]
                else:
                    for i in range(nA):
                        num = False
                        for j in range(len(subset)):
                            if subset[j][indexA] == i:
                                num = True
                                break
                        if not num:
                            subset.append([-1]*subset_size)
                            subset[-1][indexA] = i
                            subset[-1][-1] = 1
                            subset[-1][-2] = can_val_list[indexA][i]
            else:
                paf = paf_outs[line_idx*2:line_idx*2+2]
                for i in range(nA):
                    for j in range(nB):
                        vec = candB[j] - candA[i] 
                        vecNorm = np.linalg.norm(vec)
                        vec = vec / vecNorm

                        num_inter = 10
                        p_sum = 0
                        p_count = 0
                        mX = np.round(np.linspace(candA[i][0], candB[j][0], num_inter)).astype(np.int32)
                        mY = np.round(np.linspace(candB[i][1], candB[j][1], num_inter)).astype(np.int32)

                        for lm in range(num_inter):
                            direct = paf[:, mY, mX]
                            score = vec[0] * direct[1] + vec[1] * direct[0]
                            if score > 0.05:
                                p_sum += score
                                p_count += 1

                        suc_ratio = p_count / num_inter
                        mid_score = p_sum / p_count + min(height_n/vecNorm-1, 0)

                        if mid_score > 0 and suc_ratio > 0.8:
                            score = mid_score
                            temp.append((i, j, score))
                        
                if len(temp) > 0:
                    temp = np.array(temp, dtype=[('x',int), ('y',int), ('score', float)])
                    temp = np.sort(temp, dtype='score')[::-1]

                connectionK = []
                occurA = [0] * nA
                occurB = [0] * nB
                counter = 0
                for row in range(len(temp)):
                    x,y,score = temp[row]
                    
                    if occurA[x] == 0 and occurB[y] == 0:
                        connectionK.append((x, y, score))
                        counter += 1
                        if counter == min(nA, nB):
                            break
                        occurA[x] = 1
                        occurB[y] = 1

                def gen_data(i):
                    hm_score_indexA = can_val_list[indexA][connectionK[i][0]]
                    hm_score_indexB = can_val_list[indexB][connectionK[i][1]]
                    data = [-1] * subset_size
                    data[indexA] = connectionK[i][0]
                    data[indexB] = connectionK[i][1]
                    data[-1] = 2
                    data[-2] = connectionK[i][2] + hm_score_indexA + hm_score_indexB
                
                if line_idx == 0:
                    for i in range(len(connectionK)):
                        subset.append(gen_data(i))
                else:
                    for i in range(len(connectionK)):
                        num = 0
                        for j in range(len(subset)):
                            if subset[j][indexA] == connectionK[i][0]:
                                num += 1
                                hm_score_indexB = can_val_list[indexB][connectionK[i][1]]
                                subset[j][indexB] = connectionK[i][1]
                                subset[j][-1] = subset[j][-1] + 1
                                subset[j][-2] = subset[j][-2] + hm_score_indexB + connectionK[i][2]
                        if num == 0:
                            subset.append(gen_data(i))

    def extract_coordinate(self, heatmap_outs):
        shape = heatmap_outs.get_shape().as_list()
        batch_size = tf.shape(heatmap_outs)[0]
        height = shape[1]
        width = shape[2]
        output_shape = (height, width)
        
        # coordinate extract from output heatmap
        y = [i for i in range(output_shape[0])]
        x = [i for i in range(output_shape[1])]
        xx, yy = tf.meshgrid(x, y)
        xx = tf.to_float(xx) + 1
        yy = tf.to_float(yy) + 1
        
        heatmap_outs = tf.reshape(tf.transpose(heatmap_outs, [0, 3, 1, 2]), [batch_size, cfg.num_kps, -1])
        heatmap_outs = tf.nn.softmax(heatmap_outs)
        heatmap_outs = tf.transpose(tf.reshape(heatmap_outs, [batch_size, cfg.num_kps, output_shape[0], output_shape[1]]), [0, 2, 3, 1])

        x_out = tf.reduce_sum(tf.multiply(heatmap_outs, tf.tile(tf.reshape(xx,[1, output_shape[0], output_shape[1], 1]), [batch_size, 1, 1, cfg.num_kps])), [1,2])
        y_out = tf.reduce_sum(tf.multiply(heatmap_outs, tf.tile(tf.reshape(yy,[1, output_shape[0], output_shape[1], 1]), [batch_size, 1, 1, cfg.num_kps])), [1,2])
        coord_out = tf.concat([tf.reshape(x_out, [batch_size, cfg.num_kps, 1])\
            ,tf.reshape(y_out, [batch_size, cfg.num_kps, 1])]\
                    , axis=2)
        coord_out = coord_out - 1

        coord_out = coord_out / output_shape[0] * cfg.input_shape[0]

        return coord_out
 
    def render_onehot_heatmap(self, coord, output_shape):
        
        batch_size = tf.shape(coord)[0]

        x = tf.reshape(coord[:,:,0] / cfg.input_shape[1] * output_shape[1],[-1])
        y = tf.reshape(coord[:,:,1] / cfg.input_shape[0] * output_shape[0],[-1])
        x_floor = tf.floor(x)
        y_floor = tf.floor(y)

        x_floor = tf.clip_by_value(x_floor, 0, output_shape[1] - 1)  # fix out-of-bounds x
        y_floor = tf.clip_by_value(y_floor, 0, output_shape[0] - 1)  # fix out-of-bounds y

        indices_batch = tf.expand_dims(tf.to_float(\
                tf.reshape(
                tf.transpose(\
                tf.tile(\
                tf.expand_dims(tf.range(batch_size),0)\
                ,[cfg.num_kps,1])\
                ,[1,0])\
                ,[-1])),1)
        indices_batch = tf.concat([indices_batch, indices_batch, indices_batch, indices_batch], axis=0)
        indices_joint = tf.to_float(tf.expand_dims(tf.tile(tf.range(cfg.num_kps),[batch_size]),1))
        indices_joint = tf.concat([indices_joint, indices_joint, indices_joint, indices_joint], axis=0)
        
        indices_lt = tf.concat([tf.expand_dims(y_floor,1), tf.expand_dims(x_floor,1)], axis=1)
        indices_lb = tf.concat([tf.expand_dims(y_floor+1,1), tf.expand_dims(x_floor,1)], axis=1)
        indices_rt = tf.concat([tf.expand_dims(y_floor,1), tf.expand_dims(x_floor+1,1)], axis=1)
        indices_rb = tf.concat([tf.expand_dims(y_floor+1,1), tf.expand_dims(x_floor+1,1)], axis=1)

        indices = tf.concat([indices_lt, indices_lb, indices_rt, indices_rb], axis=0)
        indices = tf.cast(tf.concat([indices_batch, indices, indices_joint], axis=1),tf.int32)

        prob_lt = (1 - (x - x_floor)) * (1 - (y - y_floor))
        prob_lb = (1 - (x - x_floor)) * (y - y_floor)
        prob_rt = (x - x_floor) * (1 - (y - y_floor))
        prob_rb = (x - x_floor) * (y - y_floor)
        probs = tf.concat([prob_lt, prob_lb, prob_rt, prob_rb], axis=0)

        heatmap = tf.scatter_nd(indices, probs, (batch_size, *output_shape, cfg.num_kps))
        normalizer = tf.reshape(tf.reduce_sum(heatmap,axis=[1,2]),[batch_size,1,1,cfg.num_kps])
        normalizer = tf.where(tf.equal(normalizer,0),tf.ones_like(normalizer),normalizer)
        heatmap = heatmap / normalizer
        
        return heatmap 
  
    def render_gaussian_heatmap(self, coord, output_shape, sigma, valid=None):
        
        x = [i for i in range(output_shape[1])]
        y = [i for i in range(output_shape[0])]
        xx,yy = tf.meshgrid(x,y)
        xx = tf.reshape(tf.to_float(xx), (1,*output_shape,1))
        yy = tf.reshape(tf.to_float(yy), (1,*output_shape,1))
              
        x = tf.reshape(coord[:,:,0],[-1,1,1,cfg.num_kps]) / cfg.input_shape[1] * output_shape[1]
        y = tf.reshape(coord[:,:,1],[-1,1,1,cfg.num_kps]) / cfg.input_shape[0] * output_shape[0]

        heatmap = tf.exp(-(((xx-x)/tf.to_float(sigma))**2)/tf.to_float(2) -(((yy-y)/tf.to_float(sigma))**2)/tf.to_float(2))

        if valid is not None:
            valid_mask = tf.reshape(valid, [-1, 1, 1, cfg.num_kps])
            heatmap = heatmap * valid_mask

        return heatmap * 255.
   
    def make_network(self, is_train, add_paf_loss=False):
        if is_train:
            image = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.input_shape, 3])
            target_coord = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_kps, 2])
            input_pose_coord = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_kps, 2])
            target_valid = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_kps])
            input_pose_valid = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_kps])
            
            if add_paf_loss:
                target_paf_valid = tf.placeholder(tf.float32, shape=[cfg.batch_size, len(cfg.kps_lines)*2])
                target_paf = tf.placeholder(tf.float32, shape=[cfg.batch_size, len(cfg.kps_lines)*2, cfg.output_shape[0], cfg.output_shape[1]])
                self.set_inputs(image, target_coord, input_pose_coord, target_valid, input_pose_valid, target_paf, target_paf_valid)
            else:
                self.set_inputs(image, target_coord, input_pose_coord, target_valid, input_pose_valid)
        else:
            image = tf.placeholder(tf.float32, shape=[None, *cfg.input_shape, 3])
            input_pose_coord = tf.placeholder(tf.float32, shape=[None, cfg.num_kps, 2])
            input_pose_valid = tf.placeholder(tf.float32, shape=[None, cfg.num_kps])
            self.set_inputs(image, input_pose_coord, input_pose_valid)

        input_pose_hm = tf.stop_gradient(self.render_gaussian_heatmap(input_pose_coord, cfg.input_shape, cfg.input_sigma, input_pose_valid))
        backbone = eval(cfg.backbone)
        resnet_fms = backbone([image, input_pose_hm], is_train, bn_trainable=True)
        if add_paf_loss:
            heatmap_outs, paf_outs = self.head_net(resnet_fms, is_train, add_paf_output=add_paf_loss)
        else:
            heatmap_outs = self.head_net(resnet_fms, is_train)
        
        if is_train:
            
            if add_paf_loss:
                gt_heatmap = tf.stop_gradient(self.render_gaussian_heatmap(target_coord, cfg.output_shape, 1) / 255.0)
                valid_mask = tf.reshape(target_valid, [cfg.batch_size, cfg.num_kps])

                loss_hm = tf.mean_squared_error(gt_heatmap * valid_mask, heatmap_outs * valid_mask)
                loss_paf = tf.mean_squared_error(target_paf * target_paf_valid, paf_outs * target_paf_valid)
                loss = loss_hm + loss_paf

                self.add_tower_summary('loss_h', loss_hm)
                self.add_tower_summary('loss_p', loss_paf)

                self.set_loss(loss)
            else:
                gt_heatmap = tf.stop_gradient(tf.reshape(tf.transpose(\
                        self.render_onehot_heatmap(target_coord, cfg.output_shape),\
                        [0, 3, 1, 2]), [cfg.batch_size, cfg.num_kps, -1]))
                gt_coord = target_coord / cfg.input_shape[0] * cfg.output_shape[0]

                # heatmap loss
                out = tf.reshape(tf.transpose(heatmap_outs, [0, 3, 1, 2]), [cfg.batch_size, cfg.num_kps, -1])
                gt = gt_heatmap
                valid_mask = tf.reshape(target_valid, [cfg.batch_size, cfg.num_kps])
                loss_heatmap = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=out) * valid_mask)

                # coordinate loss
                out = self.extract_coordinate(heatmap_outs) / cfg.input_shape[0] * cfg.output_shape[0]
                gt = gt_coord
                valid_mask = tf.reshape(target_valid, [cfg.batch_size, cfg.num_kps, 1])
                loss_coord = tf.reduce_mean(tf.abs(out - gt) * valid_mask)

                loss = loss_heatmap + loss_coord + loss_paf

                self.add_tower_summary('loss_h', loss_heatmap)
                self.add_tower_summary('loss_c', loss_coord)
                self.add_tower_summary('loss_p', loss_paf)

                self.set_loss(loss)
            
        else:
            out = self.extract_coordinate(heatmap_outs)
            self.set_outputs(out)
            self.set_heatmaps(tf.transpose(heatmap_outs, [0, 3, 1, 2]))

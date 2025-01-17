# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torch
import torch.utils.data
from torch.utils.data.sampler import RandomSampler
from utils.dataset import coco, h36m
from opt import opt
from tqdm import tqdm
from models.FastPose import createModel
from utils.eval import DataLogger, accuracy
from utils.img import flip_v, shuffleLR_v, vis_heatmap, torch_to_im
from evaluation import prediction
from tensorboardX import SummaryWriter
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3,4,5,6,7'

def train(train_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.train()

    train_loader_desc = tqdm(train_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(train_loader_desc):
        inps = inps.cuda().requires_grad_()
        labels = labels.cuda()
        setMask = setMask.cuda()
        out = m(inps)

        loss = criterion(out.mul(setMask), labels)

        acc = accuracy(out.data.mul(setMask), labels.data, train_loader.dataset)

        accLogger.update(acc[0], inps.size(0))
        lossLogger.update(loss.item(), inps.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opt.trainIters += 1
        # Tensorboard
        writer.add_scalar(
            'Train/Loss', lossLogger.avg, opt.trainIters)
        writer.add_scalar(
            'Train/Acc', accLogger.avg, opt.trainIters)

        # TQDM
        train_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f}'.format(
                loss=lossLogger.avg,
                acc=accLogger.avg * 100)
        )

    train_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def valid(val_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.eval()

    val_loader_desc = tqdm(val_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(val_loader_desc):
        inps = inps.cuda()
        labels = labels.cuda()
        setMask = setMask.cuda()

        with torch.no_grad():
            out = m(inps)

            loss = criterion(out.mul(setMask), labels)

            #flip_out = m(flip_v(inps, cuda=True))
            #flip_out = flip_v(shuffleLR_v(
            #    flip_out, val_loader.dataset, cuda=True), cuda=True)

            #out = (flip_out + out) / 2

        acc = accuracy(out.mul(setMask), labels, val_loader.dataset)

        lossLogger.update(loss.item(), inps.size(0))
        accLogger.update(acc[0], inps.size(0))

        opt.valIters += 1

        # Tensorboard
        writer.add_scalar(
            'Valid/Loss', lossLogger.avg, opt.valIters)
        writer.add_scalar(
            'Valid/Acc', accLogger.avg, opt.valIters)

        val_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f}'.format(
                loss=lossLogger.avg,
                acc=accLogger.avg * 100)
        )

    val_loader_desc.close()

    return lossLogger.avg, accLogger.avg

def show_loader_image(output_dir, loader, count = 10, joint_names=None):
    for i, (inps, labels, setMask, imgset) in enumerate(loader):
        if i > 10:
            break
            
        hms = labels.mul(setMask)
        for j in range(inps.shape[0]):
            print('save', i, j)
            path = os.path.join(output_dir, '{:d}_{:d}.jpg'.format(i, j))
            vis_heatmap(hms[j].numpy(), path, c=5, img_res=torch_to_im(inps[j]), joint_names=None)
            

def main():
    print(opt)
    # Model Initialize
    
    m = createModel().cuda()
    if opt.loadModel:
        print('Loading Model from {}'.format(opt.loadModel))

        '''
        ckp = torch.load(opt.loadModel)        
        for name,param in m.state_dict().items():
            if name in ckp:
                ckp_param = ckp[name]
                if ckp_param.shape == param.shape:
                    param.copy_(ckp_param)
                    print(name, 'copy successfully')
                else:
                    print(name, 'shape is inconsistent with checkpoint')
            else:
                print(name, 'can not be found in checkpoint')
        '''

        m.load_state_dict(torch.load(opt.loadModel))

        if not os.path.exists("../exp/{}/{}".format(opt.dataset, opt.expID)):
            try:
                os.mkdir("../exp/{}/{}".format(opt.dataset, opt.expID))
            except FileNotFoundError:
                os.mkdir("../exp/{}".format(opt.dataset))
                os.mkdir("../exp/{}/{}".format(opt.dataset, opt.expID))
    else:
        print('Create new model')
        #  import pdb;pdb.set_trace()
        if not os.path.exists("../exp/{}/{}".format(opt.dataset, opt.expID)):
            try:
                os.mkdir("../exp/{}/{}".format(opt.dataset, opt.expID))
            except FileNotFoundError:
                os.mkdir("../exp/{}".format(opt.dataset))
                os.mkdir("../exp/{}/{}".format(opt.dataset, opt.expID))

    criterion = torch.nn.MSELoss().cuda()

    if opt.optMethod == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(),
                                        lr=opt.LR,
                                        momentum=opt.momentum,
                                        weight_decay=opt.weightDecay)
    elif opt.optMethod == 'adam':
        optimizer = torch.optim.Adam(
            m.parameters(),
            lr=opt.LR
        )
    elif opt.optMethod == 'rmsprop_refine':
        print('opt rmsprop refine')
        optimizer = torch.optim.RMSprop(
            [
                {"params": m.preact.parameters(), "lr":1e-5},
                {"params": m.suffle1.parameters(), "lr":1e-5},
                {"params": m.duc1.parameters(), "lr":1e-5},
                {"params": m.duc2.parameters(), "lr":1e-5},
                {"params": m.conv_out.parameters(), "lr":1e-4}
            ],
            lr=opt.LR,
            momentum=opt.momentum,
            weight_decay=opt.weightDecay)
    else:
        raise Exception
    if opt.loadOptimizer:
        optimizer = torch.load(opt.loadOptimizer)
        
    writer = SummaryWriter(
        '.tensorboard/{}/{}'.format(opt.dataset, opt.expID))
    
    # Prepare Dataset
    if opt.dataset == 'coco':
        train_dataset = coco.Mscoco(train=True)
        val_dataset = coco.Mscoco(train=False)
    elif opt.dataset == 'h36m':
        train_dataset = h36m.H36M(train=True)
        val_dataset = h36m.H36M(train=False)


    '''
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.trainBatch, shuffle=False, sampler=RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset)//10), num_workers=opt.nThreads, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.validBatch, shuffle=False, sampler=RandomSampler(val_dataset, replacement=True, num_samples=len(val_dataset)//10), num_workers=opt.nThreads, pin_memory=True)
    '''
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.trainBatch, shuffle=True, num_workers=opt.nThreads, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.validBatch, shuffle=False, sampler=RandomSampler(val_dataset, replacement=True, num_samples=len(val_dataset)//8), num_workers=opt.nThreads, pin_memory=True)
    #show_loader_image('train_check_images', train_loader, joint_names=train_dataset.joint_names)
    #show_loader_image('valid_check_images', val_loader, joint_names=val_dataset.joint_names)
    #return

    # Model Transfer
    m = torch.nn.DataParallel(m).cuda()

    # Start Training
    for i in range(opt.nEpochs):
        opt.epoch = i

        print('############# Starting Epoch {} #############'.format(opt.epoch))
        loss, acc = train(train_loader, m, criterion, optimizer, writer)

        print('Train-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=opt.epoch,
            loss=loss,
            acc=acc
        ))

        opt.acc = acc
        opt.loss = loss
        m_dev = m.module
        if i % opt.snapshot == 0:
            torch.save(
                m_dev.state_dict(), '../exp/{}/{}/model_{}.pkl'.format(opt.dataset, opt.expID, opt.epoch))
            torch.save(
                opt, '../exp/{}/{}/option.pkl'.format(opt.dataset, opt.expID, opt.epoch))
            torch.save(
                optimizer, '../exp/{}/{}/optimizer.pkl'.format(opt.dataset, opt.expID))

        loss, acc = valid(val_loader, m, criterion, optimizer, writer)

        print('Valid-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=i,
            loss=loss,
            acc=acc
        ))

        '''
        if opt.dataset != 'mpii':
            with torch.no_grad():
                mAP, mAP5 = prediction(m)

            print('Prediction-{idx:d} epoch | mAP:{mAP:.3f} | mAP0.5:{mAP5:.3f}'.format(
                idx=i,
                mAP=mAP,
                mAP5=mAP5
            ))
        '''
    writer.close()


if __name__ == '__main__':
    main()

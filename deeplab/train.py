import os
import time
import logging
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils2 import create_logger, AverageMeter, accuracy, save_checkpoint, load_state, IterLRScheduler, DistributedGivenIterationSampler, simple_group_split
from torchE.D import dist_init, average_gradients, DistModule
from tensorboardX import SummaryWriter

# from pspnet import PSPNet
from utils import AverageMeter, poly_learning_rate, intersectionAndUnion
import voc12.data
from tool import pyutils, imutils, torchutils
from torchvision import transforms

# Setup
def get_parser():
    parser = ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--backbone', type=str, default='resnet', help='backbone network type')
    parser.add_argument('--layers', type=int, default=50, help='layers number of based resnet')
    parser.add_argument('--syncbn', type=int, default=1, help='adopt syncbn or not')
    parser.add_argument('--classes', type=int, default=21, help='number of classes')
    parser.add_argument('--crop_h', type=int, default=473, help='train crop size h')
    parser.add_argument('--crop_w', type=int, default=473, help='train crop size w')
    parser.add_argument('--scale_min', type=float, default=0.5, help='minimum random scale')
    parser.add_argument('--scale_max', type=float, default=2.0, help='maximum random scale')
    parser.add_argument('--rotate_min', type=float, default=-10, help='minimum random rotate')
    parser.add_argument('--rotate_max', type=float, default=10, help='maximum random rotate')
    parser.add_argument('--zoom_factor', type=int, default=1, help='zoom factor in final prediction map')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label in ground truth')
    parser.add_argument('--aux_weight', type=float, default=0.4, help='loss weight for aux branch')

    parser.add_argument('--gpu', type=int, default=[0, 1, 2, 3], nargs='+', help='used gpu')
    parser.add_argument('--workers', type=int, default=4, help='data loader workers')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--bn_group', type=int, default=16, help='group number for sync bn')
    parser.add_argument('--batch_size_val', type=int, default=1, help='batch size for validation during training, memory and speed tradeoff')
    parser.add_argument('--base_lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--power', type=float, default=0.9, help='power in poly learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: 10)')
    parser.add_argument('--save_step', type=int, default=10, help='model save step (default: 10)')
    parser.add_argument('--save_path', type=str, default='tmp', help='model and summary save path')
    parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (default: none)')
    parser.add_argument('--weight', type=str, default='', help='path to weight (default: none)')
    parser.add_argument('--evaluate', type=int, default=0, help='evaluate on validation set, extra gpu memory needed and small batch_size_val is recommend')
    parser.add_argument('--port', default='23456', type=str)



    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--voc12_root", default="../VOC2012", type=str)
    return parser






# logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def load_state(path, model, optimizer=None):
    def map_func(storage, location):
        return storage.cuda()
    if os.path.isfile(path):
        logger.info("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(path))

class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img
    
    
def main():
    global args, logger, writer
    args = get_parser().parse_args()
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    rank, world_size = dist_init(args.port)
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    #os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
    #if len(args.gpu) == 1:
    #   args.syncbn = False
    if rank == 0:
        logger.info(args)

    if args.bn_group == 1:
        args.bn_group_comm = None
    else:
        assert world_size % args.bn_group == 0
        args.bn_group_comm = simple_group_split(world_size, rank, world_size // args.bn_group)

    if rank == 0:
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))

    from pspnet import PSPNet
    model = PSPNet(backbone=args.backbone, layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, syncbn=args.syncbn, group_size=args.bn_group, group=args.bn_group_comm).cuda()
    logger.info(model)
    # optimizer = torch.optim.SGD(model.parameters(), args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # newly introduced layer with lr x10
    optimizer = torch.optim.SGD(
        [{'params': model.layer0.parameters()},
         {'params': model.layer1.parameters()},
         {'params': model.layer2.parameters()},
         {'params': model.layer3.parameters()},
         {'params': model.layer4.parameters()},
         {'params': model.ppm.parameters(), 'lr': args.base_lr * 10},
         {'params': model.cls1.parameters(), 'lr': args.base_lr * 10},
         {'params': model.cls3.parameters(), 'lr': args.base_lr * 10},
         {'params': model.cls6.parameters(), 'lr': args.base_lr * 10},
         {'params': model.cls9.parameters(), 'lr': args.base_lr * 10}
        ],
        lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #model = torch.nn.DataParallel(model).cuda()
    model = DistModule(model)
    #if args.syncbn:
    #    from lib.syncbn import patch_replication_callback
    #    patch_replication_callback(model)
    cudnn.enabled = True
    cudnn.benchmark = True

    
    
    
    
    
    if args.weight:
        def map_func(storage, location):
            return storage.cuda()
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location=map_func)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        load_state(args.resume, model, optimizer)

        
    normalize = Normalize()
    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                        imutils.RandomResizeLong(256, 512),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray,
                        normalize,
                        imutils.RandomCrop(args.crop_size),
                        imutils.HWC_to_CHW,
                        torch.from_numpy
                    ]))
    
        
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

        
    
    


    for epoch in range(args.start_epoch, args.epochs + 1):
        loss_train = train(train_data_loader, model, optimizer, epoch, args.batch_size)
        if rank == 0:
            writer.add_scalar('loss_train', loss_train, epoch)


        if epoch % args.save_step == 0 and rank == 0:
            filename = args.save_path + '/train_epoch_' + str(epoch) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)



def train(train_loader, model, optimizer, epoch, batch_size):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    loss_meter = AverageMeter()

    model.train()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    end = time.time()
    for i, (_, input, target) in enumerate(train_loader):
        # to avoid bn problem in ppm module with bin size 1x1, sometimes n may get 1 on one gpu during the last batch, so just discard
        # if input.shape[0] < batch_size:
        #     continue
        data_time.update(time.time() - end)
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = args.epochs * len(train_loader)
        poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power, index_split=4)

        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        output1, output3, output6, output9 = model(input_var)
        output1 = output1.squeeze(3).squeeze(2)
        output3 = output3.squeeze(3).squeeze(2)
        output6 = output6.squeeze(3).squeeze(2)
        output9 = output9.squeeze(3).squeeze(2)

        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)
        
        main_loss =(F.multilabel_soft_margin_loss(output1, target_var) +  F.multilabel_soft_margin_loss(output3, target_var)  + F.multilabel_soft_margin_loss(output6, target_var) + F.multilabel_soft_margin_loss(output9, target_var)) / world_size

        loss = main_loss 
        optimizer.zero_grad()
        loss.backward()
        average_gradients(model)
        optimizer.step()



        reduced_loss = loss.data.clone()
        reduced_main_loss = main_loss.data.clone()
        dist.all_reduce(reduced_loss)
        dist.all_reduce(reduced_main_loss)

        main_loss_meter.update(reduced_main_loss[0], input.size(0))
        loss_meter.update(reduced_loss[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if rank == 0:
            if (i + 1) % args.print_freq == 0:
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'MainLoss {main_loss_meter.val:.4f} '
                            'Loss {loss_meter.val:.4f} '.format(epoch, args.epochs, i + 1, len(train_loader),
                                                              batch_time=batch_time,
                                                              data_time=data_time,
                                                              remain_time=remain_time,
                                                              main_loss_meter=main_loss_meter,
                                                              loss_meter=loss_meter))


    if rank == 0:
        logger.info('Train result at epoch [{}/{}]'.format(epoch, args.epochs))
    return main_loss_meter.avg



if __name__ == '__main__':
    main()


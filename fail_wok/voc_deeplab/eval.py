import os
import cv2
import time
import logging
from argparse import ArgumentParser
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F

from utils import AverageMeter, intersectionAndUnion, check_makedirs, colorize
cv2.ocl.setUseOpenCL(False)

import voc12.data
from tool import pyutils, imutils, torchutils
from torchvision import transforms
from torch.utils.data import DataLoader

# Setup
def get_parser():
    parser = ArgumentParser(description='PyTorch Semantic Segmentation Evaluation')
    parser.add_argument('--data_root', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012', help='data root')
    parser.add_argument('--val_list1', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012/list/val.txt', help='val list')
    parser.add_argument('--split', type=str, default='val', help='split in [train, val and test]')
    parser.add_argument('--backbone', type=str, default='resnet', help='backbone network type')
    parser.add_argument('--net_type', type=int, default=0, help='0-single branch, 1-div4 branch')
    parser.add_argument('--layers', type=int, default=50, help='layers number of based resnet')
    parser.add_argument('--classes', type=int, default=21, help='number of classes')
    parser.add_argument('--base_size1', type=int, default=512, help='based size for scaling')
    parser.add_argument('--crop_h', type=int, default=473, help='validation crop size h')
    parser.add_argument('--crop_w', type=int, default=473, help='validation crop size w')
    parser.add_argument('--zoom_factor', type=int, default=1, help='zoom factor in final prediction map')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label in ground truth')
    parser.add_argument('--scales', type=float, default=[1.0], nargs='+', help='evaluation scales')
    parser.add_argument('--has_prediction', type=int, default=0, help='has prediction already or not')

    parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='used gpu')
    parser.add_argument('--workers', type=int, default=1, help='data loader workers')
    parser.add_argument('--model_path', type=str, default='exp/voc2012/psp50/model/train_epoch_100.pth', help='evaluation model path')
    parser.add_argument('--save_folder', type=str, default='exp/voc2012/psp50/result/epoch_100/val/ss', help='results save folder')
    parser.add_argument('--colors_path', type=str, default='data/voc2012/voc2012colors.txt', help='path of dataset colors')
    parser.add_argument('--names_path', type=str, default='data/voc2012/voc2012names.txt', help='path of dataset category names')
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--crop_size", default=441, type=int)
    parser.add_argument("--voc12_root", default="../old_psa/VOC2012", type=str)
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
    global args, logger
    args = get_parser().parse_args()
    logger = get_logger()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
    logger.info(args)
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.crop_h - 1) % 8 == 0 and (args.crop_w - 1) % 8 == 0
    assert args.split in ['train', 'val', 'test']
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    normalize = Normalize()
    infer_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                                   transform=transforms.Compose(
                                                       [np.asarray,
                                                        imutils.RandomCrop(441),
                                                  #      normalize,
                                                        imutils.HWC_to_CHW]))

    val_loader1 = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    


    from pspnet import PSPNet
    model = PSPNet(backbone = args.backbone, layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, use_softmax=False, use_aux=False, pretrained=False, syncbn=False).cuda()

    logger.info(model)
  #  model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    cudnn.enabled = True
    cudnn.benchmark = True
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
     #   model.load_state_dict(checkpoint['state_dict'], strict=False)
     #   logger.info("=> loaded checkpoint '{}'".format(args.model_path))


        pretrained_dict = {k.replace('module.',''): v for k, v in checkpoint['state_dict'].items()}

        dict1 = model.state_dict()
        model.load_state_dict(pretrained_dict, strict=False)

    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    cv2.setNumThreads(0)
    validate(val_loader1, model, args.classes, mean, std, args.base_size1)




def validate(val_loader, model, classes, mean, std, base_size):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (_, input, att) in enumerate(val_loader):
        if i < 10:
            continue
        data_time.update(time.time() - end)
        input = np.squeeze(input.numpy(), axis=0).astype(np.float32)
        image = np.transpose(input, (1, 2, 0))
        cv2.imwrite('image.png', image)
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        
        input = torch.from_numpy(image.transpose((2, 0, 1))).float()   #共享存储空间????
        if std is None:
            for t, m in zip(input, mean):
                t.sub_(m)
        else:
            for t, m, s in zip(input, mean, std):
                t.sub_(m).div_(s)
        input = input.contiguous()
        input = input.unsqueeze(0).cuda(async=True)
        input_var = torch.autograd.Variable(input)
        #result, output = model(input_var)
        result= model.forward_cam(input_var)
        print(result)
        np.save('result', result.squeeze(0).data.cpu().numpy())
        #print(result)
        h, w, _ = image.shape
    
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(val_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))

        break
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')




if __name__ == '__main__':
    main()

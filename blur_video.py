from __future__ import print_function 
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import WIDERFace_ROOT , WIDERFace_CLASSES as labelmap
from PIL import Image
from data import WIDERFaceDetection, WIDERFaceAnnotationTransform, WIDERFace_CLASSES, WIDERFace_ROOT, BaseTransform , TestBaseTransform
from data import *
import torch.utils.data as data
from light_face_ssd import build_ssd
#from resnet50_ssd import build_sfd
import pdb
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time
plt.switch_backend('agg')

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('file', type=str,
                    help="Video file path")
parser.add_argument('out', type=str,
                    help='Output video path')
parser.add_argument('--vertical', type=int, default=0,
                    help='0 : horizontal video(default), 1 : vertical video')
parser.add_argument('--verbose', type=int, default=0,
                    help='Show current progress and remaining time')
parser.add_argument('--reduce_scale', type=float, default=2,
                    help='Reduce scale ratio. ex) 2 = half size of the input. Default : 2')

parser.add_argument('--trained_model', default='weights/light_DSFD.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--threshold', default=0.25, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--widerface_root', default=WIDERFace_ROOT, help='Location of VOC root directory')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def infer(net , img , transform , thresh , cuda , shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)

    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0) , volatile=True)
    if cuda:
        x = x.cuda()
    y = net(x)      # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([ img.shape[1]/shrink, img.shape[0]/shrink,
                         img.shape[1]/shrink, img.shape[0]/shrink] )
    det = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            #label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3]) 
            det.append([pt[0], pt[1], pt[2], pt[3], score])
            j += 1
    if (len(det)) == 0:
        det = [ [0.1,0.1,0.2,0.2,0.01] ]
    det = np.array(det)

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det


def blur_faces(im,  dets , thresh=0.5, blur_level=100):
    """Blur detected bounding boxes."""
    class_name = 'face'
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        top = max(int(bbox[1]), 0)
        left = max(int(bbox[0]), 0)
        bottom = min(int(bbox[3]), im.shape[0]-1)
        right = min(int(bbox[2]), im.shape[1]-1)

        im[top:bottom, left:right] = cv2.blur(im[top:bottom, left:right], (blur_level, blur_level))


if __name__ == '__main__':

    cap = cv2.VideoCapture(args.file)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if args.vertical == 1:
        out_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // args.reduce_scale),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // args.reduce_scale))
    else:
        out_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // args.reduce_scale),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // args.reduce_scale))

    out = cv2.VideoWriter(args.out,
                          fourcc,
                          cap.get(cv2.CAP_PROP_FPS),
                          out_size
                          )
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_count = 0
    e_time = None
    s_time = None

    # load net
    cfg = widerface_640
    num_classes = len(WIDERFace_CLASSES) + 1  # +1 background
    net = build_ssd('test', cfg['min_dim'], num_classes)  # initialize SSD
    # net = nn.DataParallel(net)
    net.load_state_dict(torch.load(args.trained_model))
    net.cuda()
    net.eval()
    print('Finished loading model!')

    # evaluation
    cuda = args.cuda
    transform = TestBaseTransform((104, 117, 123))
    thresh = cfg['conf_thresh']

    shrink = 1

    while cap.isOpened():
        if args.verbose > 0 and e_time is not None:
            ittime = (e_time - s_time) * (total_frame - f_count)
            hour = int(ittime / 60.0 / 60.0)
            minute = int((ittime / 60.0) - (hour * 60))
            second = int(ittime % 60.0)

            print("Progress %d/%d(%.2f%%), Estimated time : %02d:%02d:%02d" %
                  (f_count, total_frame, (f_count / total_frame) * 100, hour, minute, second))

        s_time = time.time()

        ret, frame = cap.read()

        if frame is None:
            break

        if args.vertical == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if args.reduce_scale == 1:
            resized_img = frame
        else:
            resized_img = cv2.resize(frame, None, None, fx=args.reduce_scale, fy=args.reduce_scale)

        det = infer(net, resized_img, transform, thresh, cuda, shrink)
        blur_faces(resized_img, det, args.threshold)

        out.write(resized_img)

        e_time = time.time()
        f_count += 1

    print("Finished!")
    cap.release()
    out.release()

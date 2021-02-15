from utils.utils import *
import numpy as np
import cv2
import torch
import random
import pdb
def letterbox(img, new_shape=416, color=(128, 128, 128), mode='auto', interp=cv2.INTER_AREA):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        r = float(new_shape) / max(shape)  # ratio  = new / old
    else:
        r = max(new_shape) / max(shape)
    ratio = r, r  # width, height ratios
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode is 'scaleFill':
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratio = new_shape / shape[1], new_shape / shape[0]  # width, height ratios

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, dw, dh



def class_convertor(value, focus_cls_dic, real_cls_dic):
     for k, v in focus_cls_dic.items():
         if v== int (value):
             return  real_cls_dic['merged_class_index'][k]


def load_mosaic(self, index):
     # loads images in a mosaic

     labels4 = []
     s = self.img_size
     xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
     img4 = np.zeros((s * 2, s * 2, 3), dtype=np.uint8) + 128  # base image with 4 tiles
     indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
     for i, index in enumerate(indices):
         # Load image
         img = load_image( self, index)
         h, w, _ = img.shape

         # place img in img4
         if i == 0:  # top left
             x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
             x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
         elif i == 1:  # top right
             x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
             x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
         elif i == 2:  # bottom left
             x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
             x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
         elif i == 3:  # bottom right
             x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
             x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

         img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
         padw = x1a - x1b
         padh = y1a - y1b

         # Load labels
         assert (self.lbl_files[index].split('/')[-1].split('.')[0]) ==(self.img_files[index].split('/')[-1].split('.')[0]),\
             print ('label file does not belong to the image')

         x = self.labels[index]
         labels = []
         # Normalized xywh to pixel xyxy format
         labels = x.copy()

         labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
         labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
         labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
         labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh

         labels4.append(labels)

     if len(labels4):
         labels4 = np.concatenate(labels4, 0)

     # Center crop
     a = s // 2
     img4 = img4[a:a + s, a:a + s]
     if len(labels4):
         labels4[:, 1:] -= a


     return img4, labels4

def load_image(self, index):

     img_path = self.img_files[index]
     img = cv2.imread(img_path)  # BGR
     assert img is not None, 'Image Not Found ' + img_path
     r = self.img_size / max(img.shape)  # size ratio
     if self.augment_flg:  # if training (NOT testing), downsize to inference shape
         h, w = img.shape[:2]
         img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)  # _LINEAR fastest
     return img



def pars_infoFile(info_file):


     file = open(info_file,'r')
     lines = file.read().split('\n')
     dic_cls = {}
     last_key=''
     for l in lines:
         if l.startswith('*') or not l.strip():
             continue
         elif l.startswith('#'):

             if l.split()[1] == 'included_class':
                 last_key = 'included_class'
                 dic_cls['included_class'] = {}
             elif l.split()[1] =='excluded_class':
                 dic_cls['excluded_class'] = {}
                 last_key = 'excluded_class'
             elif l.split()[1] == 'merged_class_index':
                 last_key = 'merged_class_index'
                 dic_cls['merged_class_index'] = {}

         else:
             if last_key:
                 key, value = l.split(':')
                 dic_cls[last_key][key.strip()] = int (value)
     return dic_cls


import math
def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:  # targets = [cls, xyxy]
        targets = []
    border = 0  # width of added border (optional)
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    changed = (border != 0) or (M != np.eye(3)).any()
    if changed:
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_AREA, borderValue=(128, 128, 128))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets



def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    x = (np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1).astype(np.float32)  # random gains
    img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * x.reshape((1, 1, 3))).clip(None, 255).astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
import matplotlib
matplotlib.use('Agg')
import glob
import os
import random
import shutil
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision
from . import torch_utils  # , google_utils

matplotlib.rc('font', **{'size': 11})

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def floatn(x, n=3):  # format floats to n decimals
    return float(format(x, '.%gf' % n))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    ni = len(labels)  # number of images
    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    weights = np.hstack([gpi * ni - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco_class_weights():  # frequency of each class in coco train2014
    n = [187437, 4955, 30920, 6033, 3838, 4332, 3160, 7051, 7677, 9167, 1316, 1372, 833, 6757, 7355, 3302, 3776, 4671,
         6769, 5706, 3908, 903, 3686, 3596, 6200, 7920, 8779, 4505, 4272, 1862, 4698, 1962, 4403, 6659, 2402, 2689,
         4012, 4175, 3411, 17048, 5637, 14553, 3923, 5539, 4289, 10084, 7018, 4314, 3099, 4638, 4939, 5543, 2038, 4004,
         5053, 4578, 27292, 4113, 5931, 2905, 11174, 2873, 4036, 3415, 1517, 4122, 1980, 4464, 1190, 2302, 156, 3933,
         1877, 17630, 4337, 4624, 1075, 3468, 135, 1380]
    weights = 1 / torch.Tensor(n)
    weights /= weights.sum()
    # with open('data/coco.names', 'r') as f:
    #     for k, v in zip(f.read().splitlines(), n):
    #         print('%20s: %g' % (k, v))
    return weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    if y.ndim >1:
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2
        y[:, 2] = x[:, 2] - x[:, 0]
        y[:, 3] = x[:, 3] - x[:, 1]
    elif y.ndim==1:
        y[0] = (x[0] + x[2]) / 2
        y[1] = (x[1] + x[3]) / 2
        y[2] = x[2] - x[0]
        y[3] = x[3] - x[1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
    coords[:, [0, 2]] -= (img1_shape[1] - img0_shape[1] * gain) / 2  # x padding
    coords[:, [1, 3]] -= (img1_shape[0] - img0_shape[0] * gain) / 2  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=img_shape[1])  # clip x
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=img_shape[0])  # clip y


def clip_coords_np(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(min=0, max=img_shape[1])  # clip x
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(min=0, max=img_shape[0])  # clip y

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r.append(recall[-1])

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p.append(precision[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall, precision))

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            # ax.plot(np.concatenate(([0.], recall)), np.concatenate(([0.], precision)))
            # ax.set_xlabel('YOLOv3-SPP')
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    iou = inter_area / union_area  # iou
    if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
        c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
        c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + 1e-16  # convex area
        return iou - (c_area - union_area) / c_area  # GIoU

    return iou


def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn() https://arxiv.org/pdf/1708.02002.pdf
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=2.5)
    def __init__(self, loss_fcn, gamma=0.5, alpha=1, reduction='mean'):
        super(FocalLoss, self).__init__()
        loss_fcn.reduction = 'none'  # required to apply FL to each element
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        loss = self.loss_fcn(input, target)
        loss *= self.alpha * (1.000001 - torch.exp(-loss)) ** self.gamma  # non-zero power for gradient stability

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def generating_pseudo_b1(roi, GT, imgs, A_CNN,  psd_thrshold= None, overlap_gt=None, device=None, alpha=None, kmeans=None, beta=None, img_path=None, GT_upi=None):
    tpr_upi, fpr_upi = 0, 0
    all_retrieved = 0
    all_upi = 0

    #  roi contains boxes ifo in pixel-coordinate and their corresponding object confidence, class confidence and the class identifier
    conf = torch.nn.Softmax(dim=1)
    pic_file = 'alpha_%g_%g_voc_exp1'%(alpha,beta)
    if not os.path.isdir(pic_file):
        os.mkdir(pic_file)
    # psd_lbl = []
    psd_lbl = [[] for _ in range(len(imgs))]
    psd_roi = [[] for _ in range(len(imgs))]


    for ind_btch, r in enumerate( roi):
        ROI, LBL = [],[]

        if r is None:
            continue
        img = imgs[ind_btch]
        ROI,LBL =  [],[]

        yolo_box = [[], []]
        img_sz = img.shape[1]
        H, W = img.shape[1], img.shape[2]
        d = 1 / img_sz

        # convert boxes from x_c,y_c, w, h to x,y,x,y
        gt_img = xywh2xyxy(GT[GT[:,0]==ind_btch,-4:])

        if GT_upi is not None and len(GT_upi)>0:
            gt_upi_img =xywh2xyxy(GT_upi[GT_upi[:,0]==ind_btch,-4:])

        clip_coords(r, (H, W))
        confirm_bbox = []

        for  (*pbox, pconf, pcls_conf, pcls) in (r):

            iou = bbox_iou(torch.stack(pbox)*d, gt_img)

            if (iou > overlap_gt).any():
                # means estimated box has GT.
                continue

            # if pcls_conf*pconf> psd_thrshold:

            yolo_box[0].append(pbox)
            yolo_box[1].append(pcls_conf*pconf)

        img = img.detach().clone()

        # pdb.set_trace()
        if len(yolo_box[0])>0:

            for i, (yolo_t, yolo_pcls_conf) in enumerate(zip(yolo_box[0], yolo_box[1])):

                x0_hat = np.max([int(yolo_t[0].item()), 0])
                x1_hat = np.max([int(yolo_t[2].item()), 0])
                y0_hat = np.max([int(yolo_t[1].item()), 0])
                y1_hat = np.max([int(yolo_t[3].item()), 0])
                # print (x0_hat,x1_hat,y0_hat,y1_hat, W, H)
                assert (x0_hat <= W) and (x1_hat<= W) and y0_hat<=H and y1_hat<=H

                if x1_hat - x0_hat >= 5 and y1_hat - y0_hat >= 5:
                    img_bbx = img[:, y0_hat:y1_hat, x0_hat:x1_hat]
                    if img_bbx.shape[0]==3:
                        img_bbx = img_bbx.unsqueeze(0)
                    img_bbx_btch = transformation(img_bbx)
                    img_bbx_btch = pad_image_crop(img_bbx_btch,kmeans)


                    img_bbx_btch = img_bbx_btch.to(device)

                    # output = torch.cat([A_CNN(the_img.unsqueeze(0)) for the_img in img_bbx_btch])
                    # output = torch.mean(output, dim=0).reshape(1,-1)

                    # batch of transformed images
                    output = A_CNN(img_bbx_btch)
                    output = torch.mean(output, dim=0).reshape(1,-1)

    #                 if torch.argmax(conf(output)).item() != output.shape[1] and torch.max(
    #                         conf(output[:, :-1])).item() >= psd_thrshold and yolo_pcls_conf.item() >= psd_thrshold:
    #                     pred = output[0, :-1].argmax().item()  # get the index of the max log-probability
    #
    #                     lbl_dic = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5,
    #                                'train': 6, 'cat': 7, 'dog': 8, 'horse': 9
    #                         , 'sheep': 10, 'cow': 11}
    #                     lbl_dict = dict(zip(lbl_dic.values(), lbl_dic.keys()))
    #                     cls_conf = alpha * torch.max(conf(output[:, :-1])).item()
    #
    #                     plt.title('%s_%d' % (lbl_dict[pred], int(cls_conf * 100)))
    #                     plt.savefig(pic_file+'/img%d%d' % (ind_btch, i) + '.png')
    #
    #                     w = d * (x1_hat - x0_hat)
    #                     h = d * (y1_hat - y0_hat)
    #                     x_c, y_c = d * ((x1_hat + x0_hat) / 2), d * ((y1_hat + y0_hat) / 2)
    #                     assert (w >= 0 or w <= 1) and (h >= 0 or h <= 1) and (y_c >= 0 or x_c <= 1) and (
    #                     y_c >= 0 or y_c <= 1)
    #                     psd_lbl.append([ind_btch, pred, x_c, y_c, w, h, cls_conf])
    #                     temp = conf(output)[0, :-1].data.cpu().numpy()
    #                     temp /= np.sum(temp)
    #                     psd_lbl[-1].extend(temp)
    #
    # if len(psd_lbl) > 0:
    #     # pdb.set_trace()
    #     psd_lbl = torch.from_numpy(np.vstack(psd_lbl)).type(torch.FloatTensor)
    #     return psd_lbl
    # else:
    #     return None

                    # non-dustbin objects that having p_{yolo}(O) > threshold
                    if torch.argmax(conf(output)).item()!=output.shape[1] and \
                                    torch.max(conf(output[:,:-1])).item() >=psd_thrshold and yolo_pcls_conf.item()>= psd_thrshold :

                        all_retrieved += 1
                        if GT_upi is not None and len(GT_upi) > 0:
                            all_upi += len(gt_upi_img)
                            iou = bbox_iou(torch.stack(yolo_t) * d, gt_upi_img)
                            if (iou > overlap_gt).any():
                                tpr_upi += 1
                            else:
                                fpr_upi += 1

                        pred = output[0,:-1].argmax().item()  # get the index of the max log-probability

                        lbl_dic = {'person':0, 'bicycle':1, 'car':2, 'motorcycle':3, 'airplane':4, 'bus':5, 'train':6, 'cat':7, 'dog':8, 'horse':9
                                      , 'sheep':10, 'cow':11}
                        lbl_dict = dict(zip(lbl_dic.values(), lbl_dic.keys()))
                        cls_conf = alpha*torch.max(conf(output[:, :-1])).item()

                        plt.figure()
                        plt.imshow(img_bbx_btch[0].cpu().numpy().transpose(1, 2, 0))
                        plt.axis('off')
                        plt.title('%s_%d' % (lbl_dict[pred],int(cls_conf*100 )))
                        plt.savefig(pic_file + '/img%d%d' % (ind_btch, i) + '.png')
                        plt.close()
                        confirm_bbox.append([x0_hat, y0_hat, x1_hat, y1_hat,  lbl_dict[pred],int(cls_conf*100 ) ])

                        w = d*(x1_hat-x0_hat)
                        h = d*(y1_hat - y0_hat)
                        x_c, y_c = d*( (x1_hat+x0_hat)/2) , d*((y1_hat+y0_hat)/2)
                        assert (w >= 0 or w <= 1) and (h >= 0 or h <= 1) and (y_c >= 0 or x_c <= 1) and (y_c >= 0 or y_c <= 1)

                        # psd_lbl[ind_btch].append([ind_btch, pred, x_c, y_c, w, h, cls_conf*yolo_pcls_conf])

                        # (xc, yc, w, h, object_conf, class_conf, class )
                        ROI.append(torch.tensor([x_c, y_c, w, h, yolo_pcls_conf*cls_conf, cls_conf ,pred ]))
                        LBL.append((conf(output)[0,:-1]/conf(output)[0,:-1].sum()).data.cpu())
        if len(ROI)>0 :
            psd_lbl[ind_btch] = torch.stack(LBL, 0 )
            psd_roi[ind_btch] = torch.stack(ROI,0)
        if len(confirm_bbox)>0:

            plt.figure(figsize=(10,10))
            plt.subplot(1, 1, 1)
            plt.imshow(img.data.cpu().numpy().transpose(1, 2, 0))
            target = GT[GT[:, 0] == ind_btch]
            gt_target = xywh2xyxy(target[:, 2:6])
            gt_target = gt_target.data.cpu().numpy()
            gt_target[:,[0,2]] *= W
            gt_target[:,[1,3]] *= H
            plt.plot([gt_target[:, 0], gt_target[:, 2], gt_target[:, 2], gt_target[:, 0], gt_target[:, 0]],
                     [gt_target[:, 1], gt_target[:, 1], gt_target[:, 3], gt_target[:, 3], gt_target[:, 1]], '-',
                     color='green', linewidth=3)
            # plt.title('GT')

            plt.axis('off')

            confirm_bbox = np.vstack(confirm_bbox)
            the_box = (confirm_bbox[:,:4]).astype('int')

            # plt.subplot(1,2,2)
            # plt.imshow(img.data.cpu().numpy().transpose(1,2,0))
            plt.plot([ the_box[:, 0], the_box[:, 2], the_box[:, 2], the_box[:, 0], the_box[:, 0] ],[ the_box[:, 1], the_box[:, 1], the_box[:, 3], the_box[:, 3], the_box[:, 1] ], '-', color='darkviolet',linewidth=3)
            [plt.text(the_box[i, 0], the_box[i, 1], confirm_bbox[i, 4], color='darkviolet', fontsize=18) for i in range(len(confirm_bbox))]


            plt.axis('off')
            plt.savefig(pic_file+'/'+'Vis_'+img_path[ind_btch].split('/')[-1])
            plt.close()


    flag = any([len(psd_lbl[i])>0 for i in range(len(roi))])
    if flag:
        # (x1, y1, x2, y2, object_conf, class_conf, class )
        final_psd_lbl = merge_pesudoLabels(psd_roi, conf_thres=0.1, nms_thres=0.5, psd_target=psd_lbl)

        return final_psd_lbl, (all_retrieved, all_upi, tpr_upi, fpr_upi)
    else:
        return None



def convert_for_build_target(new_roi):
    psd_lbl=[None]*len(new_roi)
    for ind_btch, roi in enumerate(new_roi):
#         (x1, y1, x2, y2, object_conf, class_conf, class ) ==> [ind_btch, pred, x_c, y_c, w, h, cls_conf*yolo_pcls_conf]+conf(output)[0,:-1].data.cpu().numpy())

        psd_lbl.append([ind_btch, roi[6], xyxy2xywh(roi[:,:4], roi[4])])


def pad_image_crop(img_bbx_btch, kmeans):
    WH = [[the_img.shape[1],the_img.shape[0]]for the_img in img_bbx_btch]
    CluCent_HW = kmeans.cluster_centers_
    group = kmeans.predict(WH)

    pad_img_bbx_btch = [torch.nn.functional.pad(img, pad=(
    int((CluCent_HW[group[i]][0] - img.shape[2]) // 2), int((CluCent_HW[group[i]][0] - img.shape[2]) // 2 + (CluCent_HW[group[i]][0] - img.shape[2]) % 2)
    , int((CluCent_HW[group[i]][1] - img.shape[1]) / 2), int((CluCent_HW[group[i]][1] - img.shape[1]) // 2 + (CluCent_HW[group[i]][1] - img.shape[1]) % 2)), mode="constant", value=0)
                        for i, img in enumerate(img_bbx_btch)]
    return torch.stack(pad_img_bbx_btch,0)



def transformation(bbx_img):
    m = 2
    bbx_img = bbx_img[0]
    batch_img = torch.zeros(m+2,bbx_img.shape[0], bbx_img.shape[1], bbx_img.shape[2])
    # original image
    batch_img[0,...] = bbx_img[0]

    # 3*3 drop pathc
    I = np.linspace(0,3,num=4)
    J= np.linspace(0,3,num=4)
    ii, jj = np.meshgrid(I,J)
    II ,JJ = np.random.choice(ii.ravel().astype('int'), m), np.random.choice( jj.ravel().astype('int'), m)
    for m, (i,j) in enumerate(zip(II,JJ)):
        img = bbx_img.clone()
        h, w = int(img.shape[1] / 4), int(img.shape[2] / 4)
        x = j * w
        y = i * h
        img[:,y:y + h, x:x + w] = torch.zeros((3,h, w))

        batch_img[m, ...] = img

    #  vertical flip
    cur_img = torchvision.transforms.functional.to_pil_image(bbx_img.cpu())
    cur_img = torchvision.transforms.functional.hflip(cur_img)
    batch_img[-1,...] = torchvision.transforms.functional.to_tensor(cur_img)

    return batch_img






def modified_compute_loss_upi_lpi(imgs, p, targets_lpi, targets_upi,\
                                  roi, model, A_CNN, alpha = 0.5, beta=0.5, psd_threshod=0.5, device=None, epoch=None, epoch_threshold=10, total_epoch=100, kmeans=None
                          ,img_path= None):  # predictions, targets, model


    h = model.hyp  # hyperparameters
    arc = model.arc  # # (default, uCE, uBCE) detection architectures

    # Define criteria
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor

    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]))
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()  # weight=model.class_weights

    if 'F' in arc:  # add focal loss
        g = h['fl_gamma']
        BCEcls, BCEobj, BCE, CE = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCE, g), FocalLoss(CE, g)

    psd_target = None
    if epoch>epoch_threshold:


        new_roi =  (non_max_suppression( prediction= roi, conf_thres=0.1, nms_thres=0.5))

        psd_target, metrics_values = generating_pseudo_b1(new_roi, targets_lpi,  imgs, A_CNN,  psd_thrshold=0.8, overlap_gt=0.3,
                                          device=device, alpha=alpha, kmeans=kmeans,beta=beta, img_path = img_path, GT_upi=targets_upi)


        if psd_target is not None:
            tcls_psd, tbox_psd, indices_psd, av_psd, clsconf_psd, A_CNN_output = build_targets(model, psd_target.type(torch.FloatTensor).to(device))


    # Compute losses
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchor_vec = build_targets(model, targets_lpi)
    # import pdb;pdb.set_trace()
    GIOU = []
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj

        # Compute losses
        nb = len(b)
        if nb:  # number of targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            tobj[b, a, gj, gi] = 1.0  # obj
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)

            # GIoU
            pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pbox = torch.cat((pxy, torch.exp(ps[:, 2:4]) * anchor_vec[i]), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou computation
            GIOU.append(giou.data)
            lbox += (1.0 - giou).mean()  # giou loss

            if 'default' in arc and model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.zeros_like(ps[:, 5:])  # targets
                t[range(nb), tcls[i]] = 1.0
                lcls += BCEcls(ps[:, 5:], t)  # BCE


        if psd_target is not None and epoch>epoch_threshold:


            b_psd, a_psd, gj_psd, gi_psd = indices_psd[i]  # image, anchor, gridy, gridx
            # tconf = torch.zeros_like(p_l[..., 0])  # conf
            # Compute losses over pseudo labeled

            if len(b_psd):  # number of targets

                ps_psd = pi[b_psd, a_psd, gj_psd, gi_psd]  # Yolo-predictions closest to anchors
                tobj[b_psd, a_psd, gj_psd, gi_psd] = clsconf_psd[i] # conf

                if 'default' in arc and model.nc > 1:  # cls loss (only if multiple classes)
                    xx = torch.index_select(A_CNN_output[i], 1, torch.cuda.LongTensor([7,11,8,9,6,10,2,3,1,4,5,0]))
                    true_psd_prob = alpha * (beta * torch.nn.functional.softmax(ps_psd[..., 5:], dim=1) + (1 - beta) * (xx))
                    # true_psd_prob = torch.zeros_like(ps_psd[:,5:])
                    # true_psd_prob[range(len(b_psd)), xx.argmax(1)] = 1
                    # lcls += BCEcls(ps_psd[:, 5:], true_psd_prob)  # BCE

                    # A note :: input  contains log-probabilities. The targets as probabilities (i.e. without taking the logarithm).
                    psd_lcls = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(ps_psd[..., 5:], dim=1), true_psd_prob)

                    # psd_lcls =  -torch.sum(
                    #     true_psd_prob * torch.nn.functional.log_softmax(ps_psd[..., 5:], dim=1), dim=1).mean()
                    print('class loss for pseduo labels ',psd_lcls.item())
                    lcls += psd_lcls

        if 'default' in arc:  # separate obj and cls
            lobj += BCEobj(pi[..., 4], tobj)  # obj loss

        # elif 'BCE' in arc:  # unified BCE (80 classes)
        #     t = torch.zeros_like(pi[..., 5:])  # targets
        #     if nb:
        #         t[b, a, gj, gi, tcls[i]] = 1.0
        #     lobj += BCE(pi[..., 5:], t)
        #
        # elif 'CE' in arc:  # unified CE (1 background + 80 classes)
        #     t = torch.zeros_like(pi[..., 0], dtype=torch.long)  # targets
        #     if nb:
        #         t[b, a, gj, gi] = tcls[i] + 1
        #     lcls += CE(pi[..., 4:].view(-1, model.nc + 1), t.view(-1))


    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    loss = lbox + lobj + lcls
    if not torch.isfinite(lbox):
        import pdb;pdb.set_trace()
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()



def modified_compute_loss(imgs, p, targets, roi, model, A_CNN, alpha = 0.5, beta=0.5, psd_threshod=0.5, device=None, epoch=None, epoch_threshold=10, total_epoch=100, kmeans=None
                          ,img_path= None):  # predictions, targets, model


    h = model.hyp  # hyperparameters
    arc = model.arc  # # (default, uCE, uBCE) detection architectures

    # Define criteria
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor

    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]))
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()  # weight=model.class_weights

    if 'F' in arc:  # add focal loss
        g = h['fl_gamma']
        BCEcls, BCEobj, BCE, CE = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCE, g), FocalLoss(CE, g)

    psd_target = None
    if epoch>epoch_threshold:


        new_roi =  (non_max_suppression( prediction= roi, conf_thres=0.1, nms_thres=0.5))

        psd_target = generating_pseudo_b1(new_roi, targets, imgs, A_CNN,  psd_thrshold=0.8, overlap_gt=0.3, device=device, alpha=alpha, kmeans=kmeans,beta=beta, img_path = img_path)
        if psd_target is not None:
            tcls_psd, tbox_psd, indices_psd, av_psd, clsconf_psd, A_CNN_output = build_targets(model, psd_target.type(torch.FloatTensor).to(device))


    # Compute losses
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchor_vec = build_targets(model, targets)
    # import pdb;pdb.set_trace()
    GIOU = []
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj

        # Compute losses
        nb = len(b)
        if nb:  # number of targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            tobj[b, a, gj, gi] = 1.0  # obj
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)

            # GIoU
            pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pbox = torch.cat((pxy, torch.exp(ps[:, 2:4]) * anchor_vec[i]), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou computation
            GIOU.append(giou.data)
            lbox += (1.0 - giou).mean()  # giou loss

            if 'default' in arc and model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.zeros_like(ps[:, 5:])  # targets
                t[range(nb), tcls[i]] = 1.0
                lcls += BCEcls(ps[:, 5:], t)  # BCE


        if psd_target is not None and epoch>epoch_threshold:


            b_psd, a_psd, gj_psd, gi_psd = indices_psd[i]  # image, anchor, gridy, gridx
            # tconf = torch.zeros_like(p_l[..., 0])  # conf
            # Compute losses over pseudo labeled

            if len(b_psd):  # number of targets

                ps_psd = pi[b_psd, a_psd, gj_psd, gi_psd]  # Yolo-predictions closest to anchors
                tobj[b_psd, a_psd, gj_psd, gi_psd] = clsconf_psd[i] # conf

                if 'default' in arc and model.nc > 1:  # cls loss (only if multiple classes)
                    xx = torch.index_select(A_CNN_output[i], 1, torch.cuda.LongTensor([7,11,8,9,6,10,2,3,1,4,5,0]))
                    true_psd_prob = alpha * (beta * torch.nn.functional.softmax(ps_psd[..., 5:], dim=1) + (1 - beta) * (xx))
                    # true_psd_prob = torch.zeros_like(ps_psd[:,5:])
                    # true_psd_prob[range(len(b_psd)), xx.argmax(1)] = 1
                    # lcls += BCEcls(ps_psd[:, 5:], true_psd_prob)  # BCE

                    # A note :: input  contains log-probabilities. The targets as probabilities (i.e. without taking the logarithm).
                    psd_lcls = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(ps_psd[..., 5:], dim=1), true_psd_prob)

                    # psd_lcls =  -torch.sum(
                    #     true_psd_prob * torch.nn.functional.log_softmax(ps_psd[..., 5:], dim=1), dim=1).mean()
                    print('class loss for pseduo labels ',psd_lcls.item())
                    lcls += psd_lcls

        if 'default' in arc:  # separate obj and cls
            lobj += BCEobj(pi[..., 4], tobj)  # obj loss

        # elif 'BCE' in arc:  # unified BCE (80 classes)
        #     t = torch.zeros_like(pi[..., 5:])  # targets
        #     if nb:
        #         t[b, a, gj, gi, tcls[i]] = 1.0
        #     lobj += BCE(pi[..., 5:], t)
        #
        # elif 'CE' in arc:  # unified CE (1 background + 80 classes)
        #     t = torch.zeros_like(pi[..., 0], dtype=torch.long)  # targets
        #     if nb:
        #         t[b, a, gj, gi] = tcls[i] + 1
        #     lcls += CE(pi[..., 4:].view(-1, model.nc + 1), t.view(-1))


    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    loss = lbox + lobj + lcls
    if not torch.isfinite(lbox):
        import pdb;pdb.set_trace()
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()



def modified_compute_loss_teacher(imgs, p, targets, roi_teacher, p_teacher, model, A_CNN, alpha = 0.5, beta=0.5, psd_threshod=0.5, device=None, epoch=None, epoch_threshold=10, total_epoch=100, kmeans=None):  # predictions, targets, model


    h = model.hyp  # hyperparameters
    arc = model.arc  # # (default, uCE, uBCE) detection architectures

    # Define criteria
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor

    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]))
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()  # weight=model.class_weights

    if 'F' in arc:  # add focal loss
        g = h['fl_gamma']
        BCEcls, BCEobj, BCE, CE = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCE, g), FocalLoss(CE, g)

    psd_target = None
    if epoch>epoch_threshold:


        new_roi =  (non_max_suppression( prediction= roi_teacher, conf_thres=0.1, nms_thres=0.5))

        psd_target = generating_pseudo_b1(new_roi, targets, imgs, A_CNN,  psd_thrshold=0.8, overlap_theta=0.5, device=device, alpha=alpha, kmeans=kmeans, beta= beta)
        if psd_target is not None:
            tcls_psd, tbox_psd, indices_psd, anchor_psd, clsconf_psd, A_CNN_output = build_targets(model, psd_target.type(torch.FloatTensor).to(device))


    # Compute losses
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchor_vec = build_targets(model, targets)
    # import pdb;pdb.set_trace()
    for i, (pi, p_T) in enumerate(zip(p,p_teacher)):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj

        # Compute losses
        nb = len(b)
        if nb:  # number of targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            tobj[b, a, gj, gi] = 1.0  # obj
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)

            # GIoU
            pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pbox = torch.cat((pxy, torch.exp(ps[:, 2:4]) * anchor_vec[i]), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou computation
            lbox += (1.0 - giou).mean()  # giou loss

            if 'default' in arc and model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.zeros_like(ps[:, 5:])  # targets
                t[range(nb), tcls[i]] = 1.0
                lcls += BCEcls(ps[:, 5:], t)  # BCE


        if psd_target is not None and epoch>epoch_threshold:


            b_psd, a_psd, gj_psd, gi_psd = indices_psd[i]  # image, anchor, gridy, gridx
            # tconf = torch.zeros_like(p_l[..., 0])  # conf
            # Compute losses over pseudo labeled

            if len(b_psd):  # number of targets

                ps_psd = pi[b_psd, a_psd, gj_psd, gi_psd]  # Teacher-predictions closest to anchors
                tobj[b_psd, a_psd, gj_psd, gi_psd] = clsconf_psd[i] # conf
                # GIoU

                pxy_psd = torch.sigmoid(ps_psd[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
                pbox_psd = torch.cat((pxy_psd, torch.exp(ps_psd[:, 2:4]) * anchor_psd[i]), 1)  # predicted box
                giou_psd = bbox_iou(pbox_psd.t(), tbox_psd[i], x1y1x2y2=False, GIoU=True)  # giou computation
                lbox += (1.0 - giou_psd).mean()  # giou loss

                if 'default' in arc and model.nc > 1:  # cls loss (only if multiple classes)
                    xx = torch.index_select(A_CNN_output[i], 1,
                                            torch.cuda.LongTensor([7, 11, 8, 9, 6, 10, 2, 3, 1, 4, 5, 0]))
                    true_psd_prob = alpha * (beta * torch.nn.functional.softmax(ps_psd[..., 5:], dim=1) + (1 - beta) * (xx))
                    # A note :: input  contains log-probabilities. The targets as probabilities (i.e. without taking the logarithm).
                    psd_lcls = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(ps_psd[..., 5:], dim=1), true_psd_prob, reduction='batchmean')

                    # psd_lcls = BCEcls(ps_psd[:, 5:], true_psd_prob)  # BCE

                    # psd_lcls =  -torch.sum(
                    #     true_psd_prob * torch.nn.functional.log_softmax(ps_psd[..., 5:], dim=1), dim=1).mean()
                    print('class loss for pseduo labels ',psd_lcls.item())
                    lcls += psd_lcls

        if 'default' in arc:  # separate obj and cls
            lobj += BCEobj(pi[..., 4], tobj)  # obj loss

        # elif 'BCE' in arc:  # unified BCE (80 classes)
        #     t = torch.zeros_like(pi[..., 5:])  # targets
        #     if nb:
        #         t[b, a, gj, gi, tcls[i]] = 1.0
        #     lobj += BCE(pi[..., 5:], t)
        #
        # elif 'CE' in arc:  # unified CE (1 background + 80 classes)
        #     t = torch.zeros_like(pi[..., 0], dtype=torch.long)  # targets
        #     if nb:
        #         t[b, a, gj, gi] = tcls[i] + 1
        #     lcls += CE(pi[..., 4:].view(-1, model.nc + 1), t.view(-1))


    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    loss = lbox + lobj + lcls
    if not torch.isfinite(lbox):
        import pdb;pdb.set_trace()
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()




def build_targets(model, targets):
    # targets = [image, class, x, y, w, h]
    confidence_pseudo = []
    A_CNN_output = []
    nt = len(targets)
    tcls, tbox, indices, av = [], [], [], []
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for i in model.yolo_layers:
        # get number of grid points and anchor vec for this yolo layer
        if multi_gpu:
            ng, anchor_vec = model.module.module_list[i].ng, model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec

        # iou of targets-anchors
        t, a = targets, []
        gwh = t[:, 4:6] * ng
        if nt:
            iou = torch.stack([wh_iou(x, gwh) for x in anchor_vec], 0)

            use_best_anchor = False
            if use_best_anchor:
                iou, a = iou.max(0)  # best iou and anchor
            else:  # use all anchors
                na = len(anchor_vec)  # number of anchors
                a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)
                t = targets.repeat([na, 1])
                gwh = gwh.repeat([na, 1])
                iou = iou.view(-1)  # use all ious

            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            reject = True
            if reject:
                j = iou > model.hyp['iou_t']  # iou threshold hyperparameter
                t, a, gwh = t[j], a[j], gwh[j]

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4] * ng  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))

        # GIoU
        gxy -= gxy.floor()  # xy
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        av.append(anchor_vec[a])  # anchor vec

        if t.shape[1]>=7:
            confidence_pseudo.append(t[:,6]) # class confidence
            A_CNN_output.append(t[:,7:])

        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            assert c.max() <= model.nc, 'Target classes exceed model classes'
    if  targets.shape[1]>=7:
        return tcls, tbox, indices, av, confidence_pseudo, A_CNN_output
    else:
        return tcls, tbox, indices, av




def generating_pseudo_teacher(teacher_roi, GT, imgs,  psd_thrshold= None, overlap_theta=None):

    #  roi contains boxes ifo in pixel-coordinate and their corresponding object confidence, class confidence and the class identifier

    psd_lbl = []
    pic_file = 'Teacher_crop'
    if not  os.path.isdir(pic_file):
        os.mkdir(pic_file)
    for ind_btch, r in enumerate( teacher_roi):
        if r is None:
            continue
        img = imgs[ind_btch]

        img_sz = img.shape[1]
        H, W = img.shape[1], img.shape[2]
        d = 1 / img_sz
        # convert boxes from x_c,y_c, w, h to x,y,x,y
        gt_img = xywh2xyxy(GT[GT[:,0]==ind_btch,-4:])
        clip_coords(r, (H, W))
        img = img.detach().clone()
        for  i, (*pbox, pconf, pcls_conf, pcls) in enumerate(r):
            iou = bbox_iou(torch.stack(pbox)*d, gt_img)

            if (iou >= overlap_theta).any() or (pcls_conf*pconf).item() < psd_thrshold:
                # means estimated box has GT.
                continue

            x0_hat = np.max([int(pbox[0].item()), 0])
            x1_hat = np.max([int(pbox[2].item()), 0])
            y0_hat = np.max([int(pbox[1].item()), 0])
            y1_hat = np.max([int(pbox[3].item()), 0])
            # print (x0_hat,x1_hat,y0_hat,y1_hat, W, H)
            assert (x0_hat <= W) and (x1_hat <= W) and y0_hat <= H and y1_hat <= H
            # import pdb;pdb.set_trace()
            if x1_hat - x0_hat >= 5 and y1_hat - y0_hat >= 5:
                img_bbx = img[:, y0_hat:y1_hat, x0_hat:x1_hat]
                if img_bbx.shape[0] == 3:
                    img_bbx = img_bbx.unsqueeze(0)

                plt.close()
                plt.figure()
                plt.imshow(img_bbx[0].cpu().numpy().transpose(1, 2, 0))
                plt.axis('off')
                lbl =  ['cat', 'cow', "dog", "horse", 'train', "sheep"]+[ "car", "motorbike", "bicycle",'aeroplane', 'bus', 'person']
                plt.title(lbl[int(pcls.item())]+'_%g'%(pcls_conf*pconf).item())
                plt.savefig(pic_file+'/img%g%g'%(ind_btch, i ))

            # pbox scale back to [0,1] then convert to [x,y,w,h]
            box = xyxy2xywh((torch.stack(pbox)*d).view(-1,4))

            psd_lbl.append([ind_btch,pcls.item()] )
            psd_lbl[-1].extend(box.view(-1).data.cpu().numpy())
    #
    if len(psd_lbl) > 0:

        psd_lbl = torch.from_numpy(np.vstack(psd_lbl)).type(torch.FloatTensor)
        return psd_lbl
    else:
        return None



def modified_compute_loss_teacher_only(imgs, p, targets, roi_teacher, p_teacher, model, device=None):  # predictions, targets, model


    h = model.hyp  # hyperparameters
    arc = model.arc  # # (default, uCE, uBCE) detection architectures

    # Define criteria
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor

    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]))
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()  # weight=model.class_weights

    if 'F' in arc:  # add focal loss
        g = h['fl_gamma']
        BCEcls, BCEobj, BCE, CE = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCE, g), FocalLoss(CE, g)

    psd_target = None

    new_roi =  (non_max_suppression( prediction= roi_teacher, conf_thres=0.1, nms_thres=0.5))

    psd_target = generating_pseudo_teacher(new_roi, targets, imgs,  psd_thrshold=0.8, overlap_theta=0.5)
    if psd_target is not None:
        tcls_psd, tbox_psd, indices_psd, anchor_psd = build_targets(model, psd_target.type(torch.FloatTensor).to(device))


    # Compute losses
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchor_vec = build_targets(model, targets)
    # import pdb;pdb.set_trace()
    for i, (pi, p_T) in enumerate(zip(p,p_teacher)):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj

        # Compute losses
        nb = len(b)
        if nb:  # number of targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            tobj[b, a, gj, gi] = 1.0  # obj
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)

            # GIoU
            pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pbox = torch.cat((pxy, torch.exp(ps[:, 2:4]) * anchor_vec[i]), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou computation
            lbox += (1.0 - giou).mean()  # giou loss

            if 'default' in arc and model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.zeros_like(ps[:, 5:])  # targets
                t[range(nb), tcls[i]] = 1.0
                lcls += BCEcls(ps[:, 5:], t)  # BCE


        if psd_target is not None :


            b_psd, a_psd, gj_psd, gi_psd = indices_psd[i]  # image, anchor, gridy, gridx
            # tconf = torch.zeros_like(p_l[..., 0])  # conf
            # Compute losses over pseudo labeled

            if len(b_psd):  # number of targets

                ps_psd = pi[b_psd, a_psd, gj_psd, gi_psd]  # Teacher-predictions closest to anchors
                tobj[b_psd, a_psd, gj_psd, gi_psd] = p_T[b_psd, a_psd, gj_psd, gi_psd, 4] # object conf
                # GIoU
                pxy_psd = torch.sigmoid(ps_psd[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
                pbox_psd = torch.cat((pxy_psd, torch.exp(ps_psd[:, 2:4]) * anchor_psd[i]), 1)  # predicted box
                giou_psd = bbox_iou(pbox_psd.t(), tbox_psd[i], x1y1x2y2=False, GIoU=True)  # giou computation
                lbox += (1.0 - giou_psd).mean()  # giou loss

                if 'default' in arc and model.nc > 1:  # cls loss (only if multiple classes)

                    true_psd_prob =   torch.nn.functional.sigmoid(p_T[b_psd, a_psd, gj_psd, gi_psd, 5:])
                    psd_lcls = BCEcls(ps_psd[:, 5:], true_psd_prob)  # BCE
                    print('class loss for pseduo labels %g '%psd_lcls.item(), ' box loss %g'%((1.0 - giou_psd).mean().item()))
                    lcls += psd_lcls

        if 'default' in arc:  # separate obj and cls
            lobj += BCEobj(pi[..., 4], tobj)  # obj loss



    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    loss = lbox + lobj + lcls
    if not torch.isfinite(lbox):
        import pdb;pdb.set_trace()
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()





def compute_loss(p, targets, model,device):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor

    lcls, lbox, lobj = ft([0]).to(device), ft([0]).to(device), ft([0]).to(device)
    tcls, tbox, indices, anchor_vec = build_targets(model, targets)
    h = model.hyp  # hyperparameters
    arc = model.arc  # # (default, uCE, uBCE) detection architectures

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']])).to(device)
    BCE = nn.BCEWithLogitsLoss().to(device)
    CE = nn.CrossEntropyLoss()  # weight=model.class_weights

    if 'F' in arc:  # add focal loss
        g = h['fl_gamma']
        BCEcls, BCEobj, BCE, CE = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCE, g), FocalLoss(CE, g)

    # Compute losses
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx

        tobj = torch.zeros_like(pi[..., 0]).to(device)  # target obj

        # Compute losses
        nb = len(b)
        if nb:  # number of targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            tobj[b, a, gj, gi] = 1.0  # obj
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)

            # GIoU
            pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pbox = torch.cat((pxy, torch.exp(ps[:, 2:4]) * anchor_vec[i]), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou computation

            lbox += (1.0 - giou).mean()  # giou loss

            if 'default' in arc and model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.zeros_like(ps[:, 5:]).to(device)  # targets

                t[range(nb), tcls[i]] = 1.0
                lcls += BCEcls(ps[:, 5:], t)  # BCE
                # lcls += CE(ps[:, 5:], tcls[i])  # CE

                # Instance-class weighting (use with reduction='none')
                # nt = t.sum(0) + 1  # number of targets per class
                # lcls += (BCEcls(ps[:, 5:], t) / nt).mean() * nt.mean()  # v1
                # lcls += (BCEcls(ps[:, 5:], t) / nt[tcls[i]].view(-1,1)).mean() * nt.mean()  # v2

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        if 'default' in arc:  # separate obj and cls
            lobj += BCEobj(pi[..., 4], tobj)  # obj loss

        elif 'BCE' in arc:  # unified BCE (80 classes)
            t = torch.zeros_like(pi[..., 5:]).to(device)  # targets
            if nb:
                t[b, a, gj, gi, tcls[i]] = 1.0
            lobj += BCE(pi[..., 5:], t)

        elif 'CE' in arc:  # unified CE (1 background + 80 classes)
            t = torch.zeros_like(pi[..., 0], dtype=torch.long).to(device)  # targets
            if nb:
                t[b, a, gj, gi] = tcls[i] + 1
            lcls += CE(pi[..., 4:].view(-1, model.nc + 1), t.view(-1))

    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    loss = lbox + lobj + lcls
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()



def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    """

    min_wh = 2  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        # Multiply conf by class conf to get combined confidence
        class_conf, class_pred = pred[:, 5:].max(1)
        pred[:, 4] *= class_conf

        # # Merge classes (optional)
        # class_pred[(class_pred.view(-1,1) == torch.LongTensor([2, 3, 5, 6, 7]).view(1,-1)).any(1)] = 2
        #
        # # Remove classes (optional)
        # pred[class_pred != 2, 4] = 0.0

        # Select only suitable predictions
        i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & torch.isfinite(pred).all(1)
        pred = pred[i]

        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)

        # Get detections sorted by decreasing confidence scores
        pred = pred[(-pred[:, 4]).argsort()]

        det_max = []
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in pred[:, -1].unique():
            dc = pred[pred[:, -1] == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 100:
                dc = dc[:100]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117

            # Non-maximum suppression
            if nms_style == 'OR':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            elif nms_style == 'SOFT':  # soft-NMS https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:]
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
                    # dc = dc[dc[:, 4] > nms_thres]  # new line per https://github.com/ultralytics/yolov3/issues/362

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output
import pdb

def merge_pesudoLabels(prediction, conf_thres=0.5, nms_thres=0.1, psd_target=None):

    output = []

    for image_i, (pred, A_CNN_out) in enumerate(zip(prediction,psd_target)):
        if len(pred) == 0 or len(A_CNN_out)==0:
            continue


        assert(len(A_CNN_out)== len(pred));'length YOLO prediction %d should equal to length A_CNN prediction%d'%(len(A_CNN_out), len(pred))

        class_conf, class_pred = pred[:, 5], pred[:,-1]
        # object conf multiply by class conf
        pred[:, 4] *= class_conf
        # Select only suitable predictions
        i = (pred[:, 4] > conf_thres)  & torch.isfinite(pred).all(1)
        if not any(i):
            continue

        pred = pred[i]
        A_CNN_out = A_CNN_out[i]

        # If none are remaining => process next image


        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)

        # Get detections sorted by decreasing confidence scores
        pred = pred[(-pred[:, 4]).argsort()]
        A_CNN_out = A_CNN_out[(-pred[:, 4]).argsort()]

        det_max = []
        acnn_max = []
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in pred[:, -1].unique():

            dc = pred[pred[:, -1] == c]  # select class c
            acnn_out = A_CNN_out[pred[:, -1] == c]
            assert len(dc)== len(acnn_out)
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                acnn_max.append(acnn_out)

            elif n > 100:
                dc = dc[:100]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117


            if nms_style == 'MERGE' and n>1:  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        acnn_max.append(acnn_out)
                        break
                    i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    acnn_out[0, ...] = (weights * acnn_out[i]).sum(0) / weights.sum()
                    acnn_max.append(acnn_out[:1])
                    det_max.append(dc[:1])
                    dc = dc[i == 0]
                    acnn_out = acnn_out[i == 0]

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate

            coordinate = xyxy2xywh(det_max[:, :4])

            #       shape det_max  (x1, y1, x2, y2, object_conf, class_conf, class) ==> [ind_btch, pred, x_c, y_c, w, h, cls_conf*yolo_pcls_conf]+conf(output)[0,:-1].data.cpu().numpy())

            for i in range(len(det_max)):

                output.append(np.hstack([image_i,c.numpy(),
                                         coordinate[i].reshape(-1).numpy().reshape(-1),
                                         det_max[i,4].numpy(),
                                         acnn_max[i].numpy().reshape(-1)]))



    if len(output)>0:
        output = np.vstack(output)
        output = torch.from_numpy(output).type(torch.FloatTensor)

        return output
    else:return None


def get_yolo_layers(model):
    bool_vec = [x['type'] == 'yolo' for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]  # [82, 94, 106] for yolov3


def print_model_biases(model):
    # prints the bias neurons preceding each yolo layer
    print('\nModel Bias Summary (per output layer):')
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for l in model.yolo_layers:  # print pretrained biases
        if multi_gpu:
            na = model.module.module_list[l].na  # number of anchors
            b = model.module.module_list[l - 1][0].bias.view(na, -1)  # bias 3x85
        else:
            na = model.module_list[l].na
            b = model.module_list[l - 1][0].bias.view(na, -1)  # bias 3x85
        print('regression: %5.2f+/-%-5.2f ' % (b[:, :4].mean(), b[:, :4].std()),
              'objectness: %5.2f+/-%-5.2f ' % (b[:, 4].mean(), b[:, 4].std()),
              'classification: %5.2f+/-%-5.2f' % (b[:, 5:].mean(), b[:, 5:].std()))


def strip_optimizer(f='weights/last.pt'):  # from utils.utils import *; strip_optimizer()
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
    x = torch.load(f)
    x['optimizer'] = None
    torch.save(x, f)


def create_backbone(f='weights/last.pt'):  # from utils.utils import *; create_backbone()
    # create a backbone from a *.pt file
    x = torch.load(f)
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    for p in x['model'].values():
        try:
            p.requires_grad = True
        except:
            pass
    torch.save(x, 'weights/backbone.pt')


def coco_class_count(path='../coco/labels/train2014/'):
    # Histogram of occurrences per class
    nc = 80  # number classes
    x = np.zeros(nc, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nc)
        print(i, len(files))


def coco_only_people(path='../coco/labels/val2014/'):
    # Find images with only people
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def select_best_evolve(path='evolve*.txt'):  # from utils.utils import *; select_best_evolve()
    # Find best evolved mutation
    for file in sorted(glob.glob(path)):
        x = np.loadtxt(file, dtype=np.float32, ndmin=2)
        print(file, x[fitness(x).argmax()])


def crop_images_random(path='../images/', scale=0.50):  # from utils.utils import *; crop_images_random()
    # crops images into random squares up to scale fraction
    # WARNING: overwrites images!
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        img = cv2.imread(file)  # BGR
        if img is not None:
            h, w = img.shape[:2]

            # create random mask
            a = 30  # minimum size (pixels)
            mask_h = random.randint(a, int(max(a, h * scale)))  # mask height
            mask_w = mask_h  # mask width

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            cv2.imwrite(file, img[ymin:ymax, xmin:xmax])


def coco_single_class_labels(path='../coco/labels/train2014/', label_class=43):
    # Makes single-class coco datasets. from utils.utils import *; coco_single_class_labels()
    if os.path.exists('new/'):
        shutil.rmtree('new/')  # delete output folder
    os.makedirs('new/')  # make new output folder
    os.makedirs('new/labels/')
    os.makedirs('new/images/')
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        with open(file, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
        i = labels[:, 0] == label_class
        if any(i):
            img_file = file.replace('labels', 'images').replace('txt', 'jpg')
            labels[:, 0] = 0  # reset class to 0
            with open('new/images.txt', 'a') as f:  # add image to dataset list
                f.write(img_file + '\n')
            with open('new/labels/' + Path(file).name, 'a') as f:  # write label
                for l in labels[i]:
                    f.write('%g %.6f %.6f %.6f %.6f\n' % tuple(l))
            shutil.copyfile(src=img_file, dst='new/images/' + Path(file).name.replace('txt', 'jpg'))  # copy images


def kmeans_targets(path='../coco/trainvalno5k.txt', n=9, img_size=512):  # from utils.utils import *; kmeans_targets()
    # Produces a list of target kmeans suitable for use in *.cfg files
    from utils.datasets import LoadImagesAndLabels
    from scipy import cluster

    # Get label wh
    dataset = LoadImagesAndLabels(path, augment=True, rect=True, cache_labels=True)
    for s, l in zip(dataset.shapes, dataset.labels):
        l[:, [1, 3]] *= s[0]  # normalized to pixels
        l[:, [2, 4]] *= s[1]
        l[:, 1:] *= img_size / max(s) * random.uniform(0.99, 1.01)  # nominal img_size for training
    wh = np.concatenate(dataset.labels, 0)[:, 3:5]  # wh from cxywh

    # Kmeans calculation
    k = cluster.vq.kmeans(wh, n)[0]
    k = k[np.argsort(k.prod(1))]  # sort small to large

    # Measure IoUs
    iou = torch.stack([wh_iou(torch.Tensor(wh).T, torch.Tensor(x).T) for x in k], 0)
    biou = iou.max(0)[0]  # closest anchor IoU
    print('Best possible recall: %.3f' % (biou > 0.2635).float().mean())  # BPR (best possible recall)

    # Print
    print('kmeans anchors (n=%g, img_size=%g, IoU=%.2f/%.2f/%.2f-min/mean/best): ' %
          (n, img_size, biou.min(), iou.mean(), biou.mean()), end='')
    for i, x in enumerate(k):
        print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg

    # Plot
    # plt.hist(biou.numpy().ravel(), 100)


def print_mutation(hyp, results, bucket=''):
    # Print mutation results to evolve.txt (for use with main_Object_detector_train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        os.system('gsutil cp gs://%s/evolve.txt .' % bucket)  # download evolve.txt

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    np.savetxt('evolve.txt', x[np.argsort(-fitness(x))], '%10.3g')  # save sort by fitness

    if bucket:
        os.system('gsutil cp evolve.txt gs://%s' % bucket)  # upload evolve.txt


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs

    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0.shape)

            # Classes
            pred_cls1 = d[:, 6].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    return x[:, 2] * 0.8 + x[:, 3] * 0.2  # weighted mAP and F1 combination


# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_wh_methods():  # from utils.utils import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x, ya, '.-', label='yolo method')
    plt.plot(x, yb ** 2, '.-', label='^2 power method')
    plt.plot(x, yb ** 2.5, '.-', label='^2.5 power method')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.legend()
    fig.tight_layout()
    fig.savefig('comparison.png', dpi=200)


def plot_images(imgs, targets, paths=None, fname='images.jpg'):
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()
    # targets = targets[targets[:, 1] == 21]  # plot only one class

    fig = plt.figure(figsize=(10, 10))
    bs, _, h, w = imgs.shape  # batch size, _, height, width
    bs = min(bs, 16)  # limit plot to 16 images
    ns = np.ceil(bs ** 0.5)  # number of subplots

    for i in range(bs):
        boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T
        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
        plt.subplot(ns, ns, i + 1).imshow(imgs[i].transpose(1, 2, 0))
        plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '-')
        plt.axis('off')
        if paths is not None:
            s = Path(paths[i]).name
            plt.title(s[:min(len(s), 40)], fontdict={'size': 8})  # limit to 40 characters
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close()



def plot_test_txt():  # from utils.utils import *; plot_test()
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig('hist2d.jpg', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    fig.tight_layout()
    plt.savefig('hist1d.jpg', dpi=200)


def plot_targets_txt():  # from utils.utils import *; plot_targets_txt()
    # Plot test.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32)
    x = x.T

    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    fig.tight_layout()
    plt.savefig('targets.jpg', dpi=200)


def plot_evolution_results(hyp):  # from utils.utils import *; plot_evolution_results(hyp)
    # Plot hyperparameter evolution results in evolve.txt
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    weights = (f - f.min()) ** 2  # for weighted results
    fig = plt.figure(figsize=(12, 10))
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 5]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(4, 5, i + 1)
        plt.plot(mu, f.max(), 'o', markersize=10)
        plt.plot(y, f, '.')
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        print('%15s: %.3g' % (k, mu))
    fig.tight_layout()
    plt.savefig('evolve.png', dpi=200)


def plot_results(start=0, stop=0):  # from utils.utils import *; plot_results()
    # Plot training results files 'results*.txt'
    fig, ax = plt.subplots(2, 5, figsize=(14, 7))
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'F1']
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        for i in range(10):
            y = results[i, x]
            if i in [0, 1, 2, 5, 6, 7]:
                y[y == 0] = np.nan  # dont show zero loss values
            ax[i].plot(x, y, marker='.', label=f.replace('.txt', ''))
            ax[i].set_title(s[i])
            if i in [5, 6, 7]:  # share train and val loss y axes
                ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])

    fig.tight_layout()
    ax[1].legend()
    fig.savefig('results.png', dpi=200)


def plot_results_overlay(start=0, stop=0):  # from utils.utils import *; plot_results_overlay()
    # Plot training results files 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val', 'val', 'Recall', 'F1']  # legends
    t = ['GIoU', 'Objectness', 'Classification', 'P-R', 'mAP-F1']  # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5))
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                if i in [0, 1, 2]:
                    y[y == 0] = np.nan  # dont show zero loss values
                ax[i].plot(x, y, marker='.', label=s[j])
            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.tight_layout()
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def version_to_tuple(version):
    # Used to compare versions of library
    return tuple(map(int, (version.split("."))))





def read_data_stp(path_merged_data_stp):
    data_stp_dic = []  # list of dictionaries, each for a dataset
    f = open(path_merged_data_stp, 'r')
    lines = f.read().split('\n')
    for l in lines:

        if l.strip():
            if l.startswith('#'):
                data_stp_dic.append({})
                data_stp_dic[-1]['dataset_name'] = l.split('#')[-1].strip()

            else:
                k, v = l.split('=')
                data_stp_dic[-1][k.strip()] = str(v.strip())

    return data_stp_dic




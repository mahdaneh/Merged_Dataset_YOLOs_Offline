import os
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import cv2
import torch
from PIL import Image, ExifTags
import glob
import pdb
import random
from utils.utils import *
import utils.utils_dataset as ult_dst
""" in this dataset calss we load the fully-labelled dataset but keep only included classes (create missing label instance)
or the excluded classes (as ground truth for the missing label instances)"""

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

class incld_or_excld_dataset(Dataset):
     def __init__(self, info_file, flag_incld = True, path=None,
                  img_size=416, augment=False, rect=True, batch_size=None, hyp=None):
         self.rect = rect
         self.augment_flg = augment
         self.batch_size  =batch_size
         self.hyp = hyp
         self.dic_cls = ult_dst.pars_infoFile(info_file)

         self.class_names = [*self.dic_cls['merged_class_index'].keys()]

         if flag_incld: self.fcs_lbl = self.dic_cls['included_class']
         else: self.fcs_lbl = self.dic_cls['excluded_class']



         img_files = glob.glob(path+'/*.png')+glob.glob(path+'/*.jpg')+glob.glob(path+'/*.jpeg')+glob.glob(path+'/*.bmp')


         n = len(img_files)
         assert n > 0, 'No images found in %s' % path

         self.img_size = img_size
         self.labels = []
         self.lbl_files =[]
         self.img_files = []

         # ----------------- load label folder -----------------------------

         label_files = [x.replace('JPEGImages', 'labels_original').replace('JPEGImages','labelled_GT_BT').replace(os.path.splitext(x)[-1], '.txt')
                             for x in img_files]# all label files

         pbar = tqdm(label_files, desc='Reading labels')
         nm, nf, ne, ns =  0, 0, 0, 0  # number missing, number found, number empty, number datasubset
         for i, file in enumerate(pbar):

             try:
                 with open(file, 'r') as f:
                     l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
             except:
                 nm += 1  # print('missing labels for image %s' % self.img_files[i])  # file missing
                 continue

             if l.shape[0]:
                 assert l.shape[1] == 5, '> 5 label columns: %s' % file
                 assert (l >= 0).all(), 'negative labels: %s' % file
                 assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                 # only consider the labels indicated in self.fcs_lbl.values(), the function class_index_merged transform the index in the origianl dataset to a proper index
                 # related to the problem in hand


                 lbl = np.asarray(
                     [[ult_dst.class_convertor(value=x[0], focus_cls_dic= self.fcs_lbl, real_cls_dic=self.dic_cls )] + x[1:].tolist() \
                      for x in l if x[0] in self.fcs_lbl.values()])

                 if len(lbl)>0:
                    self.labels.append(lbl)
                    self.lbl_files.append(label_files[i])
                    self.img_files.append(img_files[i])
             pbar.desc = 'length of images %d and labels %d'%(len(self.img_files), len (self.labels))

         if self.rect:
             n = len(self.img_files)
             bi = np.floor(np.arange(n) / self.batch_size).astype(np.int)  # batch index
             nb = bi[-1] + 1  # number of batches
             self.batch = bi
             # Read image shapes

             s = [exif_size(Image.open(f)) for f in tqdm(self.img_files, desc='Reading image shapes')]
             # Sort by aspect ratio
             s = np.array(s, dtype=np.float64)
             ar = s[:, 1] / s[:, 0]  # aspect ratio
             i = ar.argsort()
             self.img_files = [self.img_files[i] for i in i]
             self.lbl_files = [self.lbl_files[i] for i in i]
             self.shapes = s[i]
             ar = ar[i]

             # Set training image shapes
             shapes = [[1, 1]] * nb
             for i in range(nb):
                 ari = ar[bi == i]
                 mini, maxi = ari.min(), ari.max()
                 if maxi < 1:
                     shapes[i] = [maxi, 1]
                 elif mini > 1:
                     shapes[i] = [1, 1 / mini]

             self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32.).astype(np.int) * 32
         self.__len__()



     def __len__(self):
         print('datapath %s Size of dataset %d' % (os.path.split(self.img_files[0])[0], len(self.img_files)))
         return len(self.img_files)



     def __getitem__(self, index):

         this_img_file = self.img_files [index]
         this_lbl_file  = self.lbl_files[index]
         this_lbl = self.labels[index]
         # only if the images contain a label in focus (fcs_lbl)
         if len(this_lbl):
             mosaic = True and self.augment_flg
             if mosaic:
                 # Load mosaic
                 img, labels = ult_dst.load_mosaic(self, index)
                 h, w = img.shape[:2]

             else: # inference time
                 # Load image
                 img = ult_dst.load_image(self, index)

                 # Letterbox
                 h, w = img.shape[:2]

                 if self.rect:
                     img, ratio, padw, padh = ult_dst.letterbox(img, self.batch_shapes[self.batch[index]], mode='rect')
                 else:
                     img, ratio, padw, padh = ult_dst.letterbox(img, self.img_size, mode='square')

                 labels = this_lbl.copy()
                 labels[:, 1] = ratio[0] * w * (this_lbl[:, 1] - this_lbl[:, 3] / 2) + padw
                 labels[:, 2] = ratio[1] * h * (this_lbl[:, 2] - this_lbl[:, 4] / 2) + padh
                 labels[:, 3] = ratio[0] * w * (this_lbl[:, 1] + this_lbl[:, 3] / 2) + padw
                 labels[:, 4] = ratio[1] * h * (this_lbl[:, 2] + this_lbl[:, 4] / 2) + padh
             if self.augment_flg:
                 # Augment colorspace
                 ult_dst.augment_hsv(img, hgain=self.hyp['hsv_h'], sgain=self.hyp['hsv_s'], vgain=self.hyp['hsv_v'])

                 # Augment imagespace
                 g = 0.0 if mosaic else 1.0  # do not augment mosaics
                 hyp = self.hyp

                 img, labels = ult_dst.random_affine(img, labels,
                                             degrees=hyp['degrees'] * g,
                                             translate=hyp['translate'] * g,
                                             scale=hyp['scale'] * g,
                                             shear=hyp['shear'] * g)


             # img = cv2.imread(this_img_file)  # BGR
             # assert img is not None, 'Image Not Found %s' % this_lbl_file
             # r = self.img_size / max(img.shape)  # size ratio
             # if r < 1:  # if training (NOT testing), downsize to inference shape
             #     h, w = img.shape[:2]
             #     img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)  # or INTER_AREA
             #
             # h, w = img.shape[:2]
             # # always make a square image
             # img, ratio, padw, padh = letterbox(img, self.img_size, mode='square')
             #
             #
             # # Normalized xywh to pixel xyxy format
             # labels = this_lbl.copy()
             # labels[:, 1] = ratio[0] * w * (this_lbl[:, 1] - this_lbl[:, 3] / 2) + padw
             # labels[:, 2] = ratio[1] * h * (this_lbl[:, 2] - this_lbl[:, 4] / 2) + padh
             # labels[:, 3] = ratio[0] * w * (this_lbl[:, 1] + this_lbl[:, 3] / 2) + padw
             # labels[:, 4] = ratio[1] * h * (this_lbl[:, 2] + this_lbl[:, 4] / 2) + padh

         nL = len(labels)  # number of labels
         if nL:
             # convert xyxy to xywh
             labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

             # Normalize coordinates 0 - 1
             labels[:, [2, 4]] /= img.shape[0]  # height
             labels[:, [1, 3]] /= img.shape[1]  # width

         if self.augment_flg:
             # random left-right flip
             lr_flip = True
             if lr_flip and random.random() < 0.5:
                 img = np.fliplr(img)
                 if nL:
                     labels[:, 1] = 1 - labels[:, 1]

             # random up-down flip
             ud_flip = False
             if ud_flip and random.random() < 0.5:
                 img = np.flipud(img)
                 if nL:
                     labels[:, 2] = 1 - labels[:, 2]

         if (labels<0).any(): print ('negative*****************')
         labels_out = torch.zeros((len(labels), 6))
         labels_out[:, 1:] = torch.from_numpy(labels)


         # Normalize
         img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
         img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
         img /= 255.0  # 0 - 255 to 0.0 - 1.0

         return torch.from_numpy(img), labels_out, this_lbl_file, (h, w)









class incld_and_excld_dataset(Dataset):

    def __init__(self,info_file, path=None, img_size=416):
        dic_cls = ult_dst.pars_infoFile(info_file)
        incld_cls = dic_cls['included_class']
        excld_cls = dic_cls['excluded_class']

        self.num_class = self._num_cls(dic_cls)

        img_files = glob.glob(path + '/*.png') + glob.glob(path + '/*.jpg') + glob.glob(path + '/*.jpeg') + glob.glob(
            path + '/*.bmp')

        n = len(img_files)
        assert n > 0, 'No images found in %s' % path

        self.img_size = img_size
        self.labels_incld = []
        self.img_files = []
        self.labels_excld= []

        # ----------------- load label folder -----------------------------

        self.label_files = [x.replace('JPEGImages', 'labels_original').replace('JPEGImages', 'labelled_GT_BT').replace(
            os.path.splitext(x)[-1], '.txt')
                            for x in img_files]  # all label files
        pbar = tqdm(self.label_files, desc='Reading labels')
        nm, nf, ne, ns = 0, 0, 0, 0  # number missing, number found, number empty, number datasubset
        for i, file in enumerate(pbar):

            try:
                with open(file, 'r') as f:
                    l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            except:
                nm += 1  # print('missing labels for image %s' % self.img_files[i])  # file missing
                continue

            if l.shape[0]:
                assert l.shape[1] == 5, '> 5 label columns: %s' % file
                assert (l >= 0).all(), 'negative labels: %s' % file
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                # only consider the labels indicated in self.fcs_lbl.values(), the function class_index_merged transform the index in the origianl dataset to a proper index
                # related to the problem in hand


                lbl_incld = np.asarray(
                    [[ult_dst.class_convertor(x[0], incld_cls, dic_cls)] + x[1:].tolist() for x in l if x[0] in incld_cls.values()])

                lbl_exlcd = np.asarray(
                    [[ult_dstclass_convertor(x[0], excld_cls, dic_cls)] + x[1:].tolist() for x in l if x[0] in excld_cls.values()])
                if len(lbl_incld) > 0 or len(lbl_exlcd)>0:
                    self.labels_incld.append(lbl_incld)
                    self.labels_excld.append(lbl_exlcd)
                    self.img_files.append(img_files[i])
            pbar.desc = 'length of images %d ; included labels %d ; excluded labels %d' % (len(self.img_files), len(self.labels_incld) , len(self.labels_excld))
        self._check_length()
        self.__len__()


    def _num_cls (self, dic_cls):
        return len(dic_cls['merged_class_index'])

    def _check_length(self):

        assert len(self.img_files) == len(
            self.labels_incld)==len(self.labels_excld), "length of image_filename list, that of label_filename list and that of label list should be equal"

    def __len__(self):
        print('datapath %s Size of dataset %d' % (os.path.split( self.img_files[0])[0], len(self.img_files)))
        return len(self.img_files)

    def __getitem__(self, index):

        this_img_file = self.img_files[index]
        this_lbl_file = self.label_files[index]
        this_lbl_incld = self.labels_incld[index]
        this_lbl_excld = self.labels_excld[index]

        img = cv2.imread(this_img_file)  # BGR
        assert img is not None, 'Image Not Found %s' % this_lbl_file
        r = self.img_size / max(img.shape)  # size ratio
        if r < 1:  # if training (NOT testing), downsize to inference shape
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)  # or INTER_AREA

        h, w = img.shape[:2]
        # always make a square image
        img, ratio, padw, padh = ult_dst.letterbox(img, self.img_size, mode='square')


        # only if the images contain a label in focus (fcs_lbl)
        if len(this_lbl_incld) :
            # Normalized xywh to pixel xyxy format
            labels_incld = this_lbl_incld.copy()
            labels_incld[:, 1] = ratio[0] * w * (this_lbl_incld[:, 1] - this_lbl_incld[:, 3] / 2) + padw
            labels_incld[:, 2] = ratio[1] * h * (this_lbl_incld[:, 2] - this_lbl_incld[:, 4] / 2) + padh
            labels_incld[:, 3] = ratio[0] * w * (this_lbl_incld[:, 1] + this_lbl_incld[:, 3] / 2) + padw
            labels_incld[:, 4] = ratio[1] * h * (this_lbl_incld[:, 2] + this_lbl_incld[:, 4] / 2) + padh

        if len(this_lbl_excld):
            labels_excld = this_lbl_excld.copy()
            labels_excld[:, 1] = ratio[0] * w * (this_lbl_excld[:, 1] - this_lbl_excld[:, 3] / 2) + padw
            labels_excld[:, 2] = ratio[1] * h * (this_lbl_excld[:, 2] - this_lbl_excld[:, 4] / 2) + padh
            labels_excld[:, 3] = ratio[0] * w * (this_lbl_excld[:, 1] + this_lbl_excld[:, 3] / 2) + padw
            labels_excld[:, 4] = ratio[1] * h * (this_lbl_excld[:, 2] + this_lbl_excld[:, 4] / 2) + padh


        if len(this_lbl_incld):
            # convert xyxy to xywh
            labels_incld[:, 1:5] = xyxy2xywh(labels_incld[:, 1:5])

            # Normalize coordinates 0 - 1
            labels_incld[:, [2, 4]] /= img.shape[0]  # height
            labels_incld[:, [1, 3]] /= img.shape[1]  # width

        labels_out_incld = torch.zeros((len(this_lbl_incld), 6))
        if len (this_lbl_incld):labels_out_incld[:, 1:] = torch.from_numpy(labels_incld)


        if  len(this_lbl_excld) :
            # convert xyxy to xywh
            labels_excld[:, 1:5] = xyxy2xywh(labels_excld[:, 1:5])

            # Normalize coordinates 0 - 1
            labels_excld[:, [2, 4]] /= img.shape[0]  # height
            labels_excld[:, [1, 3]] /= img.shape[1]  # width

        labels_out_excld = torch.zeros((len(this_lbl_excld), 6))
        if len(this_lbl_excld) : labels_out_excld[:, 1:] = torch.from_numpy(labels_excld)

        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return torch.from_numpy(img), labels_out_incld, labels_out_excld, this_img_file, (h, w)



# -------------------------- End of Class ---------------------------------------




def collate_fn(batch):
     img, label, path, hw = list(zip(*batch))  # transposed
     for i, l in enumerate(label):
         l[:, 0] = i  # add target image index for build_targets()
     return torch.stack(img, 0), torch.cat(label, 0), path, hw


def collate_fn_excd_and_incd(batch):
    img, label_incld, label_excld, path, hw = list(zip(*batch))  # transposed
    for i, l in enumerate(label_incld):
        l[:, 0] = i  # add target image index for build_targets()

    for i, l in enumerate(label_excld):
        l[:, 0] = i
    return torch.stack(img, 0), torch.cat(label_incld, 0), torch.cat(label_excld, 0), path, hw





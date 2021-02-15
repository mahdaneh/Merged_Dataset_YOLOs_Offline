from tqdm import tqdm
from utils.utils import *
import torch
class metrics ():
    # dataloader is the loader on the concatenated datasets
    def __init__(self, dataloader, model,kmeans, A_CNN,  nms_conf=0.001,nms_thres=0.5, p_n_iou_thrs = None, conf_thrsh_psd=0.8):
        self.kmeans = kmeans
        self.A_CNN = A_CNN
        self.dataloader_UPI = dataloader
        self.model = model
        self.p_n_iou_thrs = p_n_iou_thrs # this is the threshold used for indicating whether a estimated ROI is - or +
        self.nms_conf = nms_conf
        self.nms_thres = nms_thres
        self.total_retrieved = 0
        self.rtrv_negative = 0
        self.rtrv_positive = 0
        self.total_positive = 0
        self.device = next(self.model.parameters()).device
        self.conf_thrsh_psd = conf_thrsh_psd

        self.A_CNN.to(self.device)
        self.A_CNN.eval()


    def TPR_FPR_Recall (self): #

        self.total_retrieved, self.rtrv_positive, self.rtrv_negative, self.total_positive=0, 0, 0, 0

        print(('\n' + '%20s ' * 7) % ('Pstv_Rtrvd', 'Ngtv_Rtrvd', 'Ttl_Rtrvd', 'Ttl_Pstv', 'TPR', 'FPR', 'Recall'))
        pbar = tqdm(enumerate(self.dataloader_UPI))
        conf = torch.nn.Softmax(dim=1)

        for i, (imgs, UPI_targets, LPI_targets,  paths, _) in pbar:
            UPI_targets = UPI_targets.to(self.device)
            LPI_targets = LPI_targets.to(self.device)
            imgs = imgs.to(self.device)
            roi, pred = self.model(imgs)  # inference and training outputs

            # how many of fullly-labelled targets should be dropped and how
            output = non_max_suppression(roi, conf_thres=self.nms_conf, nms_thres=self.nms_thres)
            self._positive_recall(output, imgs, targets_upi=UPI_targets, targets_lpi= LPI_targets, conf_function = conf)


            self.total_positive += len(UPI_targets)

            s = ( '%20d' * 3 + '%20.3g' * 4) % (  self.rtrv_positive, self.rtrv_negative,  self.total_retrieved, self.total_positive,
                                              self.rtrv_positive / self.total_retrieved, self.rtrv_negative/self.total_retrieved, self.rtrv_positive / self.total_positive)
            pbar.desc = s

        return  self.rtrv_positive / self.total_retrieved, self.rtrv_negative/self.total_retrieved, self.rtrv_positive / self.total_positive




    def _positive_recall (self, roi, imgs, targets_upi, targets_lpi=None, conf_function = None ):
        _,_, height, width = imgs.shape



        # length(roi) == batch_size, each element of roi is a list of bounding boxes for an image
        for i, pred in enumerate(roi):
            img = imgs[i]

            yolo_box = [[], []]
            img_sz = img.shape[1]
            H, W = img.shape[1], img.shape[2]

            tbox_upi, tbox_lpi = None, None

            if pred is None: continue

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            if targets_lpi is not None:
                tlabels_lpi = targets_lpi[targets_lpi[:, 0] == i, 1:]
                tbox_lpi = xywh2xyxy(tlabels_lpi[:, 1:5])
                tbox_lpi[:, [0, 2]] *= width
                tbox_lpi[:, [1, 3]] *= height

            tlabels_upi = targets_upi[targets_upi[:, 0] == i, 1:]
            if len(tlabels_upi):
                # target boxes
                tbox_upi = xywh2xyxy(tlabels_upi[:, 1:5])
                # convert from [0,1] to [0, width] and [0, height]
                tbox_upi[:, [0, 2]] *= width
                tbox_upi[:, [1, 3]] *= height


            # Search for the UPIs correctly classified as positive
            for (*pbox, pconf, pcls_conf, pcls) in (pred):

                iou = bbox_iou(torch.stack(pbox), tbox_lpi)

                if (iou > self.p_n_iou_thrs).any():
                    # means estimated box is a GT.
                    continue

                yolo_box[0].append(pbox)
                yolo_box[1].append(pcls_conf * pconf)

            img = img.detach().clone()

            if len(yolo_box[0]) > 0:

                for i, (yolo_t, yolo_pcls_conf) in enumerate(zip(yolo_box[0], yolo_box[1])):

                    x0_hat = np.max([int(yolo_t[0].item()), 0])
                    x1_hat = np.max([int(yolo_t[2].item()), 0])
                    y0_hat = np.max([int(yolo_t[1].item()), 0])
                    y1_hat = np.max([int(yolo_t[3].item()), 0])
                    # print (x0_hat,x1_hat,y0_hat,y1_hat, W, H)
                    assert (x0_hat <= W) and (x1_hat <= W) and y0_hat <= H and y1_hat <= H

                    if x1_hat - x0_hat >= 5 and y1_hat - y0_hat >= 5:
                        img_bbx = img[:, y0_hat:y1_hat, x0_hat:x1_hat]
                        if img_bbx.shape[0] == 3:
                            img_bbx = img_bbx.unsqueeze(0)
                        img_bbx_btch = transformation(img_bbx)
                        img_bbx_btch = pad_image_crop(img_bbx_btch, self.kmeans)

                        img_bbx_btch = img_bbx_btch.to(self.device)

                        # batch of transformed images
                        output = self.A_CNN(img_bbx_btch)
                        output = torch.mean(output, dim=0).reshape(1, -1)
                    if torch.argmax(conf_function(output)).item() != output.shape[1] and \
                                    torch.max(conf_function(output[:,
                                                   :-1])).item() >= self.conf_thrsh_psd and yolo_pcls_conf.item() >= self.conf_thrsh_psd:

                        self.total_retrieved += 1
                        if targets_upi is not None and len(tlabels_upi) > 0:

                            iou = bbox_iou(torch.stack(yolo_t), tbox_upi)
                            if (iou > self.p_n_iou_thrs).any():
                                self.rtrv_positive += 1
                            else:
                                self.rtrv_negative += 1



    # def _PR_OD(self): #Object detector Percision and Recall (D_A(S_B)) meaning how well a given object detector D_A can perform on a giving dataset S_B, which is only annotated only for classes B, no annotation for classes A exist
    #
    #
    #
    #
    #                             #
    # def Percision_Recall (self):
    #
    #     total_positive =  0
    #     pbar = tqdm(enumerate(self.dataloader_UPI))
    #     for i, (imgs, UPI_targets, paths, _) in pbar:
    #
    #         UPI_targets = UPI_targets.to(self.device)
    #         imgs = imgs.to(self.device)
    #         roi, pred = self.model(imgs)  # inference and training outputs
    #
    #         # how many of fullly-labelled targets should be dropped and how
    #         output = non_max_suppression(roi, conf_thres=self.nms_conf, nms_thres=self.nms_thres)
    #         self._positive_recall(output,imgs, targets_upi=UPI_targets)
    #         total_positive +=len(UPI_targets)
    #
    #
    #         pbar.desc = ' *** At  iteration %d \t \t TPR %.2f  ; Recall %.2f ; all retrieved %d , all_positive %d'\
    #                     %(i+1, self.rtrv_positive/self.total_retrieved,  self.rtrv_positive/total_positive, self.total_retrieved, total_positive)
    #
    #
    #
    #     return self.rtrv_positive/self.total_retrieved,  self.rtrv_positive/total_positive

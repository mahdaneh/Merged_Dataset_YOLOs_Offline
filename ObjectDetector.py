import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import custom_datasets
import os
import torch.optim as optim
import torch
import math
from  models import *
import numpy as np
from models import *
from utils.utils import *
import torchvision
class offline_ObjctDtctr():
    def __init__(self,  device, args):

        self.hyp = {'giou': 3.31,  # giou loss gain
               'cls': 42.4,  # cls loss gain
               'cls_pw': 1.0,  # cls BCELoss positive_weight
               'obj': 40.0,  # obj loss gain (*=img_size/320 * 1.1 if img_size > 320)
               'obj_pw': 1.0,  # obj BCELoss positive_weight
               'iou_t': 0.213,  # iou training threshold
               'lr0': 0.001,  # initial learning rate (SGD=1E-3, Adam=9E-5)
               'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
               'momentum': 0.949,  # SGD momentum
               'weight_decay': 0.001,  # optimizer weight decay
               'fl_gamma': 0.5,  # focal loss gamma
               'hsv_h': 0.0103,  # image HSV-Hue augmentation (fraction)
               'hsv_s': 0.691,  # image HSV-Saturation augmentation (fraction)
               'hsv_v': 0.433,  # image HSV-Value augmentation (fraction)
               'degrees': 1.43,  # image rotation (+/- deg)
               'translate': 0.0663,  # image translation (+/- fraction)
               'scale': 0.11,  # image scale (+/- gain)
               'shear': 0.384}  # image shear (+/- deg)


        self.device = device
        self.args = args

        print ('--- model created on device %s'%self.device)
        self.model = Darknet(self.args.model_cfg, arc='default').to(self.device)


    def hyp_setup(self):
        # Hyperparameters (k-series, 53.3 mAP yolov3-spp-320) https://github.com/ultralytics/yolov3/issues/310


        # Overwrite hyp with hyp*.txt (optional)
        f = glob.glob('hyp*.txt')
        if f:
            for k, v in zip(self.hyp.keys(), np.loadtxt(f[0])):
                self.hyp[k] = v




    def _setup_dataloader(self, data_info ,flg_incld,augment):

        dataset= custom_datasets.incld_or_excld_dataset(info_file=data_info['class_configuration'],
                                                        flag_incld= flg_incld,
                                                        path = data_info['dataset_path'], img_size= self.args.img_size,
                                                        augment=augment, rect=self.args.rect,
                                                        batch_size=self.args.batch_size, hyp = self.hyp)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.args.batch_size,
                                                 num_workers=0,
                                                 shuffle=not self.args.rect,
                                                 # Shuffle=True unless rectangular training is used
                                                 pin_memory=True,
                                                 collate_fn=custom_datasets.collate_fn)
        return dataset, dataloader

    def _setup_optimizer(self):
        print ('==== setup optimizer === \n %s with Milestone %s'%('Adam' if self.args.adam else 'SGD', self.args.milestone ))
        pg0, pg1 = [], []  # optimizer parameter groups
        for k, v in dict(self.model.named_parameters()).items():
            if 'Conv2d.weight' in k:
                pg1 += [v]  # parameter group 1 (apply weight_decay)
            else:
                pg0 += [v]  # parameter group 0

        if self.args.adam: # ADAM
            self.optimizer= optim.Adam(pg0, lr=self.hyp['lr0'])

        else: # SGD
            self.optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

        self.optimizer.add_param_group({'params': pg1, 'weight_decay': self.hyp['weight_decay']})  # add pg1 with weight_decay

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.milestone, gamma=0.1)

        return  self.optimizer, self.scheduler

    def fit(self,data_info, flg_incld=None,weight_dir=None):

        self.weight_dir = weight_dir
        try:
            os.makedirs(self.weight_dir)
        except:
            pass


        self.dataset, self.dataloader = self._setup_dataloader(data_info, flg_incld=flg_incld, augment=True)

        self.names = self.dataset.class_names
        focusd_names = [*self.dataset.fcs_lbl.keys()]

        self.optimizer, self.scheduler = self._setup_optimizer()

        nc = len(self.names)
        last = self.weight_dir +'/last.pt'
        best = self.weight_dir + '/best.pt'

        batch_size = self.args.batch_size

        self.model.nc = nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model

        torch_utils.model_info(self.model, report='summary')  # 'full' or 'summary'

        nb = len(self.dataloader)

        f_result = open(self.weight_dir+'/training_results.txt','a')
        f_result.write(('%10s' * 7)% ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets'))
        
        st_epoch, best_fitness = load_pre_weights( self.device, self.model, self.optimizer, self.args)


        for epoch in range( st_epoch ,self.args.epochs):  # epoch ------------------------------------------------------------------
            self.model.train()
            for g in self.optimizer.param_groups: print('Learning Rate ', g['lr'])

            print(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets'))

            mloss = torch.zeros(4).to(self.device)  # mean losses

            pbar = tqdm(enumerate(self.dataloader))  # progress bar
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------

                ni = i + nb * epoch  # number integrated batches (since train start)
                if ni%10==0:
                    fname = 'train_batch%g.jpg' % i
                    plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)


                roi, pred = self.model(imgs)

                loss, loss_items = compute_loss(pred, targets, self.model,self.device)

                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return



                # Scale loss by nominal batch_size of 64
                loss *= batch_size / (batch_size *self.args.accumulate)

                loss.backward()

                # Accumulate gradient for x batches before optimizing
                if ni % self.args.accumulate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Print batch results
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)

                s = ('%10g/%g' ) %(epoch, self.args.epochs-1)+ '%10.3g' * 6 %(mem, *mloss, len(targets))
                pbar.set_description(s)

                # end batch ------------------------------------------------------------------------------------------------

            # Update scheduler
            self.scheduler.step()

            f_result.write(s+'\n')
            # Update best mAP
            fitness = mloss[-1]  # total loss

            if fitness < best_fitness:
                best_fitness = fitness

            chkpt = {'epoch': epoch,
                     'best_fitness': best_fitness,
                     'classes':self.names,
                     'focused_classes':focusd_names,
                     'model': self.model.module.state_dict() if type(self.model) is nn.parallel.DistributedDataParallel else self.model.state_dict(),
                     'optimizer': None if (epoch+1)==self.args.epochs else self.optimizer.state_dict()}


            # Save last checkpoint
            torch.save(chkpt, last)

            # Save best checkpoint
            if best_fitness == fitness:
                torch.save(chkpt, best)

            # Delete checkpoint
            del chkpt



    def predict_evaluate(self, data_info_test=None, flg_incld = None, model_weights  = None, nms_thres=None, nms_conf_thres= None, iou_thres=None):
        if model_weights is None: f_result = open(self.weight_dir +'/results_%s'%data_info_test['dataset_name'],'a')
        else: f_result = open('/'.join(model_weights.split('/')[:-1])+'/results_%s'%data_info_test['dataset_name'],'a')

        if data_info_test is None:
            print ('Test performs on training set with the classes on focus %s'%self.dataset.fcs_lbl)
            dataloader = self.dataloader
            names = self.names
        else:
            dataset, dataloader = self._setup_dataloader(data_info_test, flg_incld=flg_incld, augment=False)
            names = dataset.class_names

        nc = len(names)

        model = self.model
        device = next(model.parameters()).device  # get model device
        print(model_weights)
        if model_weights is not None:

            model.load_state_dict(torch.load(model_weights, map_location=device)['model'])

        print ('the used device by model %s'%device)
        verbose = True



        with torch.no_grad():
            seen = 0
            model.eval()

            s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
            p, r, f1, mp, mr, map, mf1, mem = 0., 0., 0., 0., 0., 0., 0. , 0.
            loss = torch.zeros(3)
            jdict, stats, ap, ap_class = [], [], [], []
            for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):

                targets = targets.to(device)
                imgs = imgs.to(device)
                _, _, height, width = imgs.shape  # batch size, channels, height, width

                # Run model
                inf_out, train_out = model(imgs)  # inference and training outputs


                # Run NMS
                output = non_max_suppression(inf_out, conf_thres=nms_conf_thres, nms_thres=nms_thres)

                # Statistics per image
                for si, pred in enumerate(output):
                    labels = targets[targets[:, 0] == si, 1:]
                    nl = len(labels)
                    tcls = labels[:, 0].tolist() if nl else []  # target class
                    seen += 1

                    if pred is None:
                        if nl:
                            stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                        continue

                    # Clip boxes to image bounds
                    clip_coords(pred, (height, width))

                    # Assign all predictions as incorrect
                    correct = [0] * len(pred)
                    if nl:
                        detected = []
                        tcls_tensor = labels[:, 0]

                        # target boxes
                        tbox = xywh2xyxy(labels[:, 1:5])
                        tbox[:, [0, 2]] *= width
                        tbox[:, [1, 3]] *= height


                        # Search for correct predictions
                        for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                            # Break if all targets already located in image
                            if len(detected) == nl:
                                break

                            # Continue if predicted class not among image classes
                            if pcls.item() not in tcls:
                                continue

                            # Best iou, index between pred and targets
                            m = (pcls == tcls_tensor).nonzero().view(-1)
                            iou, bi = bbox_iou(pbox, tbox[m]).max(0)


                            # If iou > threshold and class is correct mark as correct
                            if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                                correct[i] = 1
                                detected.append(m[bi])

                    # Append statistics (correct, conf, pcls, tcls)
                    stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))




        # Compute statistics
        stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
        if len(stats):
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%10.3g' * 6  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

        f_result.write(pf % ('all', seen, nt.sum(), mp, mr, map, mf1)+'\n')
        # Print results per class
        if verbose and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):

                print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
                f_result.write(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i])+'\n')

        # Return results
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]

        return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps


    def save_pseudo_label(self,data_info_train=None, flg_incld = None,
                          model_weights  = None, nms_thres=None, nms_conf_thres= None, pesudo_thrshld=None):

        dataset, dataloader = self._setup_dataloader(data_info_train, flg_incld=flg_incld,augment=False)
        names = dataset.class_names

        nc = len(names)

        model = self.model
        device = next(model.parameters()).device  # get model device

        model.load_state_dict(torch.load(model_weights, map_location=device)['model'])

        print ('the used device by model %s'%device)

        with torch.no_grad():
            model.eval()
            for batch_i, (imgs, _, paths, shapes) in enumerate(tqdm(dataloader)):


                imgs = imgs.to(device)
                _, _, height, width = imgs.shape  # batch size, channels, height, width

                # Run model
                inf_out, train_out = model(imgs)  # inference and training outputs


                # Run NMS
                output = non_max_suppression(inf_out, conf_thres=nms_conf_thres, nms_thres=nms_thres)

                # Statistics per image
                for si, pred in enumerate(output):

                    path = paths[si]
                    path = path.replace('labels_original', 'labels_pseudo')
                    try:
                        pseudo_dir = '/'.join(path.split('/')[:-1])
                        os.makedirs( pseudo_dir, exist_ok=True)
                    except:
                        pass



                    # Clip boxes to image bounds
                    clip_coords(pred, (height, width))


                    plt.imshow(imgs[si].cpu().numpy().transpose(1, 2, 0))

                    # Search for correct predictions
                    for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):
                        pbox= np.asarray([a.item() for a in pbox])
                        if torch.max(pcls_conf).cpu()>= pesudo_thrshld:
                            f = open(path, 'a')
                            plt.plot(pbox[[0, 2, 2, 0, 0]], pbox[[1, 1, 3, 3, 1]], '-')
                            plt.axis('off')

                            pbox = xyxy2xywh(pbox)
                            pbox[[0,2]] /= width
                            pbox[[1,3]] /= height
                            f.write(str(int(pcls.item()))+' '+ " ".join([str(a) for a in pbox])+'\n')
                    plt.savefig(path.replace('.txt','.jpg'))







def load_pre_weights(  device, model, optimizer, opt):
    st_epoch, best_fitness = 0, np.inf

    if opt.resume:
        best = opt.weight_dir + '/best.pt'
        chkpt = torch.load(best, map_location=device)

        print ('Load pre-trained models at %s'%best)
        chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(chkpt['model'], strict=False)

        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])

            best_fitness = chkpt['best_fitness']


        st_epoch = chkpt['epoch'] + 1
        best_fitness = chkpt['best_fitness']
        del chkpt

    if opt.transfer:
        chkpt = torch.load(opt.pre_weight)
        model.load_state_dict(chkpt['model'])

    return st_epoch, best_fitness













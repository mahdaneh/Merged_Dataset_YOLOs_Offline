import ObjectDetector
import argparse
import torch
import os
from utils.utils import *
def main ():

    # python offline_ODs.py --batch-size 16 --accumulate 4 --model-cfg cfg/yolov3-spp.cfg --working-dir  voc7-voc12-Exp1 --milestone 22 40 150

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273)  # 500200 batches at bs 16, 117263 images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=4)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
    parser.add_argument('--model-cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--phase', type=str, default='train')

    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--transfer', action='store_true', help='transfer learning')
    parser.add_argument('--pre-weight', type=str)
    parser.add_argument('--working-dir', type=str, default=None, help='initial weights')

    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    # adding
    parser.add_argument('--milestone', type=int, nargs='+')

    args = parser.parse_args()

    print (args)
    merged_data_info = os.path.join(args.working_dir, 'data_configuration')
    weight_dir = os.path.join(args.working_dir, 'weights')
    try:
        os.makedirs(weight_dir, exist_ok=True)
    except:
        pass

    data_setup = read_data_stp(merged_data_info)
    device_indx = [0, 1]
    devices = [torch.device(de) for de in device_indx]

    [print(data) for data in data_setup]


    ODs = [ObjectDetector.offline_ObjctDtctr( devices[1], args), ObjectDetector.offline_ObjctDtctr( devices[1], args)]


    if args.phase =='train':

        for ii, (OD,data) in enumerate(zip(ODs, data_setup)):
            """for now flg_incld=Flase is correct for having the same setting as the one for Our_YOLO"""
            if ii ==0:
                OD.fit(data, flg_incld =True, weight_dir= weight_dir + '/' + data['dataset_name'])
                OD.predict_evaluate(data_info_test=data, flg_incld = True, model_weights  = None,
                                    nms_thres=0.5, nms_conf_thres= 0.001, iou_thres=0.5)

    elif args.phase=='test': #trained on S_A but test on S_B by removing B classes but including GTs for classes A=> for generalization ability of OD_{S_A}

        models_weights = [weight_dir + '/' + data['dataset_name']+'/last.pt' for data in data_setup]


        for OD, data_test, model_weight in zip(ODs, data_setup , models_weights):
            #::TODO I set here flg_incld=True since mistakenly trained OD_{S_A} on excluded classes
            with torch.no_grad():
                OD.predict_evaluate(data_info_test=data_test, flg_incld=False, \
                       model_weights=model_weight, nms_thres=0.5, nms_conf_thres= 0.001 , iou_thres=0.5)

    elif args.phase=='pseudo_label':

        # models_weights = [weight_dir + '/' + data['dataset_name']+'/last.pt' for data in data_setup]
        #
        # for OD, data_test, model_weight in zip(ODs, data_setup , models_weights):
            model_weight = weight_dir + '/' + data_setup[0]['dataset_name']+'/last.pt'
            data_test = data_setup[0]
            with torch.no_grad():
                ODs.save_pseudo_label(data_info_train=data_test, flg_incld = False,
                              model_weights  = model_weight, nms_thres=0.5, nms_conf_thres= 0.001, pesudo_thrshld=0.8)






if __name__=='__main__':
    main()


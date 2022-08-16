#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
import sys

from matplotlib.pyplot import draw_if_interactive
sys.path.remove('/home/lab602.10977014_0n1/.pipeline2/10977014/YOLOX')
sys.path.append("/home/lab602.10977014_0n1/.pipeline/10977014/YOLOXP/")


import cv2

import torch
import numpy as np
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis, vis_z

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "-demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-segs", default=True, help="draw segmentation",
                        action="store_true")

    parser.add_argument(
        "--path", default="datasets/1.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/example/custom/yolox_s_seg.py",
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default="YOLOX_outputs/yolox_s_seg_paper/latest_ckpt.pth", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.1, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def get_image_gt(image_names) :
    from yolox.data import COCODataset
    data = COCODataset(
        data_dir=None,
        json_file="instances_val2017.json",
        name="val2017",
        preproc=None,
        cache=False,
        is_train=False
        )
    # res, img_info, resized_info, file_name = data.pull_item(image_names)
    # print(data.json_file)
    # res, img_info, resized_info, _ = data.annotations[image_names-1]
    res, img_info, resized_info, file_name, _=data.load_anno_from_ids(image_names)
    # print("img_info" +str(img_info))
    # print("-------get_image_gt---------")
    # print(res)
    # print("-------get_image_gt---------")
    return res


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=ES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
        segs=False
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.img_channel = exp.img_channel
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.segs = segs
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            if self.img_channel == 4:
                img = cv2.imread(img, -1)
            else:
                img = cv2.imread(img)
        else:
            img_info["file_name"] = None



        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

   
        img, _, _ = self.preproc(img, None, self.test_size, None )
        


        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()

        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs, seg_output = self.model(img)
            if self.decoder is not None:  # None
                outputs, seg_output = self.decoder(outputs, seg_output, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}fps".format(1/(time.time() - t0)))
        return outputs, seg_output, img_info

    def visual(self, output, seg_output, img_info, cls_conf=0.35, draw_seg=False):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img, np.zeros_like(img)
        output = output.cpu()
        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= ratio
        # kps = output[:, 7:] / ratio if draw_kp else []
        if draw_seg:


            seg = seg_output.max(axis=0)[1].cpu().numpy()

            h, w, _ = img.shape
            sh, sw = seg.shape



            # seg = cv2.resize(
            #     seg, (int(sw / ratio), int(sh / ratio)),
            #     interpolation=cv2.INTER_NEAREST)[:h, :w]

            seg = cv2.resize(
                seg, (int(sw / ratio), int(sh / ratio)),
                interpolation=cv2.INTER_NEAREST)[:h, :w]


        else:
            seg = []

        z = output[:,4]
        cls = output[:, 7]
        scores = output[:, 5] * output[:, 6]

        # img_id = img_info["file_name"].split('.')[0]        
        # gt = get_image_gt(int(img_id)) 
        vis_res, seg_mask = vis(img, bboxes, scores, z ,cls, cls_conf, self.cls_names, seg)


        # gt_bboxes = gt[:, 0:4]
        # gt_bboxes /= ratio
        # gt_z = gt[:,4]
        # vis_res_gt = vis_z(vis_res,gt_bboxes,gt_z)

        return vis_res , seg_mask
        # return vis_res_gt , seg_mask


def image_demo(predictor, vis_folder, path, current_time, save_result, draw_seg):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, seg_outputs, img_info = predictor.inference(image_name)
        if seg_outputs is None:
            seg_outputs = [None for _ in range(len(outputs))]
        
        result_image, seg_mask = predictor.visual(outputs[0], seg_outputs[0], img_info,
                                        predictor.confthre, draw_seg)

        print(seg_mask.shape)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
            if draw_seg:
                if '.jpg' in save_file_name:
                    cv2.imwrite(save_file_name.replace('.jpg', '_seg.jpg'), seg_mask)
                else:
                    cv2.imwrite(save_file_name.replace('.png', '_seg.png'), seg_mask)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

   

    counter = 0
    totalfps = 0 

    while True:
        start_time = time.time()
        ret_val, frame = cap.read()
        
        if ret_val:

            outputs, seg_outputs, img_info = predictor.inference(frame)
            print("FPS: ", 1.0 / (time.time() - start_time))
            fps = 1.0 / (time.time() - start_time)
            
            result_frame, seg_mask = predictor.visual(outputs[0], seg_outputs[0], img_info, predictor.confthre, True)

            totalfps += fps 
            counter+=1


            # result_frame = cv2.resize(result_frame, (1920, 1080))	
            if args.save_result:
                vid_writer.write(result_frame)
                

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                print("APFPS: ", totalfps/counter)
                break
        else:
            break

    print("APFPS: ", totalfps/counter)
    # print("FPS: ", counter / (time.time() - start_time))


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size, exp.img_channel)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device,
                          args.fp16, args.legacy )

    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result,
                   args.segs)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)

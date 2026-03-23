import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# import psutil

from PIL import Image

# from .data.dataset import FSSDataset
from .backbone import Backbone
from .utils import commonutils as utils
from .utils import segutils as segutils
from .utils import crfhelper as crfutils
from . import contrastivehead as ctrutils
from . import denseaffinity as dautils

def set_args(_args):
    global args
    args = _args
    args.backbone = 'resnet50'
    args.nworker = 0
    args.bsz = 1
    args.fold = 0

# def makeDataloader():
#     FSSDataset.initialize(img_size=400, datapath=args.datapath)
#     dataloader = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)
#     return dataloader

def makeConfig():
    config = ctrutils.ContrastiveConfig()
    config.fitting.keepvarloss = True
    config.fitting.maskloss = True
    config.fitting.triplet_loss = True
    config.fitting.proto_loss = False
    config.fitting.selfattention_loss = False
    config.fitting.o_t_contr_proto_loss = False
    config.fitting.symmetricloss = False
    config.fitting.q_nceloss = True
    config.fitting.s_nceloss = True
    config.fitting.split = True
    config.fitting.num_epochs = 20
    config.fitting.lr = 1e-2
    config.model.out_channels = 64
    config.featext.fit_every_episode = False
    config.aug.blurkernelsize = [1]
    config.aug.n_transformed_imgs = 2
    config.aug.maxjitter = 0.0
    config.aug.maxangle = 0
    config.aug.maxscale = 1
    config.aug.maxshear = 20
    config.aug.apply_affine = True
    config.fitting.num_epochs = 300
    return config

def makeFeatureMaker(config, device='cpu', randseed=2, feat_extr_method=None, num_classes=5):
    utils.fix_randseed(randseed)
    if feat_extr_method is None:
        feat_extr_method = Backbone("resnet50").to(device).extract_feats
    feat_maker = ctrutils.FeatureMaker(feat_extr_method, num_classes, config)
    utils.fix_randseed(randseed)
    feat_maker.norm_bb_feats = False
    return feat_maker

class TVGTANet(nn.Module):
    def __init__(self, num_classes=5, verbosity=0, pascal_path="data/pascal", num_pascal_masks=100):
        super(TVGTANet, self).__init__()
        self.damat_comp = dautils.DAMatComparison()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.feat_maker = makeFeatureMaker(makeConfig(), device=self.device, num_classes=num_classes)
        self.thresh_method = 'pred_mean'
        self.post_proc_method = 'off'
        self.verbosity = verbosity
        self.inference_time = None
        self.memory_usage = None
        # self._ps_process = psutil.Process() if psutil else None
        
        segmentations_pascal = os.path.join(pascal_path, "SegmentationClass")
        
        self.num_pascal_masks = num_pascal_masks
        # Load 100 random masks from the Pascal dataset
        self.pascal_masks = []
        for mask_file in os.listdir(segmentations_pascal)[:num_pascal_masks]:
            if mask_file.endswith(".png"):
                mask_path = os.path.join(segmentations_pascal, mask_file)
                self.pascal_masks.append(self.read_mask(mask_path))
    
    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def taskAdapt(self, b):
        
        self.q_pascal = self.pascal_masks[0].to(self.device).float()  # For simplicity, using the first mask as the query Pascal mask
        self.s_pascal = [mask.to(self.device).float() for mask in self.pascal_masks[:b["support_imgs"].shape[1]]] 
        # Using the first N masks for support, where N is the number of support images
        # Resize to match the input image size if necessary
        self.q_pascal = F.interpolate(self.q_pascal.unsqueeze(0).unsqueeze(0), size=(b['query_img'].shape[2], b['query_img'].shape[3]), mode='nearest')[0, 0]
        self.s_pascal = [F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(b['support_imgs'].shape[3], b['support_imgs'].shape[4]), mode='nearest')[0, 0] for mask in self.s_pascal]
        self.s_pascal = torch.stack(self.s_pascal, dim=0).unsqueeze(0)  # Shape: [1, N_support, H, W]
        
        self.q_img, self.s_img, self.s_mask, self.class_id = \
            b['query_img'], b['support_imgs'], b['support_masks'], b['class_id']
        self.task_adapted = self.feat_maker.taskAdapt(self.q_img, self.s_img, self.s_mask, self.class_id, self.q_pascal, self.s_pascal)

    def compare_feats(self):
        if self.task_adapted is None:
            print("error, do task adaption first")
            return None
        self.logit_mask = self.damat_comp.forward(self.task_adapted[0], self.task_adapted[1], self.s_mask)
        return self.logit_mask

    def threshold(self, method=None):
        if self.logit_mask is None:
            print("error, calculate logit mask first (do forward pass)")
        if method is None:
            method = self.thresh_method
        self.thresh = segutils.calcthresh(self.logit_mask, self.s_mask, method)
        self.pred_mask = (self.logit_mask > self.thresh).float()
        return self.thresh, self.pred_mask

    def postprocess(self):
        if self.post_proc_method == 'off':
            apply = False
        elif self.post_proc_method == 'always':
            apply = True
        elif self.post_proc_method == 'dynamic':
            apply = crfutils.crf_is_good(self)
        else:
            apply = False
            print(f'Unknown postproc method: {self.post_proc_method=}')
        return crfutils.apply_crf(self.q_img, self.logit_mask, segutils.thresh_fn(self.thresh_method)).to(self.device) if apply else self.pred_mask

    def predict_mask_nshot(self, batch):
        
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        self.taskAdapt(batch)
        start_time = time.time()
        self.logit_mask = self.compare_feats()
        self.thresh, self.pred_mask = self.threshold()
        self.pred_mask = self.postprocess()
        end_time = time.time()
        self.inference_time = end_time - start_time
        if self.device.type == 'cuda':
            self.memory_usage = torch.cuda.max_memory_allocated(self.device)
        # elif self._ps_process is not None:
        #     self.memory_usage = self._ps_process.memory_info().rss
        return self.logit_mask, self.pred_mask

    # def calc_metrics(self):
        # self.area_inter, self.area_union = Evaluator.classify_prediction(self.pred_mask, self.batch)
        # self.fgratio_pred = self.pred_mask.float().mean()
        # self.fgratio_gt = self.batch['query_mask'].float().mean()
        # return self.area_inter[1] / self.area_union[1]

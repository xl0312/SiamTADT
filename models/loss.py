# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from siamdth.core.config import cfg
from siamdth.models.iou_loss import linear_iou


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    if cfg.BAN.BAN:
        diff = (pred_loc - label_loc).abs()
        diff = diff.sum(dim=1)
    else:
        diff = None
    loss = diff * loss_weight
    return loss.sum().div(pred_loc.size()[0])


def select_iou_loss(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    pos = label_cls.data.eq(1).nonzero().squeeze().cuda()

    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = torch.index_select(pred_loc, 0, pos)

    label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    label_loc = torch.index_select(label_loc, 0, pos)

    return linear_iou(pred_loc, label_loc)
class SiamCARLossComputation(object):
    """
    This class computes the SiamCAR losses.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.centerness_loss_func = nn.BCEWithLogitsLoss()


    def prepare_targets(self, points, labels, gt_bbox):

        labels, reg_targets = self.compute_targets_for_locations(
            points, labels, gt_bbox
        )

        return labels, reg_targets

    def compute_targets_for_locations(self, locations, labels, gt_bbox):
        # reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        bboxes = gt_bbox

        labels = labels.view(self.cfg.TRAIN.OUTPUT_SIZE**2, -1)


        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        s1 = reg_targets_per_im[:, :, 0] > 0.6*((bboxes[:, 2]-bboxes[:, 0])/2).float()
        s2 = reg_targets_per_im[:, :, 2] > 0.6*((bboxes[:, 2]-bboxes[:, 0])/2).float()
        s3 = reg_targets_per_im[:, :, 1] > 0.6*((bboxes[:, 3]-bboxes[:, 1])/2).float()
        s4 = reg_targets_per_im[:, :, 3] > 0.6*((bboxes[:, 3]-bboxes[:, 1])/2).float()
        is_in_boxes = s1*s2*s3*s4
        pos = np.where(is_in_boxes.cpu() == 1)
        labels[pos] = 1

        return labels.permute(1, 0).contiguous(), reg_targets_per_im.permute(1,0,2).contiguous()

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, centerness, labels, reg_targets):



        label_cls, reg_targets = self.prepare_targets(locations, labels, reg_targets)
        labels_flatten = (label_cls.view(-1))
        reg_targets_flatten = (reg_targets.view(-1, 4))
        centerness_flatten = (centerness.view(-1))

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)


        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]


        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)


            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:

            centerness_loss = centerness_flatten.sum()

        return centerness_loss
    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x: x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params

def make_siamcar_loss_evaluator(cfg):
    loss_evaluator = SiamCARLossComputation(cfg)
    return loss_evaluator
'''
l  = bb
'''
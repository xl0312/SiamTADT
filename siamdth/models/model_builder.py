# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamdth.core.config import cfg
from siamdth.models.loss import select_cross_entropy_loss, select_iou_loss, make_siamcar_loss_evaluator
from siamdth.models.backbone import get_backbone
from siamdth.models.head import get_ban_head
from siamdth.models.neck import get_neck
from siamdth.core.xcorr import xcorr_depthwise
from siamdth.utils.feature_tower import matrix
from siamdth.models.head.ban import CARHead
from siamdth.utils.location_grid import compute_locations
from siamdth.models.head.ban import Channel_Attention

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)
        self.channel_attention = Channel_Attention(in_dim=256)
        self.featuretower = matrix(256, 256)
        self.towrconv1 = nn.Conv2d(768, 256, (1, 1))
        self.centerness = CARHead(in_channels=256)
        self.centernessloss = make_siamcar_loss_evaluator(cfg)
        # build ban head


    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        def at(list):
            for i in range(3):
                list[i] = self.channel_attention(list[i])
            return list



        channel_attention_zf = at(self.zf)
        channel_attention_xf = at(xf)

        cls, loc, stage_cls, channel_stage_loc = self.head(channel_attention_zf, channel_attention_xf)
        spatisl_cls, loc, stage_cls_spital, stage_loc = self.head(self.zf, xf)

        zf_tower, xf_tower = self.featuretower(self.zf[-1], xf[-1])
        # cls_tower, loc_tower, stage_cls_tower, stage_loc_tower = self.xcorr_depthwise(zf_tower, xf_tower)
        features = xcorr_depthwise(xf_tower[0], zf_tower[0])
        for i in range(len(xf_tower) - 1):
            features_new = xcorr_depthwise(xf_tower[i + 1], zf_tower[i + 1])
            features = torch.cat([features, features_new], 1)
        features = self.towrconv1(features)
        centerness = self.centerness(features, cls)

        return {
                'cls': cls,
                'loc': loc,
                'cen': centerness
               }


    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        centercls = data['centercls'].cuda()
        CARBOX = data['CARbbox'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        def at(list):
            for i in range(3):
               list[i] = self.channel_attention(list[i])
            return list
        channel_attention_zf = at(zf)
        channel_attention_xf = at(xf)

        cls, loc, stage_cls, channel_stage_loc = self.head(channel_attention_zf, channel_attention_xf)
        spatisl_cls, loc, stage_cls_spital, stage_loc = self.head(zf, xf)




        zf_tower, xf_tower = self.featuretower(zf[-1], xf[-1])
        # cls_tower, loc_tower, stage_cls_tower, stage_loc_tower = self.xcorr_depthwise(zf_tower, xf_tower)
        features = xcorr_depthwise(xf_tower[0], zf_tower[0])
        for i in range(len(xf_tower) - 1):
            features_new = xcorr_depthwise(xf_tower[i + 1], zf_tower[i + 1])
            features = torch.cat([features, features_new], 1)
        features = self.towrconv1(features)
        centerness = self.centerness(features, cls)
        #a = cls
        # get loss

        # cls loss with cross entropy loss
        locations = compute_locations(cls, cfg.POINT.STRIDE)


        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        # loc loss with iou loss
        loc_loss = select_iou_loss(loc, label_loc, label_cls)

        def stage_cls_loss(stage_loc_output):
            stage_cls_loss = []
            for i in range(len(stage_loc_output)):
                c = stage_loc_output[i]
                c = self.log_softmax(c)
                c = select_cross_entropy_loss(c, label_cls)*cfg.TRAIN.CLS_WEIGHT
                stage_cls_loss.append(c)
            return stage_cls_loss

        def stage_loc_loss(stage_loc_outpot):
            stage_loc_loss =[]
            for i in range(len(stage_loc_outpot)):
                l = stage_loc_outpot[i]
                l = select_iou_loss(l, label_loc, label_cls)*cfg.TRAIN.LOC_WEIGHT
                stage_loc_loss.append(l)
            return stage_loc_loss

        def list_add(a, b):
            c = []
            for i in range(len(a)):
                c.append(a[i] + b[i])
            return c
        stage_loc_loss = stage_loc_loss(stage_loc)
        stage_cls_loss = stage_cls_loss(stage_cls)
        stage_loss = list_add(stage_cls_loss, stage_loc_loss)
        stage_loss_p2 = stage_loss[0]
        stage_loss_p3 = stage_loss[1]
        stage_loss_p4 = stage_loss[2]
        centernessloss = self.centernessloss(locations, centerness, centercls, CARBOX)

        total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss + \
            stage_loss_p2 + stage_loss_p3 + stage_loss_p4 + centernessloss


        outputs = {}
        outputs['total_loss'] = total_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['centernessloss'] = centernessloss
        #outputs['fusion_loss'] = fusion_loss

        return outputs
'''
    def xcorr_depthwise_tower(x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
'''
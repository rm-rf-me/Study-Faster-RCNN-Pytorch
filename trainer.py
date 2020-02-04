from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch as t
from utils import array_tool as at
from utils.vis_tool import Visualizer

from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])
# 非常强大的四个loss

class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma  #是在_faster_rcnn_loc_loss调用用来计算位置损失函数用到的超参数，

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        # 用于从20000个候选anchor中产生256个anchor进行二分类和位置回归，
        # 也就是为rpn网络产生的预测位置和预测类别提供真正的ground_truth标准
        self.proposal_target_creator = ProposalTargetCreator()
        # AnchorTargetCreator和ProposalTargetCreator是为了生成训练的目标（或称ground truth），只在训练阶段用到，
        # ProposalCreator是RPN为Fast R-CNN生成RoIs，在训练和测试阶段都会用到。
        # 所以测试阶段直接输进来300个RoIs，而训练阶段会有AnchorTargetCreator的再次干预。

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean    #(0., 0., 0., 0.)
        self.loc_normalize_std = faster_rcnn.loc_normalize_std      #(0.1, 0.1, 0.2, 0.2)

        self.optimizer = self.faster_rcnn.get_optimizer()           #SGD
        # visdom wrapper
        self.vis = Visualizer(env=opt.env)

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)     #混淆矩阵，就是验证预测值与真实值精确度的矩阵ConfusionMeter(2)括号里的参数指的是类别数
        self.roi_cm = ConfusionMeter(21)    #roi的类别有21种（20个object类+1个background）
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.
        当前batch size只为一

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]         #batch size = 1
        if n != 1:                  #绝了
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape     #N，C，H，W
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        # emmmmmm。。。
        bbox = bboxes[0]                #bbox维度(N, R, 4)
        label = labels[0]               #labels维度为（N，R）
        rpn_score = rpn_scores[0]       #hh*ww*9
        rpn_loc = rpn_locs[0]           #（hh*ww*9，4）
        roi = rois                      # (2000,4)

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        # rpn_score为rpn网络得到的（20000个）与anchor_target_creator得到的2000个label求交叉熵损失
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]     #不计算背景类
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]

        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())   # 混淆矩阵


        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]     # roi_cls_loc为VGG16RoIHead的输出（128*84）， n_sample=128
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4) # roi_cls_loc=（128,21,4）
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()     # 128个标签
        gt_roi_loc = at.totensor(gt_roi_loc)    # proposal_target_creator()生成的128个proposal与bbox求得的偏移量dx,dy,dw,dh

        roi_loc_loss = _fast_rcnn_loc_loss(     #采用smooth_l1_loss
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())    #求交叉熵损失

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())    #混淆矩阵

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]       #四个loss加起来
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()      # 将梯度数据全部清零
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()       # 更新参数
        self.update_meters(losses)  # 将所有损失的数据更新到可视化界面上,最后将losses返回
        return losses               # 返回loss

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        # 将所有损失的数据更新到可视化界面上,最后将losses返回
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def _smooth_l1_loss(x, t, in_weight, sigma):
    # sigma设置为1
    # 具体损失函数公式见论文

    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    # rpn_loc为rpn网络回归出来的偏移量（20000个）即pred loc
    # gt_rpn_loc为anchor_target_creator函数得到2000个anchor与bbox的偏移量，即gt loc
    # rpn_sigma=1.
    # 具体损失函数公式见论文

    in_weight = t.zeros(gt_loc.shape).cuda()
    # in_weight代表的是权重，
    # 用in_weight来作为权重，只将那些不是背景的anchor/ROIs的位置加入到损失函数的计算中来，
    # 方法就是只给不是背景的anchor/ROIs的in_weight设置为1,这样就可以完成loc_loss的求和计算

    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1   #非背景权重为1

    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss 除去背景类
    return loc_loss

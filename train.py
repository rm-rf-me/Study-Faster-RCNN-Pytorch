from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    # 预测框的位置，预测框的类别和分数以及相应的真实值的类别分数
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()

    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]    # 宽高
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num:
            break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse (kwargs)     # 全部的设置

    dataset = Dataset(opt)  # 数据集
    print('load data')

    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    # pin memory：锁页内存，内存为所欲为的时候为true，详情见：https://blog.csdn.net/yangwangnndd/article/details/95385628
    # num worker：加载数据的线程数，默认为8。具体数值的选取由训练时间决定，当训练时间快于加载时间时则需要增加线程
    # shuffle=True允许数据打乱排序
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:   #接下来判断opt.load_path是否存在，如果存在，直接从opt.load_path读取预训练模型，然后将训练数据的label进行可视化操作
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr

    for epoch in range(opt.epoch):  # 训练迭代的次数opt.epoch=14也在config.py文件中都预先定义好，属于超参数
        trainer.reset_meters()      # 首先在可视化界面重设所有数据
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            # 然后从训练数据中枚举dataloader,设置好缩放范围，将img,bbox,label,scale全部设置为可gpu加速
            trainer.train_step(img, bbox, label, scale)     # 调用trainer.py中的函数trainer.train_step(img,bbox,label,scale)进行一次参数迭代优化过程

            # 判断数据读取次数是否能够整除plot_every(是否达到了画图次数)
            if (ii + 1) % opt.plot_every == 0:
                # 如果达到判断debug_file是否存在，用ipdb工具设置断点
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                # 调用trainer中的trainer.vis.plot_many(trainer.get_meter_data())将训练数据读取并上传完成可视化
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)
                # 将每次迭代读取的图片用dataset文件里面的inverse_normalize()函数进行预处理，将处理后的图片调用Visdom_bbox

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                # 调用 trainer.vis.text将rpn_cm也就是RPN网络的混淆矩阵在可视化工具中显示出来
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']   # learning rate
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)       # 将损失学习率以及map等信息及时显示更新

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)     # 用if判断语句永远保存效果最好的map
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay                        # if判断语句如果学习的epoch达到了9就将学习率*0.1变成原来的十分之一

        if epoch == 13: 
            break


if __name__ == '__main__':
    import fire

    fire.Fire()

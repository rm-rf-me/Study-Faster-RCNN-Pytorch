import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]                        # 9
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)  # 首先是加pad的512个3*3大小卷积核，输出仍为（N，512，h，w）
        #然后左右两边各有一个1 * 1卷积。
        # 左路为18个1 * 1卷积，输出为（N，18，h，w），即所有anchor的0 - 1类别概率（h * w约为2400，h * w * 9约为20000）。
        # 右路为36个1 * 1卷积，输出为（N，36，h，w），即所有anchor的回归位置参数。
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0) # 18，分类得分
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)   # 36，回归参数

        normal_init(self.conv1, 0, 0.01)        # 归一初始化
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """Forward Region Proposal Network.

        输入特征即feature map，调用函数_enumerate_shifted_anchor生成全部20000个anchor。
        然后特征经过卷积，在经过两路卷积分别输出rpn_locs, rpn_scores。
        然后rpn_locs, rpn_scores作为ProposalCreator的输入产生2000个rois，同时还有 roi_indices，
        这个 roi_indices在此代码中是多余的，
        因为我们实现的是batch_size=1的网络，一个batch只会输入一张图象。
        如果多张图象的话就需要存储索引以找到对应图像的roi。

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(     # hh * ww * 9个anchor对应到原图的坐标
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww) #9
        h = F.relu(self.conv1(x))   # 512个3x3卷积(n, 512, H/16,W/16)

        rpn_locs = self.loc(h)      # n_anchor（9）* 4个1x1卷积，回归坐标偏移量。（n, 9*4, hh, ww）
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4) # 转换为（n，hh，ww，9*4）后变为（n，hh*ww*9，4）
        # permute为维度转换函数
        # 当调用contiguous后，就会创建一个独立的布局和原来的那个变量独立开来。即深拷贝。

        rpn_scores = self.score(h)  #n_anchor（9）* 2个1x1卷积，回归类别。（9*2，hh,ww）
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()    #转换为（n，hh，ww，9*2）

        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)  #计算{Softmax}(x_{i}) = \{exp(x_i)}{\sum_j exp(x_j)}

        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()  #得到前景的分类概率
        rpn_fg_scores = rpn_fg_scores.view(n, -1)       #得到所有anchor的前景分类概率
        rpn_scores = rpn_scores.view(n, -1, 2)          #得到每一张feature map上所有anchor的网络输出值

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            # 调用ProposalCreator函数，
            # rpn_locs维度（hh*ww*9，4），
            # rpn_fg_scores维度为（hh*ww*9），
            # anchor的维度为（hh*ww*9，4），
            # img_size的维度为（3，H，W），H和W是经过数据预处理后的。
            # 计算（H/16）x(W/16)x9(大概20000)个anchor属于前景的概率，
            # 取前12000个并经过NMS得到2000个近似目标框G^的坐标。roi的维度为(2000,4)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    # 产生整个特征图所有的anchor

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    import numpy as xp

    # feat stride默认为16，即放大倍数，这里将横纵坐标都放大了16倍
    shift_y = xp.arange(0, height * feat_stride, feat_stride)   # 以feat_stride为间距产生从(0,height*feat_stride)的一行
    shift_x = xp.arange(0, width * feat_stride, feat_stride)    # shift_x就是以feat_stride产生从(0,width*feat_stride)的一行

    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    # meshgrid是一个画网格函数，做的事情就是求出不同x和y的坐标组合
    # 返回list,有两个元素,第一个元素是X轴的取值,第二个元素是Y轴的取值
    # 那么整体来看，将X和Y合并，得到的每一个 (X[i][j], Y[i][j]) 就是一种用X[i]和Y[j]得到的组合方式

    # X. Y = np.meshgrid (x, y)后，
    # 产生的大X以x的行为行，以y的元素个数为列构成矩阵
    # 产生的Y以y的行作为列，以x的元素个数作为列数产生矩阵

    # 以shift_x为行，以shift_y的行为列产生矩阵
    # 形成了一个纵横向偏移量的矩阵，也就是特征图的每一点都能够通过这个矩阵找到映射在原图中的具体位置

    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)
    # ravel将矩阵合并为一行，
    # stack将给出的各个arrays按照axis指定的维度整合，
    # y和x交替使用对应着左上和右下坐标的四个值的偏移量，即为（1， 4）
    # axis = 1堆叠后即为 （1， K， 4）

    A = anchor_base.shape[0]    # 读取基本anchor个数，应该为9
    K = shift.shape[0]          # 特征图元素总个数
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))  # anchor偏移量加上中心点偏移量得到最终的两点坐标
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)  # 整合形状
    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

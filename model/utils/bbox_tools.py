import numpy as np
import numpy as xp

import six
from six import __init__


def loc2bbox(src_bbox, loc):
    """Decode bounding boxes from bounding box offsets and scales.
    已知源bbox和位置偏差求目标框G

    Given bounding box offsets and scales computed by
    :meth:`bbox2loc`, this function decodes the representation to
    coordinates in 2D image coordinates.

    Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding
    box whose center is :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`,
    the decoded bounding box's center :math:`\\hat{g}_y`, :math:`\\hat{g}_x`
    and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are calculated
    by the following formulas.

    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`

    此处的坐标转换公式源于RCNN论文，公式中用到的为中心点坐标和宽、高，所以需要先行根据框左上和右下坐标计算

    The decoding formulas are used in works such as R-CNN [#]_.

    The output is same type as the type of the inputs.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
            已知的源边框，包括边框个数和每个边框对应的左上角和右下角作标值
        loc (array): An array with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_y, t_x, t_h, t_w`.
            x、y、h、w四个值的位置偏差

    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
        \\hat{g}_{ymax}, \\hat{g}_{xmax}`.

    """

    if src_bbox.shape[0] == 0:  #当R为0即没有边框时返回空
        return xp.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]        #根据左上角和右下角坐标计算中心坐标和宽高
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]       #[start:stop:step]，即分别从1-4开始每间隔4取值
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, xp.newaxis] + src_ctr_y[:, xp.newaxis]   #核心公式
    ctr_x = dx * src_width[:, xp.newaxis] + src_ctr_x[:, xp.newaxis]
    h = xp.exp(dh) * src_height[:, xp.newaxis]
    w = xp.exp(dw) * src_width[:, xp.newaxis]

    dst_bbox = xp.zeros(loc.shape, dtype=loc.dtype) #由中心坐标转换回左上右下坐标
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".
    已知源框和目标框求偏差

    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is
    :math:`(y, x) = p_y, p_x` and
    size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
    :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.

    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`

    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        dst_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`.
            These coordinates are
            :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.

    Returns:
        array:
        Bounding box offsets and scales from :obj:`src_bbox` \
        to :obj:`dst_bbox`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_y, t_x, t_h, t_w`.

    """

    height = src_bbox[:, 2] - src_bbox[:, 0]    #源框中心坐标
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]   #目标框中心坐标
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = xp.finfo(height.dtype).eps    #找到最小正数
    height = xp.maximum(height, eps)
    width = xp.maximum(width, eps)      #保证宽高都是不小于eps的正数

    dy = (base_ctr_y - ctr_y) / height  #根据公式计算偏移量
    dx = (base_ctr_x - ctr_x) / width
    dh = xp.log(base_height / height)
    dw = xp.log(base_width / width)

    loc = xp.vstack((dy, dx, dh, dw)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    求两个bbox相交的交并比，即交集和并集的比值
    两个边框的交集比上并集，显然IoU值的意义就是两个边框重叠部分的大小

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    #tl为交叉部分框左上角坐标最大值，为了利用numpy的广播性质，
    #bbox_a[:, None, :2]的shape是(N,1,2)，bbox_b[:, :2]shape是(K,2),
    #由numpy的广播性质，两个数组shape都变成(N,K,2)，
    #也就是对a里每个bbox都分别和b里的每个bbox求左上角点坐标最大值

    # bottom right
    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    #同理为右下角的最小值

    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2) #交集
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1) #a面积
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1) #b面积
    return area_i / (area_a[:, None] + area_b - area_i)


def __test():
    pass


if __name__ == '__main__':
    __test()


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    """Generate anchor base windows by enumerating aspect ratio and scales.

    对特征图features以基准长度为16、选择合适的ratios和scales取基准锚点anchor_base。
    （选择长度为16的原因是图片大小为600*800左右，基准长度16对应的原图区域是256*256，
    考虑放缩后的大小有128*128，512*512比较合适）

    根据基准点生成9个基本的anchor的功能，
    ratios=[0.5,1,2],anchor_scales=[8,16,32]是长宽比和缩放比例,
    anchor_scales也就是在base_size的基础上再增加的量，
    本代码中对应着三种面积的大小(16*8)^2 ,(16*16)^2  (16*32)^2
    也就是128,256,512的平方大小

    这个函数三个参数base_size, ratios 和scales，且都有默认值。
    base_size要和scales集合起来看，anchor的基础尺度=base_size*scales，
    例如这里默认base_size=16, scales=(8,16,32),
    那么三个基本尺度分别是128，256和512。
    然后在这三个基本尺度上才有3个ratio。

    ratio是在确定了框的尺度基础上调整长宽比的参数，
    默认值为1：1和正负2：1，即为一个正方形和两个长方形

    其实，Faster-rcnn的重要思想就是在这个地方体现出来了，
    到底怎样进行目标检测？如何才能不漏下任何一个目标？
    那就是遍历的方法，不是遍历图片，而是遍历特征图，
    对一次提取的特征图进行遍历(3*3的卷积核挨个特征产生anchor)
    依次产生9个长宽比尺寸不同的anchor，力求将所有的在图中的目标都框住，
    产生完anchor之后再送入到9×2和9×4的Fc网络用来做分类和回归，
    对产生的anchor进行进一步的修正，这样几乎以极大的概率可以将图中所有的目标全部框住了！
    后续再进行一些处理，如非极大值抑制，抑制住重复框住的anchor，产生良好的可视效果！

    Generate anchors that are scaled and modified to the given aspect ratios.
    Area of a scaled anchor is preserved when modifying to the given aspect
    ratio.

    :obj:`R = len(ratios) * len(anchor_scales)` anchors are generated by this
    function.
    The :obj:`i * len(anchor_scales) + j` th anchor corresponds to an anchor
    generated by :obj:`ratios[i]` and :obj:`anchor_scales[j]`.

    For example, if the scale is :math:`8` and the ratio is :math:`0.25`,
    the width and the height of the base window will be stretched by :math:`8`.
    For modifying the anchor to the given aspect ratio,
    the height is halved and the width is doubled.

    Args:
        base_size (number): The width and the height of the reference window.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    Returns:
        ~numpy.ndarray:
        An array of shape :math:`(R, 4)`.
        Each element is a set of coordinates of a bounding box.
        The second axis corresponds to
        :math:`(y_{min}, x_{min}, y_{max}, x_{max})` of a bounding box.

    """
    py = base_size / 2.
    px = base_size / 2.
    #因为原图到feature map经过四次pooling层，
    #即feature map中点为原图的1/16，
    #中心点在中间，所以坐标为base size的半个单位，

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    #三种scale和三种ratios共产生9种不同的框

    for i in six.moves.range(len(ratios)):  #six.moves 是用来处理那些在python2 和 3里面函数的位置有变化的，直接用six.moves就可以屏蔽掉这些变化
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j      #索引值
            anchor_base[index, 0] = py - h / 2.     #计算两角坐标
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.

    #返回的为单个feature中元素对应的9种不同框相对于中心点的偏移量
    #而所有框的生成在rpn类中的_enumerate_shifted_anchor函数
    return anchor_base


'''
这9个anchor形状应为：
1：2：
90.50967 *181.01933    = 128^2
181.01933 * 362.03867 = 256^2
362.03867 * 724.07733 = 512^2

1：1：
128.0 * 128.0 = 128^2
256.0 * 256.0 = 256^2
512.0 * 512.0 = 512^2

2：1：
181.01933 * 90.50967   = 128^2
362.03867 * 181.01933 = 256^2
724.07733 * 362.03867 = 512^2

该函数返回值为anchor_base，形状9*4，是9个anchor的左上右下坐标：
-37.2548 -82.5097 53.2548 98.5097
-82.5097	-173.019	98.5097	189.019
-173.019	-354.039	189.019	370.039
-56	-56	72	72
-120	-120	136	136
-248	-248	264	264
-82.5097	-37.2548	98.5097	53.2548
-173.019	-82.5097	189.019	98.5097
-354.039	-173.019	370.039	189.019
'''
import torch,sys
from torch import nn
from torchvision import models
import numpy as np
from d2l import torch as d2l

# d2l = sys.modules[__name__]

def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel.
    size:原图长宽缩放比
    ratios:原图宽(w)高(h)比的ratios倍为锚框的宽(w')高(h')比

    w'*h'=w*h*s²
    w'/h'=w/h*r
    解得：
    w'=ws根号r h'=hs/根号r

    锚框数量：
    选取s1和不同r的组合+选取r1和不同s的组合
    """
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5  # 单位像素大小是1x1，所以中心是0.5x0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis 每个单位像素归一化的步长
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    '''
    torch.meshgrid函数:以a为行，以b为列
    a=[1,2,3] b=[4,5,6]
    x,y=torch.meshgrid(a,b)
    x=[[1,1,1],
       [2,2,2],
       [3,3,3]]
    y=[[4,5,6],
       [4,5,6],
       [4,5,6]]
    '''
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h # 每个像素归一化后的中心点坐标
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    # 得到特征图的所有中心点坐标
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack(
        (-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()
        model_conv = models.vgg16(pretrained=True)
        model_conv = nn.Sequential(*list(model_conv.children())[:-2]) # 不要avgpool 和classifier(包含3个linear+ReLU)
        self.cnn = model_conv

    def forward(self, img):
        return self.cnn(img)

class PredictionConvolutions(nn.Module):
    def __init__(self, n_classes):
        super(PredictionConvolutions, self).__init__()
        self.n_classes = n_classes
        n_boxes = 5
        self.loc_conv = nn.Conv2d(512, n_boxes * 4, kernel_size=3, padding=1)
        self.cl_conv = nn.Conv2d(512, n_boxes * n_classes, kernel_size=3, padding=1)
        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, pool5_feats):
        batch_size = pool5_feats.size(0)
        l_conv = self.loc_conv(pool5_feats)
        l_conv = l_conv.permute(0, 2, 3, 1).contiguous()
        locs = l_conv.view(batch_size, -1, 4)
        c_conv = self.cl_conv(pool5_feats)
        c_conv = c_conv.permute(0, 2, 3, 1).contiguous()
        classes_scores = c_conv.view(batch_size, -1, self.n_classes)
        return locs, classes_scores

class SSD(nn.Module):
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.base = VGGBase()
        self.pred_convs = PredictionConvolutions(num_classes)
        self.sizes =[0.75, 0.5, 0.25] # 原图缩放比
        self.ratios = [1, 2, 0.5] # 框的高宽比

    def forward(self, image):
        image = self.base(image) #[b, 3, 224, 224] -> [b, 512, 7, 7]
        anchors = multibox_prior(image, self.sizes, self.ratios)
        locs, classes_scores = self.pred_convs(image)
        locs = locs.reshape(locs.shape[0], -1)
        return anchors, locs, classes_scores

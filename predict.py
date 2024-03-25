from model import SSD
import torch
from dataset import VOCDataset
from torch.utils.data import DataLoader
import numpy as np
from d2l import torch as dltools
from matplotlib import pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm

### 预测
model_predict = SSD(21)
model_predict.load_state_dict(torch.load('./model.pth'))
model_predict = model_predict.cuda()


def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""
    anc = dltools.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = dltools.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = dltools.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = dltools.box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return dltools.tensor(keep, device=boxes.device)


def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = dltools.bbox_to_rect(dltools.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center',
                      ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
            


def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat(
            (class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    return dltools.stack(out)


            

def predict(image, model):
    model.eval()
    anchors, bbox_preds, cls_preds = model(image.cuda())
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row  in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

def display(image, output,i, threshold):
    fig = plt.imshow(image.permute(1, 2, 0).numpy()[:, :, ::-1])
    for row in output:
        score = float(row[1])
        predict_label = int(row[0])
        score_class = classes[predict_label] + ':' + str(score)
        if score < threshold:
            continue
        bbox = [row[2:6] * torch.tensor((224, 224, 224, 224), device=row.device)]
        # print(bbox)
        show_bboxes(fig.axes, bbox, score_class, 'w')
    plt.savefig('predict/'+str(i)+'.png')
    plt.close()


classes = ['person','bird', 'cat', 'cow', 'dog', 'horse', 'sheep','aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']


val = VOCDataset(is_train=False)
val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=0,pin_memory=True,collate_fn=val.collate_fn)


for i,(image, label) in enumerate(tqdm(val_loader)):

# image, label = next(iter(val_loader))
    output = predict(image[0].unsqueeze(0), model_predict)
    display(image[0], output.cpu(),i, threshold=0.1)


# print(label[0][:, 1:] * torch.tensor([224, 224, 224, 224]))
# fig = plt.imshow(image[0].permute(1, 2, 0).numpy()[:, :, ::-1])
# # show_bboxes(fig.axes, label[0] * torch.tensor([224]), [1, 1, 1], 'w')
# true_label = [classes[int(i)] for i in label[0][:, 0]]
# show_bboxes(fig.axes, label[0][:, 1:] * torch.tensor((224, 224, 224, 224)), true_label)

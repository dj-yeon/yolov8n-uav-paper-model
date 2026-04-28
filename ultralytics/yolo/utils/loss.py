# Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import bbox_iou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") *
                    weight).sum()
        return loss


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False, box_loss="ciou"):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.box_loss = box_loss
        self.register_buffer("iou_mean", torch.tensor(1.0))

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        if self.box_loss == "wise_iou":
            loss_iou = (self._wise_iou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask]) * weight).sum() / \
                       target_scores_sum
        else:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    def _wise_iou_loss(self, pred_bboxes, target_bboxes, eps=1e-7, momentum=0.01, alpha=1.9, delta=3.0):
        # Wise-IoU v3: distance attention plus dynamic non-monotonic focusing.
        px1, py1, px2, py2 = pred_bboxes.chunk(4, -1)
        tx1, ty1, tx2, ty2 = target_bboxes.chunk(4, -1)

        inter = (px2.minimum(tx2) - px1.maximum(tx1)).clamp(0) * \
                (py2.minimum(ty2) - py1.maximum(ty1)).clamp(0)
        pred_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
        target_area = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
        iou = inter / (pred_area + target_area - inter + eps)
        iou_loss = 1.0 - iou

        cw = px2.maximum(tx2) - px1.minimum(tx1)
        ch = py2.maximum(ty2) - py1.minimum(ty1)
        c2 = cw.pow(2) + ch.pow(2) + eps
        rho2 = ((tx1 + tx2 - px1 - px2).pow(2) + (ty1 + ty2 - py1 - py2).pow(2)) / 4
        distance_attention = torch.exp((rho2 / c2).detach())

        if self.training:
            self.iou_mean.mul_(1 - momentum).add_(iou_loss.detach().mean() * momentum)
        beta = iou_loss.detach() / self.iou_mean.clamp(min=eps)
        focusing = beta / (delta * torch.pow(alpha, beta - delta) + eps)

        return focusing * distance_attention * iou_loss

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)

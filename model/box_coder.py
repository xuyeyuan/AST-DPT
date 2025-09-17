import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ["pointCoder", "pointwhCoder"]
from timm.models.layers import to_2tuple
class pointCoder(nn.Module):
    def __init__(self, input_size, patch_count, weights=(1., 1.), tanh=True):
        super().__init__()
        self.input_size = to_2tuple(input_size) if isinstance(input_size, int) else input_size
        self.patch_count = to_2tuple(patch_count) if isinstance(patch_count, int) else patch_count
        self.weights = weights
        self._generate_anchor()
        self.tanh = tanh

    def _generate_anchor(self):
        anchors = []
        patch_stride_h = 1. / self.patch_count[0]
        patch_stride_w = 1. / self.patch_count[1]
        for i in range(self.patch_count[0]):
            for j in range(self.patch_count[1]):
                y = (0.5+i)*patch_stride_h
                x = (0.5+j)*patch_stride_w
                anchors.append([x, y])
        anchors = torch.as_tensor(anchors)
        self.register_buffer("anchor", anchors)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, pts, model_offset=None):
        assert model_offset is None
        self.boxes = self.decode(pts)
        return self.boxes

    def decode(self, rel_codes):
        boxes = self.anchor
        pixel_h = 1. / self.patch_count[0]
        pixel_w = 1. / self.patch_count[1]
        wx, wy = self.weights

        dx = F.tanh(rel_codes[:, :, 0] / wx) * pixel_w if self.tanh else rel_codes[:, :, 0] * pixel_w / wx
        dy = F.tanh(rel_codes[:, :, 1] / wy) * pixel_h if self.tanh else rel_codes[:, :, 1] * pixel_h / wy

        pred_boxes = torch.zeros_like(rel_codes)

        ref_x = boxes[:, 0].unsqueeze(0)
        ref_y = boxes[:, 1].unsqueeze(0)

        pred_boxes[:, :, 0] = dx + ref_x
        pred_boxes[:, :, 1] = dy + ref_y
        pred_boxes = pred_boxes.clamp_(min=0., max=1.)

        return pred_boxes

    def get_offsets(self):
        return (self.boxes - self.anchor) * torch.tensor([self.input_size[1], self.input_size[0]],
                                                         device=self.boxes.device)
class pointwhCoder(pointCoder):
    def __init__(self, input_size, patch_count, weights=(1., 1.), pts=1, tanh=True, wh_bias=None):
        super().__init__(input_size=input_size, patch_count=patch_count, weights=weights, tanh=tanh)
        self.patch_pixel = pts
        self.wh_bias = None
        if wh_bias is not None:
            self.wh_bias = nn.Parameter(torch.zeros(2) + wh_bias)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, pts, model_offset=None):
        assert model_offset is None
        if self.wh_bias is not None:
            pts[:, :, 2:] = pts[:, :, 2:] + self.wh_bias
        self.boxes = self.decode(pts)
        points = self.meshgrid(self.boxes)
        return points

    def decode(self, rel_codes):
        boxes = self.anchor
        pixel_h = 1. / self.patch_count[0]
        pixel_w = 1. / self.patch_count[1]
        wx, wy, wh, ww = self.weights

        dx = F.tanh(rel_codes[:, :, 0] / wx) * pixel_w if self.tanh else rel_codes[:, :, 0] * pixel_w / wx
        dy = F.tanh(rel_codes[:, :, 1] / wy) * pixel_h if self.tanh else rel_codes[:, :, 1] * pixel_h / wy

        dw = F.relu(F.tanh(rel_codes[:, :, 2] / ww)) * pixel_w
        dh = F.relu(F.tanh(rel_codes[:, :, 3] / wh)) * pixel_h

        pred_boxes = torch.zeros_like(rel_codes)

        ref_x = boxes[:, 0].unsqueeze(0)
        ref_y = boxes[:, 1].unsqueeze(0)

        pred_boxes[:, :, 0] = dx + ref_x - dw
        pred_boxes[:, :, 1] = dy + ref_y - dh
        pred_boxes[:, :, 2] = dx + ref_x + dw
        pred_boxes[:, :, 3] = dy + ref_y + dh
        pred_boxes = pred_boxes.clamp_(min=0., max=1.)
        return pred_boxes

    def get_offsets(self):
        return (self.boxes - self.anchor.repeat(1, 2)) * torch.tensor([self.input_size[1], self.input_size[0],
                                                                       self.input_size[1], self.input_size[0]],
                                                                      device=self.boxes.device)

    def get_scales(self):
        return (self.boxes[:, :, 2:] - self.boxes[:, :, :2]) * torch.tensor([self.input_size[1], self.input_size[0]],
                                                                            device=self.boxes.device)

    def meshgrid(self, boxes):
        B = boxes.shape[0]
        xs, ys = boxes[:, :, 0::2], boxes[:, :, 1::2]
        xs = torch.nn.functional.interpolate(xs, size=self.patch_pixel, mode='linear', align_corners=True)
        ys = torch.nn.functional.interpolate(ys, size=self.patch_pixel, mode='linear', align_corners=True)
        xs, ys = xs.unsqueeze(3).repeat_interleave(self.patch_pixel, dim=3), ys.unsqueeze(2).repeat_interleave(
            self.patch_pixel, dim=2)
        results = torch.stack([xs, ys], dim=-1)
        results = results.reshape(B, self.patch_count[0] * self.patch_count[1] * self.patch_pixel * self.patch_pixel, 2)
        return results
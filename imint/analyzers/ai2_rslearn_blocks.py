"""Plain torch/torchvision building blocks of AI2's rslearn vessel models.

Shared by the Sentinel-2 port (``imint/analyzers/ai2_vessels.py``) and the
Sentinel-1 port (``imint/analyzers/ai2_vessels_s1.py``). Each block mirrors
the rslearn module named in its docstring so that rslearn checkpoints load
into it with ``strict=True`` after a prefix remap; a change here changes both
analyzers.
"""

from __future__ import annotations

import collections

import numpy as np
import torch
import torchvision
from torchvision.models.detection import rpn as _rpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign

# rslearn Swin(output_layers=[1, 3, 5, 7]): strides 4/8/16/32.
SWIN_OUTPUT_LAYERS = (1, 3, 5, 7)
# rslearn FasterRCNN anchor_sizes in both vessel configs.
ANCHOR_SIZES = [[32], [64], [128], [256]]


class SwinFeatureExtractor(torch.nn.Module):
    """``torchvision.models.swin_v2_b`` with an ``input_channels`` patch-embed stem.

    Matches ``rslearn.models.swin.Swin``: iterates ``model.features`` and
    returns the outputs of ``output_layers``, permuted NHWC -> NCHW. The final
    ``norm`` and classification ``head`` exist (checkpoints carry them) but
    are not used.
    """

    def __init__(self, input_channels: int = 9,
                 output_layers: tuple[int, ...] = SWIN_OUTPUT_LAYERS) -> None:
        super().__init__()
        self.model = torchvision.models.swin_v2_b()
        if input_channels != 3:
            self.model.features[0][0] = torch.nn.Conv2d(
                input_channels,
                self.model.features[0][0].out_channels,
                kernel_size=(4, 4),
                stride=(4, 4),
            )
        self.output_layers = output_layers

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """(B, C, H, W) -> one (B, C_i, H_i, W_i) map per entry of ``output_layers``."""
        layer_features: list[torch.Tensor] = []
        for layer in self.model.features:
            x = layer(x)
            layer_features.append(x.permute(0, 3, 1, 2))
        return [layer_features[i] for i in self.output_layers]


class FPN(torch.nn.Module):
    """``torchvision.ops.FeaturePyramidNetwork`` without extra blocks (rslearn ``Fpn``)."""

    def __init__(self, in_channels: list[int], out_channels: int = 128) -> None:
        super().__init__()
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels,
                                         out_channels=out_channels)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        inp = collections.OrderedDict((f"feat{i}", f) for i, f in enumerate(features))
        return list(self.fpn(inp).values())


class NoopTransform(torch.nn.Module):
    """Matches ``rslearn.models.faster_rcnn.NoopTransform``: pad to /32, no resize."""

    def __init__(self) -> None:
        super().__init__()
        self.transform = GeneralizedRCNNTransform(
            min_size=800, max_size=800, image_mean=[], image_std=[])

    def forward(self, images, targets):
        images = self.transform.batch_images(images, size_divisible=32)
        image_sizes = [(img.shape[1], img.shape[2]) for img in images]
        return ImageList(images, image_sizes), targets


class FasterRCNNHead(torch.nn.Module):
    """RPN + RoI heads of ``rslearn.models.faster_rcnn.FasterRCNN`` (inference)."""

    def __init__(
        self,
        num_channels: int = 128,
        num_classes: int = 2,
        anchor_sizes: list[list[int]] | None = None,
        box_score_thresh: float = 0.05,
    ) -> None:
        super().__init__()
        if anchor_sizes is None:
            anchor_sizes = ANCHOR_SIZES
        n_scales = len(anchor_sizes)
        featmap_names = [f"feat{i}" for i in range(n_scales)]

        self.noop_transform = NoopTransform()

        rpn_anchor_generator = AnchorGenerator(anchor_sizes, ((0.5, 1.0, 2.0),) * n_scales)
        rpn_head = _rpn.RPNHead(num_channels,
                                rpn_anchor_generator.num_anchors_per_location()[0])
        self.rpn = _rpn.RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=2000),
            post_nms_top_n=dict(training=2000, testing=2000),
            nms_thresh=0.7,
        )

        box_roi_pool = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=7,
                                          sampling_ratio=2)
        box_head = TwoMLPHead(num_channels * box_roi_pool.output_size[0] ** 2, 1024)
        box_predictor = FastRCNNPredictor(1024, num_classes)
        self.roi_heads = RoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=box_score_thresh,
            nms_thresh=0.5,
            detections_per_img=100,
        )

    def forward(self, feature_maps: list[torch.Tensor],
                raw_images: list[torch.Tensor]) -> list[dict]:
        """FPN outputs + per-image tensors (for the image sizes) -> torchvision detections."""
        images, _ = self.noop_transform(raw_images, None)
        feature_dict = collections.OrderedDict(
            (f"feat{i}", fm) for i, fm in enumerate(feature_maps))
        proposals, _ = self.rpn(images, feature_dict, None)
        detections, _ = self.roi_heads(feature_dict, proposals, images.image_sizes, None)
        return detections


class PoolingDecoder(torch.nn.Module):
    """``rslearn.models.pooling_decoder.PoolingDecoder``.

    ``Conv(in -> conv_ch, 3x3) + ReLU`` (x num_conv_layers) -> global max pool
    -> ``Linear(prev -> fc_ch) + ReLU`` (x num_fc_layers) -> ``Linear(fc_ch -> out)``,
    on the last encoder feature map only.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_conv_layers: int = 0,
        num_fc_layers: int = 0,
        conv_channels: int = 128,
        fc_channels: int = 512,
    ) -> None:
        super().__init__()
        conv_layers = []
        prev_ch = in_channels
        for _ in range(num_conv_layers):
            conv_layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(prev_ch, conv_channels, 3, padding=1),
                torch.nn.ReLU(inplace=True)))
            prev_ch = conv_channels
        self.conv_layers = torch.nn.Sequential(*conv_layers)

        fc_layers = []
        for _ in range(num_fc_layers):
            fc_layers.append(torch.nn.Sequential(
                torch.nn.Linear(prev_ch, fc_channels),
                torch.nn.ReLU(inplace=True)))
            prev_ch = fc_channels
        self.fc_layers = torch.nn.Sequential(*fc_layers)

        self.output_layer = torch.nn.Linear(prev_ch, out_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, out_channels)."""
        x = self.conv_layers(features)
        x = torch.amax(x, dim=(2, 3))
        return self.output_layer(self.fc_layers(x))


def crop_centered(image: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    """(C, size, size) crop covering ``[cy - size//2, cy + size - size//2)`` rows
    (likewise columns) of a (C, H, W) array, zero-padded outside the image."""
    _, h, w = image.shape
    half = size // 2
    y0, x0 = cy - half, cx - half
    crop = np.zeros((image.shape[0], size, size), dtype=image.dtype)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(h, y0 + size), min(w, x0 + size)
    crop[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[:, sy0:sy1, sx0:sx1]
    return crop

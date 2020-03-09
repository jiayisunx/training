# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
    if os.environ.get('PROFILE') == "1":
        proposals = torch.tensor([[  82.5511,  646.1293,  325.9225,  744.2648],
            [  94.2009,  609.9208,  317.6983,  685.3846],
            [ 211.5207,  120.7487,  424.9951,  283.5504],
            [  70.7335,  596.4381,  342.1802,  642.3331],
            [  86.7681,  642.8627,  198.5215,  865.7788],
            [   0.0000,  735.7050,  169.9834,  846.5739],
            [ 276.1382,  661.9395,  571.9825,  729.1837],
            [   1.6147,  659.8272,  123.9205,  857.7624],
            [   0.0000,  931.3787,  154.2995, 1028.5431],
            [   0.0000,  690.2159,   90.5213,  803.3977],
            [ 197.8719,  115.0647,  313.2966,  312.4772],
            [ 101.1255,  664.3341,  386.0901,  925.5726],
            [  60.8799,  656.2911,  333.9336,  926.3220],
            [   5.0932,  649.5626,  331.4645,  864.6251],
            [ 205.5613,   59.2320,  473.2962,  389.3442],
            [ 196.8089,   35.5371,  485.6968,  309.3773],
            [   3.9977,  689.7617,  316.9275,  902.4243],
            [ 180.1806,  119.7364,  419.4564,  372.0585],
            [ 114.3640,  629.5065,  316.5851,  987.7375],
            [ 137.8055,   67.1483,  507.4560,  399.6963],
            [ 128.7643,  648.0876,  459.8578,  926.9767],
            [ 180.5812,  553.0164,  648.9570,  726.1486],
            [   0.0000,   90.3015,  129.3312,  681.2150],
            [  80.2707,  578.4922,  476.6324,  873.2165],
            [  87.2667,  645.8794,  379.5333,  906.7496],
            [  87.7333,  589.2648,  293.7000,  686.6539],
            [ 193.0667,  377.4140,  253.1000,  500.9534],
            [ 186.7333,   70.2727,  442.2333,  322.2984]])
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )

            if os.environ.get('PROFILE') == "1":
                matched_targets.extra_fields['labels'] = torch.tensor([ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  8,
                 8,  3,  8,  8,  3,  3,  3,  3,  3,  3,  3,  3,  3, 75,  3,  3,  3,  3,
                 3,  3,  3,  3,  3,  3, 75,  3,  3,  3,  3,  3, 75,  3,  3,  3,  3,  3,
                 3,  3,  3,  8, 12, 75])
                matched_targets.extra_fields['matched_idxs'] = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,
                 1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  3,  0,  0,  0,  0,
                -1,  0, -1, -1, -1,  0,  3, -1, -1, -1,  0, -1,  3, -1, -1, -1, -1, -1,
                -1, -1,  0,  1,  2,  3])
                matched_targets.bbox = torch.tensor([
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.7333, 589.2648, 293.7000, 686.6539],
                [ 87.7333, 589.2648, 293.7000, 686.6539],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.7333, 589.2648, 293.7000, 686.6539],
                [ 87.7333, 589.2648, 293.7000, 686.6539],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [186.7333,  70.2727, 442.2333, 322.2984],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [186.7333,  70.2727, 442.2333, 322.2984],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [186.7333,  70.2727, 442.2333, 322.2984],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.2667, 645.8794, 379.5333, 906.7496],
                [ 87.7333, 589.2648, 293.7000, 686.6539],
                [193.0667, 377.4140, 253.1000, 500.9534],
                [186.7333,  70.2727, 442.2333, 322.2984]])

            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            if os.environ.get('PROFILE') == "1":
                positive_inds = torch.tensor([11, 12, 14, 15, 17, 24, 25, 26, 27, 28, 29, 33, 34, 35, 37, 38, 40, 41, 42, 44, 45, 46, 53, 55, 56, 57, 58, 59])
            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals, self.discretization_size
            )

            labels.append(labels_per_image)
            masks.append(masks_per_image)

        return labels, masks

    def __call__(self, proposals, mask_logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, mask_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]

        if os.environ.get('PROFILE') == "1":
            A = torch.tensor([11, 12, 14, 15, 17, 24, 25, 26, 27, 28, 29, 33, 34, 35, 37, 38, 40, 41,
                 42, 44, 45, 46, 53, 55, 56, 57, 58, 59])
            B = torch.tensor([ 8,  8, 75,  8,  8,  3,  3,  3,  3,  3,  3, 75, 75,  3,  8,  3,  3,  3,
                 3, 75,  3, 75,  3, 75,  3,  8, 12, 75])
            positive_inds = A
            labels_pos = B
            for i in range(len(proposals)-1):
                positive_inds = torch.cat((positive_inds, A), 0)
                labels_pos = torch.cat((labels_pos, B), 0)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_logits.is_mkldnn:
            mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits.to_dense()[positive_inds, labels_pos], mask_targets
            )
        else:
            mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits[positive_inds, labels_pos], mask_targets
            )
        return mask_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
    )

    return loss_evaluator

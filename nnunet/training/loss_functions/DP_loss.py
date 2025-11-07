import torch
from torch.nn.modules.loss import _Loss
from typing import Callable, List, Optional, Sequence, Union
from monai.losses import DiceLoss
import torch.nn as nn
from nnunet.utilities.tensor_utilities import sum_tensor
import numpy as np
from nnunet.training.loss_functions.dice_loss import AsymLoss
from torch.nn.functional import softmax
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss

def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def NormalizeData(data):
        norm = (data - torch.min(data)) / (torch.max(data) + 1e-8 - torch.min(data))
        return norm


class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)

        return super(CrossentropyND, self).forward(inp, target)


class DPCE_loss(torch.nn.Module):
    """
    DPCE loss version 2
    Overlay dist map (dist) on onehot-encoded target (y_onehot) to get dist_y and calculate CELoss
    """
    def forward(self, net_output, target, dist):
        # net_output = softmax_helper(net_output)
        
        with torch.no_grad():
            target = target.long()
            # one hot code for target
            y_onehot = torch.zeros(net_output.shape)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, target, 1)
            # dist to cuda
            dist = torch.from_numpy(dist)
            if dist.device != net_output.device:
                dist = dist.to(net_output.device).type(torch.float32)
        # add dist map to each class
        dist_y = torch.zeros(y_onehot.shape)
        for j in range(dist_y.size(0)): # j: batch size
            for i in range(dist_y.size(1)): # i: num of class
                if i == 0:
                    dist_y[j,i,...] = y_onehot[j,i,...]
                else:
                    dist_y[j,i,...] = y_onehot[j,i,...]*dist[j,...]
                    dist_y[j,i,...] = NormalizeData(dist_y[j,i,...]) # normalize for each organ and add 1e-8 to avoid vanish gradient
        # some images are missing some label (e.g., class=3) in data_dict but cannot figure out why, so I added 1e-8 in normalization to 
        # avoid being divided by 0

        if dist_y.device != net_output.device:
                dist_y = dist_y.to(net_output.device).type(torch.float32)

        ce = nn.CrossEntropyLoss()
        dpce = ce(net_output, dist_y)

        return dpce


class DicePenalizedCELoss(_Loss):
    """
    Compute both Dice loss and Penalized Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss``. In this implementation,
    two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        """
        Args:
            ``ce_weight`` and ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example: `other_act = torch.tanh`.
                only used by the `DiceLoss`, don't need to specify activation function for `CrossEntropyLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            ce_weight: a rescaling weight given to each class for cross entropy loss.
                See ``torch.nn.CrossEntropyLoss()`` for more information.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        # self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def penalize_ce(self, inp: torch.Tensor, target: torch.Tensor, dist):
        """
        Compute Penalized CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        [add reference here later]

        """
        dist = torch.from_numpy(dist)
        if dist.device != inp.device:
            dist = dist.to(inp.device).type(torch.float32)
        dist = dist.view(-1,)

        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)
        log_sm = torch.nn.LogSoftmax(dim=1)
        inp_logs = log_sm(inp)

        target = target.view(-1,)
        # loss = nll_loss(inp_logs, target)
        loss = -inp_logs[range(target.shape[0]), target]
        # print(loss.type(), dist.type())
        weighted_loss = loss*dist
        weighted_loss = weighted_loss.mean()

        return weighted_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, dist) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")

        dice_loss = self.dice(input, target)
        penalize_ce_loss = self.penalize_ce(input, target, dist)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * penalize_ce_loss

        return total_loss

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

class DC_and_DPCE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_DPCE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.dpce = DisPenalizedCE_July23()

        self.ignore_label = ignore_label

        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)


    def forward(self, net_output, target, dist):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        dpce_loss = self.dpce(net_output, target, dist) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            dpce_loss *= mask[:, 0]
            dpce_loss = dpce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * dpce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class DisPenalizedCE(torch.nn.Module):
    """
    For binary and multi-class 3D segmentation
    Network has to have NO NONLINEARITY!
    """

    def forward(self, inp, target, dist):
        # print(inp.shape, target.shape) # (batch, 2, xyz), (batch, 2, xyz)
        # compute distance map of ground truth
        
        # don't need add one for the new dist map (max=8, min=1, padded=0)
        # dist = dist + 1
        if dist.device != inp.device:
            dist = dist.to(inp.device).type(torch.float32)
        dist = dist.view(-1,)

        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)
        log_sm = torch.nn.LogSoftmax(dim=1)
        inp_logs = log_sm(inp)

        target = target.view(-1,)
        # loss = nll_loss(inp_logs, target)
        loss = -inp_logs[range(target.shape[0]), target]
        # print(loss.type(), dist.type())
        weighted_loss = loss*dist

        return weighted_loss.mean()

'''GPT implementation with higher efficiency'''
import torch
import torch.nn.functional as F

class DisPenalizedCE_July23(torch.nn.Module):
    """
    For binary and multi-class 3D segmentation
    Network has to have NO NONLINEARITY!
    """

    def forward(self, inp, target, dist):
        # Ensure dist is on the same device as inp
        if dist.device != inp.device:
            dist = dist.to(inp.device).float()

        # Flatten dist
        dist = dist.view(-1,)

        # Prepare target tensor
        target = target.long().view(-1)

        # Prepare input tensor by moving channel dimension to the end
        inp = inp.moveaxis(1, -1).contiguous()

        # Compute log softmax along channel dimension
        inp_logs = F.log_softmax(inp.view(-1, inp.shape[-1]), dim=1)

        # Compute the loss
        loss = -inp_logs[range(target.shape[0]), target]

        # Apply distance weights
        weighted_loss = loss * dist

        return weighted_loss.mean()



'''Implementation from ChatGPT'''
import torch
import torch.nn.functional as F
import torch.nn as nn

class DistancePenalizedCrossEntropy_GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, dist):
        targets = targets.long()
        num_classes = inputs.size()[1]

        # This part rearranges dimensions such that class dimension comes last
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        
        # Flattening the tensor for applying cross entropy loss
        inputs = inputs.view(-1, num_classes)
        inputs = self.logsoftmax(inputs)

        targets = targets.view(-1)  # Flatten the target tensor
        dist = dist.view(-1)  # Flatten the distance tensor

        # For the device consistency
        if dist.device != inputs.device:
            dist = dist.to(inputs.device).type(torch.float32)

        # Compute loss, weight by dist
        loss = -inputs[range(targets.shape[0]), targets]  # Cross entropy
        weighted_loss = loss * dist  # Weight by distance

        return weighted_loss.mean()

class Weighted_DisPenalizedCE(nn.Module):
    def forward(self, inp, target, dist, weights):
        # Ensure dist is on the correct device and the correct type
        if dist.device != inp.device:
            dist = dist.to(inp.device).type(torch.float32)
        dist = dist.view(-1,)

        # Ensure weights are on the correct device
        if weights.device != inp.device:
            weights = weights.to(inp.device).type(torch.float32)

        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        log_sm = torch.nn.LogSoftmax(dim=1)
        inp_logs = log_sm(inp)

        target = target.view(-1,)
        loss = -inp_logs[range(target.shape[0]), target]
        weighted_loss = loss * dist * weights[target]  # Apply class weights

        return weighted_loss.mean()

class Weighted_DistancePenalizedCrossEntropy_GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, dist, weights):
        targets = targets.long()
        num_classes = inputs.size()[1]

        # This part rearranges dimensions such that class dimension comes last
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()

        # Flattening the tensor for applying cross entropy loss
        inputs = inputs.view(-1, num_classes)
        inputs = self.logsoftmax(inputs)

        targets = targets.view(-1)  # Flatten the target tensor
        dist = dist.view(-1)  # Flatten the distance tensor

        # For the device consistency
        if dist.device != inputs.device:
            dist = dist.to(inputs.device).type(torch.float32)

        # Make sure weights are on the right device
        if weights.device != inputs.device:
            weights = weights.to(inputs.device).type(torch.float32)

        # Compute loss, weight by dist
        loss = -inputs[range(targets.shape[0]), targets]  # Cross entropy
        weighted_loss = loss * dist * weights[targets]  # Weight by distance and class weights

        return weighted_loss.mean()



class DistBinaryDiceLoss(nn.Module):
    """
    Distance map penalized Dice loss
    Motivated by: https://openreview.net/forum?id=B1eIcvS45V
    Distance Map Loss Penalty Term for Semantic Segmentation        
    """
    def __init__(self, smooth=1e-5):
        super(DistBinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, net_output, gt, dist):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # one hot code for gt
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
        
        # don't need add one for the new dist map (max=8, min=1, padded=0)
        # dist = dist + 1 

        if dist.device != net_output.device:
            dist = dist.to(net_output.device).type(torch.float32)
        
        tp = net_output * y_onehot
        tp = torch.sum(tp[:,1,...] * dist, (1,2,3))
        
        dc = (2 * tp + self.smooth) / (torch.sum(net_output[:,1,...], (1,2,3)) + torch.sum(y_onehot[:,1,...], (1,2,3)) + self.smooth)

        dc = dc.mean()

        return -dc


class DistMultiDiceLoss_OLD(nn.Module):
    """
    Multi-class segmentation: Distance map penalized Dice loss
    Motivated by: https://openreview.net/forum?id=B1eIcvS45V
    Distance Map Loss Penalty Term for Semantic Segmentation        
    """
    def __init__(self, smooth=1e-5):
        super(DistMultiDiceLoss_OLD, self).__init__()
        self.smooth = smooth

    def forward(self, net_output, gt, dist):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        shp_out = net_output.shape
        net_output = softmax_helper(net_output)
        axes = list(range(2, len(shp_out)))
        # one hot code for gt
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
        
        # don't need add one for the new dist map (max=8, min=1, padded=0)
        # dist = dist + 1 

        if dist.device != net_output.device:
            dist = dist.to(net_output.device).type(torch.float32)
        
        # axes = list(range(2, len(shp_out)))
        tp = net_output * y_onehot
        fp = net_output * (1 - y_onehot)
        fn = (1 - net_output) * y_onehot
        for i in range(tp.shape[1]):
            tp[:,i,...] = tp[:,i,...]*dist
        
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)
        dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

class DistMultiClassDiceLoss(nn.Module):
    """
    Distance map penalized Dice loss
    Motivated by: https://openreview.net/forum?id=B1eIcvS45V
    Distance Map Loss Penalty Term for Semantic Segmentation        
    """
    def __init__(self, num_classes, smooth=1e-5):
        super(DistMultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, net_output, gt, dist):
        """
        net_output: (batch_size, num_classes, x,y,z)
        num_classes should include background!
        target: ground truth, shape: (batch_size, 1, x,y,z)
        dist: distance map, shape: (batch_size, 1, x,y,z)
        """
        net_output = softmax_helper(net_output)

        # One hot encoding
        gt = gt.long()
        y_onehot = torch.zeros_like(net_output)
        y_onehot.scatter_(1, gt, 1)

        if dist.device != net_output.device:
            dist = dist.to(net_output.device).type(torch.float32)

        total_loss = 0
        for class_i in range(self.num_classes):
            tp = net_output[:, class_i, ...] * y_onehot[:, class_i, ...]
            tp = torch.sum(tp * dist, (1,2,3))

            dc = (2 * tp + self.smooth) / (torch.sum(net_output[:, class_i, ...], (1,2,3)) + torch.sum(y_onehot[:, class_i, ...], (1,2,3)) + self.smooth)

            dc = dc.mean()
            total_loss += dc

        return -total_loss / self.num_classes

class DistMultiClassDiceLoss_v2(nn.Module):
    """
    Distance map penalized Dice loss
    Motivated by: https://openreview.net/forum?id=B1eIcvS45V
    Distance Map Loss Penalty Term for Semantic Segmentation        
    """
    def __init__(self, num_classes, smooth=1e-5):
        super(DistMultiClassDiceLoss_v2, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, net_output, gt, dist):
        """
        net_output: (batch_size, num_classes, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        dist: distance map, shape: (batch_size, 1, x,y,z)
        """
        net_output = softmax_helper(net_output)

        # One hot encoding
        gt = gt.long()
        y_onehot = torch.zeros_like(net_output)
        y_onehot.scatter_(1, gt, 1)

        if dist.device != net_output.device:
            dist = dist.to(net_output.device).type(torch.float32)

        total_loss = 0
        for class_i in range(self.num_classes):
            tp = net_output[:, class_i, ...] * y_onehot[:, class_i, ...]
            tp = torch.sum(tp * dist, (1,2,3))

            dc = (2 * tp + self.smooth) / (torch.sum(net_output[:, class_i, ...], (1,2,3)) + torch.sum(y_onehot[:, class_i, ...], (1,2,3)) + self.smooth)

            dc = dc.mean()
            total_loss += dc

        return -total_loss



class DPDC_and_CE_loss(nn.Module):
    def __init__(self, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DPDC_and_CE_loss, self).__init__()
        # if ignore_label is not None:
        #     assert not square_dice, 'not implemented'
        #     ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss()

        self.ignore_label = ignore_label

        self.dpdc = DistMultiClassDiceLoss(num_classes=5, smooth=1e-5)

    def forward(self, net_output, target, dist):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dpdc_loss = self.dpdc(net_output, target, dist)

        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dpdc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class DPDC_and_CE_loss_v2(nn.Module):
    def __init__(self, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DPDC_and_CE_loss_v2, self).__init__()
        # if ignore_label is not None:
        #     assert not square_dice, 'not implemented'
        #     ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss()

        self.ignore_label = ignore_label

        self.dpdc = DistMultiClassDiceLoss_v2(num_classes=4, smooth=1e-5)

    def forward(self, net_output, target, dist):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dpdc_loss = self.dpdc(net_output, target, dist)

        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dpdc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class DPDC_and_DPCE_loss(nn.Module):
    def __init__(self, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DPDC_and_DPCE_loss, self).__init__()
        # if ignore_label is not None:
        #     assert not square_dice, 'not implemented'
        #     ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.dpce = DisPenalizedCE()

        self.ignore_label = ignore_label

        self.dpdc = DistMultiClassDiceLoss(num_classes= 4, smooth=1e-5)

    def forward(self, net_output, target, dist):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dpdc_loss = self.dpdc(net_output, target, dist)

        dpce_loss = self.dpce(net_output, target, dist) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            dpce_loss *= mask[:, 0]
            dpce_loss = dpce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * dpce_loss + self.weight_dice * dpdc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class Asym_and_DPCE_loss(nn.Module):
    def __init__(self, aggregate="sum", square_dice=False, weight_ce=1, weight_asym=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Asym_and_DPCE_loss, self).__init__()
        # if ignore_label is not None:
        #     assert not square_dice, 'not implemented'
        #     ce_kwargs['reduction'] = 'none'
        self.weight_asym = weight_asym
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.dpce = DisPenalizedCE()

        self.ignore_label = ignore_label

        self.asym = AsymLoss(apply_nonlin=softmax_helper, smooth=1e-5, do_bg=False)

    def forward(self, net_output, target, dist):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        asym_loss = self.asym(net_output, target)

        dpce_loss = self.dpce(net_output, target, dist) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            dpce_loss *= mask[:, 0]
            dpce_loss = dpce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * dpce_loss + self.weight_asym * asym_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result
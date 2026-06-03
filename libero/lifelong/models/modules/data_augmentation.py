import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


from robomimic.models.base_nets import CropRandomizer


class IdentityAug(nn.Module):
    def __init__(self, input_shape=None, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

    def output_shape(self, input_shape):
        return input_shape


class TranslationAug(nn.Module):
    """
    Utilize the random crop from robomimic.
    """

    def __init__(
        self,
        input_shape,
        translation,
    ):
        super().__init__()

        self.pad_translation = translation // 2
        pad_output_shape = (
            input_shape[0],
            input_shape[1] + translation,
            input_shape[2] + translation,
        )

        self.crop_randomizer = CropRandomizer(
            input_shape=pad_output_shape,
            crop_height=input_shape[1],
            crop_width=input_shape[2],
        )

    def forward(self, x):
        if self.training:
            batch_size, temporal_len, img_c, img_h, img_w = x.shape
            x = x.reshape(batch_size, temporal_len * img_c, img_h, img_w)
            out = F.pad(x, pad=(self.pad_translation,) * 4, mode="replicate")
            out = self.crop_randomizer.forward_in(out)
            out = out.reshape(batch_size, temporal_len, img_c, img_h, img_w)
        else:
            out = x
        return out

    def output_shape(self, input_shape):
        return input_shape


class ImgColorJitterAug(torch.nn.Module):
    """
    Conduct color jittering augmentation outside of proposal boxes
    """

    def __init__(
        self,
        input_shape,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.05,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon

    def forward(self, x):
        if self.training and np.random.rand() > self.epsilon:
            out = self.color_jitter(x)
        else:
            out = x
        return out

    def output_shape(self, input_shape):
        return input_shape


class ImgColorJitterGroupAug(torch.nn.Module):
    """
    Conduct color jittering augmentation outside of proposal boxes
    """

    def __init__(
        self,
        input_shape,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.05,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon

    def forward(self, x):
        raise NotImplementedError
        if self.training and np.random.rand() > self.epsilon:
            out = self.color_jitter(x)
        else:
            out = x
        return out

    def output_shape(self, input_shape):
        return input_shape


class BatchWiseImgColorJitterAug(torch.nn.Module):
    """
    Color jittering augmentation to individual batch.
    This is to create variation in training data to combat
    BatchNorm in convolution network.
    """

    def __init__(
        self,
        input_shape,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.1,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon

    def forward(self, x):
        out = []
        for x_i in torch.split(x, 1):
            if self.training and np.random.rand() > self.epsilon:
                out.append(self.color_jitter(x_i))
            else:
                out.append(x_i)
        return torch.cat(out, dim=0)

    def output_shape(self, input_shape):
        return input_shape


class DataAugGroup(nn.Module):
    """
    Add augmentation to multiple inputs
    """

    def __init__(self, aug_list):
        super().__init__()
        self.aug_layer = nn.Sequential(*aug_list)

    def forward(self, x_groups):
        """
        x_groups: list/tuple of tensors, each of shape (B, T, C_i, H, W),
                where C_i is typically 3 (RGB) or 1 (depth/grayscale).
        Returns a tuple of the same shape, with augmentations applied per group.
        """
        if self.training:
            out_groups = []
            for x in x_groups:
                B, T, C, H, W = x.shape
                # 1) flatten time into batch
                x_flat = x.view(B * T, C, H, W)        # (B*T, C, H, W)
                # 2) augment each slice
                x_aug_flat = self.aug_layer(x_flat)    # (B*T, C, H, W) with C=3 or 1
                # 3) reshape back to (B, T, C, H, W)
                x_aug = x_aug_flat.view(B, T, C, H, W)
                out_groups.append(x_aug)
            return tuple(out_groups)
        else:
            # eval 模式不变
            return x_groups
    
class RandomErasingAug(nn.Module):
    def __init__(self, input_shape, scale=(0.02, 0.33), ratio=(0.3, 3.3), prob=0.1):
        super().__init__()
        self.transform = T.RandomErasing(p=prob, scale=scale, ratio=ratio)

    def forward(self, x):
        return self.transform(x)

class RandomAffineAug(nn.Module):
    def __init__(self, input_shape, degrees=15, translate=0.1, scale=None):
        super().__init__()
        self.transform = T.RandomAffine(
            degrees=degrees,
            translate=(translate, translate),
            scale=scale if scale else (1.0, 1.0)
        )

    def forward(self, x):
        return self.transform(x)
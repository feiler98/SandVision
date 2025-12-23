# imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
from PIL import Image
import numpy as np
from pathlib import Path
# ----------------------------------------------------------------------------------------------------------------------


# Image Segmentation Algorithm
########################################################################################################################
class DoubleConv(nn.Module):
    """
    Pipeline sub-element
    ####################

    Lateral double convolution for each encoder/decoder step like in UNET.
    Image size is not altered in this step, just kernel transformation, normalization and ReLU as activation function.
    Sequential transformation, sends image through the pipeline as listed below.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # function forwarding the non-linear transformation
    def forward(self, x):
        return self.conv(x)


class VisTransformer(nn.Module):
    """
    Main Class of the image segmentation pipeline of the project. Includes encoder & decoder structure of the image
    feature space. Output is a binary 2d array.
    """

    # the sequential horizontal compression / decompression of the feature space of the image
    feature_tuple = (64, 128, 256, 512)
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 features=feature_tuple):
        super(VisTransformer, self).__init__()
        self.ups = nn.ModuleList()  # holds submodules in a list; acts like a python list where different classes / functions for image transformation can be added
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # compress image to half size in both dims; take max value

        # DOWN-loop of VisTransformer
        #############################
        for feature in features:
            self.downs.append(
                DoubleConv(in_channels, feature)
            )
            in_channels = feature  # changes in channel size by current feature-dim

        # UP-loop of VisTransformer
        ###########################
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # lateral bottom transformation of feature matrix
        #################################################
        self.bottom_lateral = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    # function forwarding the encoder/decoder pipeline
    def forward(self, x):
        # saving the horizontal transformation of the image  --> used for connecting DOWN and UP laterally
        skip_connections = []

        # DOWN
        ######
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)  # self result of down DoubleConv
            x = self.pool(x)

        # Lateral bottom
        ################
        x = self.bottom_lateral(x)
        skip_connections = skip_connections[::-1]  # reverse skip_connections for UP-loop

        # UP
        ####
        """
        takes only every second item in ModuleList since:
        ConvTranspose2d
        DoubleConv --> target
        --> they are laternating
        --> input x in DoubleConv with reverse order than DOWN for image upsizing
        """
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection_slice = skip_connections[idx//2]

            # image size on up-scaling will always be an even number, but input could be uneven --> correct by rescale
            if x.shape != skip_connection_slice.shape:
                x = tf.resize(x, size=skip_connection_slice.shape[2:])  # get width and height of DOWN feature-space

            concat_skip = torch.cat((skip_connection_slice, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


# test
def test_uneven_img_size():
    x = torch.randn((3, 1, 161, 161))
    model = VisTransformer(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape  # raises ValueError if condition not met.


# Dataloader
########################################################################################################################
class DataLoader(Dataset):
    def __init__(self,
                 img_dir: (Path | str), mask_tag: str, data_suffix: str = ".png"):
        self.data_suffix = data_suffix
        self.img_dir = Path(img_dir)
        self.mask_tag = mask_tag
        self.image_tags = list(set([p.stem.split("__")[0] for p in Path(img_dir).rglob(f"*{data_suffix}")]))
        self.img_path_dict = {tag:img_dir/f"{tag}{data_suffix}" for tag in self.image_tags}
        self.mask_path_dict = {p.stem.split("__")[0]: p for p in Path(img_dir).rglob(f"*{mask_tag}{data_suffix}")}

    def __len__(self):
        return len(self.image_tags)

    def available_keys(self):
        return self.image_tags

    def __getitem__(self, dict_key):
        if dict_key not in self.mask_path_dict.keys():
            print(f"For key '{dict_key}' is the corresponding mask not available --> skip")
        else:
            img = np.array(Image.open(self.img_path_dict[dict_key]).convert("RGB"))  # 3 dim colorspace as tensor, not 4
            mask = np.array(Image.open(self.mask_path_dict[dict_key]).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0  # binary normalized mask
            return img, mask


# debugging
if __name__ == "__main__":
    test_uneven_img_size()









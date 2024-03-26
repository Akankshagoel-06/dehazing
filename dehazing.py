import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import mimetypes
from glob import glob
import itertools
import logging
from os.path import splitext
from os import listdir
from torch.utils.data import Dataset
from torchvision.models import resnet34
from torchvision.transforms import Compose

os.environ["TORCH_HOME"] = "C:/Users/akank/Downloads/dense_dataset"
path = "C:/Users/akank/Downloads/dense_dataset"
path_hr = path + "/clear"
path_lr = path + "/hazy"
class BasicDataset(Dataset):
    def __init__(self, imgsdr, masksdr, scale=1):
        self.imgs_dir = imgsdr
        self.masks_dir = masksdr
        self.scale = scale
        assert 0 < scale <= 1, "Scale must be between 0 and 1"

        self.ids = [
            splitext(file)[0] for file in listdir(imgsdr) if not file.startswith(".")
        ]
        # self.c = len(self.ids)

    #         logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, "Scale is too small"
        pil_img = pil_img.resize((128, 128))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = self.masks_dir + "/" + idx + ".png"
        img_file = self.imgs_dir + "/" + idx + ".png"

        mask = Image.open(mask_file)
        img = Image.open(img_file)

        assert (
            img.size == mask.size
        ), f"Image and mask {idx} should be the same size, but are {img.size} and {mask.size}"

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return (
            torch.from_numpy(img).type(torch.FloatTensor),
            torch.from_numpy(mask).type(torch.FloatTensor),
        )
dataset = BasicDataset(path_hr, path_lr, scale=0.5)
dataset.__getitem__(1)[0].permute(1, 2, 0).shape
plt.imshow(dataset.__getitem__(1)[0].permute(1, 2, 0))
plt.imshow(dataset.__getitem__(1)[1].permute(1, 2, 0))
def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        # nn.init.normal_(m.weight, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

class DownConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, initializers, padding, pool=True):
        super(DownConvBlock, self).__init__()
        layers = []

        if pool:
            layers.append(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
            )

        layers.append(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)
            )
        )
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)
            )
        )
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)
            )
        )
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, patch):
        return self.layers(patch)
class UpConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, initializers, padding, bilinear=True):
        super(UpConvBlock, self).__init__()
        self.bilinear = bilinear

        if not self.bilinear:
            self.upconv_layer = nn.ConvTranspose2d(
                input_dim, output_dim, kernel_size=2, stride=2
            )
            self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvBlock(
            input_dim, output_dim, initializers, padding, pool=False
        )

    def forward(self, x, bridge):
        if self.bilinear:
            up = nn.functional.interpolate(
                x, mode="bilinear", scale_factor=2, align_corners=True
            )
        else:
            up = self.upconv_layer(x)

        assert up.shape[3] == bridge.shape[3]
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out
class Unet(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        num_filters,
        initializers,
        apply_last_layer=True,
        padding=True,
    ):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()
        #         self.pl = nn.modules.pixelshuffle

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(
                DownConvBlock(input, output, initializers, padding, pool=pool)
            )

        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(
                UpConvBlock(input, output, initializers, padding)
            )

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output, num_classes, kernel_size=1)
            # nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_in',nonlinearity='relu')
            # nn.init.normal_(self.last_layer.bias)

    def forward(self, x, val):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i - 1])

        del blocks

        # Used for saving the activations and plotting
        #         if val:
        #             self.activation_maps.append(x)

        if self.apply_last_layer:
            x = self.last_layer(x)
        #         x = self.pl(x)

        return x
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook, tqdm
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=16, sampler=test_sampler)
for batch_idx, (mask, patch) in enumerate(train_loader):
    patch = patch.to("cuda")
    mask = mask.to("cuda")

    plt.subplot(1, 3, 1)
    plt.title('Original Patch')
    plt.imshow(patch[0].permute(1, 2, 0).cpu().detach().numpy())

    plt.subplot(1, 3, 2)
    plt.title('Mask')
    plt.imshow(mask[0][0].cpu().detach().numpy(), cmap='gray')
print("Number of training/test patches:", (len(train_indices), len(test_indices)))
print(test_indices)
import gc
net = None
gc.collect()
net = Unet(
    input_channels=3,
    num_classes=3,
    num_filters=[32, 64, 128, 192],
    initializers={"w": "he_uniform", "b": "normal"},
)
net = net.to("cuda")
from tqdm import trange
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5, weight_decay=10e-3)
criterion = nn.MSELoss(reduce="mean").to("cuda")
epochs = 4000
t = trange(epochs, desc="ML")
epoch_loss = []
for epoch in t:

    net.train()
    for step, (mask, patch) in enumerate(train_loader):
        patch = patch.to("cuda")
        mask = mask.to("cuda")
        mask2 = torch.unsqueeze(mask, 1)
        ou = net.forward(patch, mask)
        #         print(mask.shape, ou.shape)
        loss = criterion(mask.squeeze(1), ou)
        epoch_loss.append(loss.item())
        t.set_description("ML (loss=%g)" % loss)
        optimizer.zero_grad()
        #         optimizer.zero_grad()
        loss.backward()
        #         optimizer2.step()
        optimizer.step()
    if epoch % 100 == 0:
        plt.cla()
        plt.clf()
        im = net.forward(patch, mask).__getitem__(1).permute(1, 2, 0)
        plt.imshow(im.cpu().detach().numpy())
        plt.savefig(f"C:/Users/akank/Downloads/dense_dataset/intermediate/{epoch}.jpg")
    if epoch % 500 == 0:
        torch.save(net.state_dict(), "C:/Users/akank/Downloads/dense_dataset/cpt/{}.pth".format(epoch))
        im = net.forward(patch, mask).__getitem__(1).permute(1, 2, 0)
from PIL import Image
import torchvision.transforms as transforms

# Function to preprocess and dehaze a single image
def dehaze_image(model, image_path):
    # Load the image using PIL
    input_image = Image.open(image_path).convert('RGB')

    # Preprocess the image using the same transforms used during training
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    input_batch = torch.unsqueeze(input_tensor, 0).to("cuda")

    # Set the model to evaluation mode and perform the inference
    net.eval()
    with torch.no_grad():
        dehazed_image = net.forward(input_batch, mask)
    

    # Display the original and dehazed images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(input_tensor.permute(1, 2, 0).cpu().detach().numpy())

    plt.subplot(1, 2, 2)
    plt.title('Dehazed Image')
    plt.imshow(dehazed_image[0].permute(1, 2, 0).cpu().detach().numpy())

    plt.show()

# Example usage
image_path = "C:/Users/akank/Downloads/test1.jpeg"
dehaze_image(net, image_path)

from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np

# Load images
image_path1 ="C:/Users/akank/Downloads/test4.jpg" 
image_path2 = "C:/Users/akank/Downloads/ntest3.png"

# Open images
image1 = Image.open(image_path1).convert("L")  # Convert to grayscale
image2 = Image.open(image_path2).convert("L")  # Convert to grayscale

# Resize images to the same dimensions
min_width = min(image1.width, image2.width)
min_height = min(image1.height, image2.height)

image1 = image1.resize((min_width, min_height))
image2 = image2.resize((min_width, min_height))

# Convert images to numpy arrays
image1_array = np.array(image1)
image2_array = np.array(image2)

# Compute SSIM
ssim_value, _ = ssim(image1_array, image2_array, full=True)

print(f"SSIM Value: {ssim_value}")

import cv2
import numpy as np

def calculate_cnr(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate mean pixel value of the entire image
    mean_pixel_value = np.mean(gray_image)

    # Calculate standard deviation of pixel values in the image
    std_dev = np.std(gray_image)

    # Calculate CNR
    cnr = (np.max(gray_image) - mean_pixel_value) / std_dev

    return cnr

# Example usage
hazy_image_path = "C:/Users/akank/Downloads/test4.jpg"
dehazed_image_path = "C:/Users/akank/Downloads/ntest3.png"

# Load images
hazy_image = cv2.imread(hazy_image_path)
dehazed_image = cv2.imread(dehazed_image_path)

# Calculate CNR for hazy and dehazed images
cnr_hazy = calculate_cnr(hazy_image)
cnr_dehazed = calculate_cnr(dehazed_image)

print(f"CNR for Hazy Image: {cnr_hazy}")
print(f"CNR for Dehazed Image: {cnr_dehazed}")

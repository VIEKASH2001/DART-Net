##################################################################### PACKAGES ########################################################################################
import argparse
import math
import os
import time
from statistics import mean
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch import Tensor
from torch.autograd import Variable
from torch.nn import AvgPool1d
from torch.utils.data import Dataset
from torchvision import transforms

##################################################################### ARG PARSE ##############################################################################
parser = argparse.ArgumentParser(description="DART-Net")
parser.add_argument(
    "--width", type=int, default=5, help="temporal width of the input (default: 5)", choices=[3, 5, 7]
)
parser.add_argument(
    "--step", type=int, default=1, help="temporal step size of the input (default: 1)", choices=[1, 2, 3]
)
parser.add_argument(
    "--noisy-temporal", type=bool, default=True, help="noise condition on the temporal frames of input (default: True)"
)
parser.add_argument(
    "--pretrained",
    default="Random",
    help="pretrained weights for feature extractor (default: Random)",
    choices=["ImageNet", "X-ray", "X-ray_Noisy", "Random"],
)
parser.add_argument(
    "-fe",
    "--feature-extractor",
    default="ResNet101",
    help='architecture of feature extractor (default: "ResNet101")',
    choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101"],
)
parser.add_argument(
    "--ablation", action="store_true", default=False, help="switches between ablation tables 2 and 3 (default: false)"
)
parser.add_argument(
    "--input",
    default="NA-DGR",
    help="input image type for GBD module (default: NA-DGR output)",
    choices=["NA-DGR", "Augmented"],
)
parser.add_argument(
    "--patch-dim", type=int, default=42, help="patch dimension generated (default: 42)", choices=[21, 42]
)
parser.add_argument(
    "--pre",
    default="3(3,5,7)",
    help="filters in the preprocessing block (default: 3(3,5,7))",
    choices=["3(3,5,7)", "2(3,7)", "2(5,7)"],
)
parser.add_argument(
    "--LPF", type=int, default=1, help="number of LPF blocks (default: 1)", choices=[1, 2, 3]
)
parser.add_argument("--HPF", type=int, default=1, help="number of HPF blocks (default: 1)", choices=[0, 1])

args = parser.parse_args()
##################################################################### DATASET CONFIGURATION ##############################################################################
step = args.step  # step size of dataset
batch_size = args.width  # width
channel_size = 3
height, width = 320, 320
mean = 0.8313
std = 0.1875

folder = "/media/mmlab/data/OCT-Denoise/test/Final_Models"

# python gitfile.py --width 5 --step 1 --noisy-temporal True --pretrained Random --feature-extractor ResNet101 --ablation --input NA-DGR --patch-dim 42 --pre "3(3,5,7)" --LPF 1 --HPF 1
if not args.ablation:  # if we include module 2
    file_name = f"/{args.input}_{args.patch_dim}x{args.patch_dim}_{args.pre}_{args.LPF}_{args.HPF}.pt"  # path of weights to load
else:  # if we dont include module 2
    file_name = f"/{args.width}_{args.step}_{'Noisy' if args.noisy_temporal else 'X'}_{args.pretrained}_{args.feature_extractor}.pt"  # path of weights to load

# GPU Config
GPU = torch.device("cpu")
# GPU = torch.device("cuda")

# Dataset path
Test_Path = "Final_test_release"
Label_path = "Final_test_plain"


def Clean_transform(img):

    """
    This function does just the normal transformation operations on the image like resizing and normalisation.
    We call the output of this function as clean image which eventually will be used as the ground truth for our training
    """

    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


###################################################################### DATASET GENERATOR ########################################################################################
class Dataset_generator(Dataset):

    """
    Creates a generator object with the train and test dataset
    """

    def __init__(self, directory, label):
        self.directory = directory
        self.label = label
        self.filenames = []
        self.filenames_label = []
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((mean, mean, mean), (std, std, std)),
            ]
        )
        for filename in os.listdir(self.directory):
            self.In1_dir = self.directory + "/" + str(filename)
            for filename_1 in os.listdir(self.In1_dir):
                self.In2_dir = self.In1_dir + "/" + str(filename_1)
                self.filenames.append(self.In2_dir)
                self.filenames.sort()

        for filename_label in os.listdir(self.label):
            self.In1_dir = self.label + "/" + str(filename_label)
            for filename_1_label in os.listdir(self.In1_dir):
                self.In2_dir = self.In1_dir + "/" + str(filename_1_label)
                self.filenames_label.append(self.In2_dir)
                self.filenames_label.sort()

    def __len__(self):
        return int(len(self.filenames) + (-batch_size + 1) * step)

    def __getitem__(self, index):
        ls = []
        for k in range(batch_size):
            if k == int(step * (batch_size - 1) / 2):  # center image alone noisy
                inp = Clean_transform(cv2.imread(self.filenames[index + step * k]))
                inp = self.transform(inp)
            else:  # Rest all clean images
                # inp  = Clean_transform(cv2.imread(self.filenames_label[index+int(batch_size/2)]))
                inp = Clean_transform(cv2.imread(self.filenames[index + step * k]))
                inp = self.transform(inp)
            ls.append(inp)

        input = torch.stack((ls), axis=0)
        target = Clean_transform(
            cv2.imread(self.filenames_label[index + int(batch_size / 2)])
        )  # center image alone clean
        target = self.transform(target)
        return input, target, self.filenames_label[index + int(batch_size / 2)]


##################################################################### DATALOADERS ########################################################################################
def get_data():

    """
    creates a generator object for both train and test dataset.
    modify the path accordingly.
    """
    Test_dataset = Dataset_generator(Test_Path, Label_path)
    len_test_set = len(Test_dataset)
    print("The length of Test set is {}".format(len_test_set))
    return Test_dataset


def make_loader(Test_dataset):

    """
    creates a data loader object for both train and test dataset.
    """
    test_loader = torch.utils.data.DataLoader(Test_dataset, batch_size=1, shuffle=False)
    print("The length of Test loader is {}".format(len(test_loader)))

    return test_loader


##################################################################### SANITY CHECK ########################################################################################
print(" ")
print("############### Sanity Check ##################")
print(" ")
# create dataloader
test_d = get_data()
testing_data_loader = make_loader(test_d)
# load an image
image = next(iter(testing_data_loader))
print(image[0].min(), image[0].max(), image[0].shape, image[0].dtype)
print(image[1].min(), image[1].max(), image[1].shape, image[1].dtype)
print(" ")
##################################################################### ARCHITECTURE PARAMETERS ########################################################################################
# no. of layers
pre_n_layers = 3
hpf_n_layers = 3
lpf_n_layers = 3
prox_n_layers = 1  # total no.of LPF Blocks

# no. of features
Nfeat = 132  # must be multiple of 3
pre_Nfeat = int(Nfeat / 3)  # preprocessing block features: 44
pre_fnet_Nfeat = pre_Nfeat
prox_fnet_Nfeat = Nfeat
hpf_fnet_Nfeat = Nfeat

# gconv params
rank_theta = 11
stride = int(Nfeat / 3)
stride_pregconv = int(Nfeat / 3)
min_nn = 16 + 8

# Input patch
patch_dim = patch_height = patch_width = args.patch_dim
patch_size = [patch_dim, patch_dim, batch_size]
N = patch_size[0] * patch_size[1]
Recon_patch_batch_size = int((height / patch_height) * (width / patch_width))
patch_batch_size = 100

# For architecture 1
BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d
psample = lambda x, size: F.interpolate(x, size, mode="bilinear", align_corners=True)

# Local Mask function
def create_mask(N):
    local_mask = np.ones([N, N])
    for ii in range(N):
        if ii == 0:
            local_mask[ii, (ii + 1, ii + patch_size[1], ii + patch_size[1] + 1)] = 0  # top-left
        elif ii == N - 1:
            local_mask[ii, (ii - 1, ii - patch_size[1], ii - patch_size[1] - 1)] = 0  # bottom-right
        elif ii == patch_size[0] - 1:
            local_mask[ii, (ii - 1, ii + patch_size[1], ii + patch_size[1] - 1)] = 0  # top-right
        elif ii == N - patch_size[0]:
            local_mask[ii, (ii + 1, ii - patch_size[1], ii - patch_size[1] + 1)] = 0  # bottom-left
        elif ii < patch_size[0] - 1 and ii > 0:
            local_mask[
                ii, (ii + 1, ii - 1, ii + patch_size[1] - 1, ii + patch_size[1], ii + patch_size[1] + 1)
            ] = 0  # first row
        elif ii < N - 1 and ii > N - patch_size[0]:
            local_mask[
                ii, (ii + 1, ii - 1, ii - patch_size[1] - 1, ii - patch_size[1], ii - patch_size[1] + 1)
            ] = 0  # last row
        elif ii % patch_size[1] == 0:
            local_mask[
                ii, (ii + 1, ii - patch_size[1], ii + patch_size[1], ii - patch_size[1] + 1, ii + patch_size[1] + 1)
            ] = 0  # first col
        elif ii % patch_size[1] == patch_size[1] - 1:
            local_mask[
                ii, (ii - 1, ii - patch_size[1], ii + patch_size[1], ii - patch_size[1] - 1, ii + patch_size[1] - 1)
            ] = 0  # last col
        else:
            local_mask[
                ii,
                (
                    ii + 1,
                    ii - 1,
                    ii - patch_size[1],
                    ii - patch_size[1] + 1,
                    ii - patch_size[1] - 1,
                    ii + patch_size[1],
                    ii + patch_size[1] + 1,
                    ii + patch_size[1] - 1,
                ),
            ] = 0

    local_mask = local_mask[np.newaxis, :, :]
    local_mask = torch.tensor(local_mask)
    return local_mask


##################################################################### ARCHITECTURE 2 ########################################################################################
class GCDN(nn.Module):
    def __init__(self):
        super(GCDN, self).__init__()
        # basic params
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_size = patch_size
        self.N = N
        self.Ngpus = 1
        self.grad_accum = 1
        self.starter_learning_rate = 1e-4
        self.patch_batch_size = patch_batch_size
        # no. of layers
        self.pre_n_layers = pre_n_layers
        self.hpf_n_layers = hpf_n_layers
        self.lpf_n_layers = lpf_n_layers
        self.prox_n_layers = prox_n_layers  # total no.of LPF Blocks

        # no. of features
        self.Nfeat = 132  # must be multiple of 3
        self.pre_Nfeat = int(self.Nfeat / 3)  # preprocessing block features: 44
        self.pre_fnet_Nfeat = self.pre_Nfeat
        self.prox_fnet_Nfeat = self.Nfeat
        self.hpf_fnet_Nfeat = self.Nfeat

        # gconv params
        self.rank_theta = 11
        self.stride = int(self.Nfeat / 3)
        self.stride_pregconv = int(self.Nfeat / 3)
        self.min_nn = 16 + 8

        # pre-processing Stage
        # top layer
        self.pregconv3_l = pregconv(44, 44, 3, 1)
        self.pre3_1 = pre(batch_size, 44, 3, 1)
        self.pre3_2 = pre(44, 44, 3, 1)
        self.pre3_3 = pre(44, 44, 3, 1)
        # mid layer
        self.pregconv5_l = pregconv(44, 44, 5, 2)
        self.pre5_1 = pre(batch_size, 44, 5, 2)
        self.pre5_2 = pre(44, 44, 5, 2)
        self.pre5_3 = pre(44, 44, 5, 2)
        # bottom layer
        self.pregconv7_l = pregconv(44, 44, 7, 3)
        self.pre7_1 = pre(batch_size, 44, 7, 3)
        self.pre7_2 = pre(44, 44, 7, 3)
        self.pre7_3 = pre(44, 44, 7, 3)

        # high pass filter
        self.hpf = hpfconv(132, 132, 3, 1)
        self.hpf_bn = hpfbn(self.patch_batch_size, 1e-3)

        # identical 4 layers
        self.prox = proxconv(132, 132, 3, 1)
        self.prox_bn = proxbn(self.patch_batch_size, 1e-3)

        self.last = lastconv(132, batch_size, 3, 1)

    def forward(self, x, local_mask, GK):
        x_pre = self.pre_processing_block(x, local_mask)
        x_hpf = self.hpf_block(x_pre, local_mask)
        x_prox = self.prox_block(x_pre, x_hpf, local_mask, GK)
        x_last = self.last_block(x_prox, local_mask)
        return x_last

    def pre_processing_block(self, x, local_mask):
        x3 = self.pre3_1(x)
        x3 = self.pre3_2(x3)
        x3 = self.pre3_3(x3)
        x3 = x3.reshape(-1, self.N, self.pre_Nfeat)
        x3_nl = gconv(
            x3,
            local_mask,
            self.pre_Nfeat,
            self.pre_Nfeat,
            self.stride_pregconv,
            self.stride_pregconv,
            compute_graph_bool=True,
            return_graph=False,
        )
        x3_l = self.pregconv3_l(x3.reshape([-1, self.pre_Nfeat, self.patch_size[0], self.patch_size[1]])).reshape(
            [-1, self.N, self.pre_Nfeat]
        )
        x3 = lnl_aggregation(x3_l, x3_nl, nn.init.zeros_(torch.empty(1, self.pre_Nfeat)).cuda())

        x5 = self.pre5_1(x)
        x5 = self.pre5_2(x5)
        x5 = self.pre5_3(x5)
        x5 = x5.reshape(-1, self.N, self.pre_Nfeat)
        x5_nl = gconv(
            x5,
            local_mask,
            self.pre_Nfeat,
            self.pre_Nfeat,
            self.stride_pregconv,
            self.stride_pregconv,
            compute_graph_bool=True,
            return_graph=False,
        )
        x5_l = self.pregconv5_l(x5.reshape([-1, self.pre_Nfeat, self.patch_size[0], self.patch_size[1]])).reshape(
            [-1, self.N, self.pre_Nfeat]
        )
        x5 = lnl_aggregation(x5_l, x5_nl, nn.init.zeros_(torch.empty(1, self.pre_Nfeat)).cuda())

        x7 = self.pre7_1(x)
        x7 = self.pre7_2(x7)
        x7 = self.pre7_3(x7)
        x7 = x7.reshape(-1, self.N, self.pre_Nfeat)
        x7_nl = gconv(
            x7,
            local_mask,
            self.pre_Nfeat,
            self.pre_Nfeat,
            self.stride_pregconv,
            self.stride_pregconv,
            compute_graph_bool=True,
            return_graph=False,
        )
        x7_l = self.pregconv7_l(x7.reshape([-1, self.pre_Nfeat, self.patch_size[0], self.patch_size[1]])).reshape(
            [-1, self.N, self.pre_Nfeat]
        )
        x7 = lnl_aggregation(x7_l, x7_nl, nn.init.zeros_(torch.empty(1, self.pre_Nfeat)).cuda())
        x = torch.cat([x3, x5, x7], axis=2)
        x = torch.nn.LeakyReLU(0.2)(x)
        return x

    def prox_block(self, x, x_hpf, local_mask, GK):
        for i in range(self.prox_n_layers):
            weight = 0.5
            alpha = nn.init.constant_(torch.empty([]), weight)
            beta = nn.init.constant_(torch.empty([]), 1 - weight)
            X = beta * x_hpf + (alpha) * x  # 0.5*x_hpf + 0.5*x
            x_old = x + 0.0
            x = self.prox(x.reshape([-1, self.Nfeat, self.patch_size[0], self.patch_size[1]])).reshape(
                [-1, self.N, self.Nfeat]
            )
            x = self.prox_bn(torch.unsqueeze(x, 0))
            x = nn.LeakyReLU(0.2)(x)
            if GK == "True":
                x = torch.squeeze(get_gaussian_filter(kernel_size=3, sigma=2, channels=100)(x.cuda()), 0)
            else:
                x = torch.squeeze(x, 0)
            for j in range(self.lpf_n_layers):
                if j == 0:
                    x_nl, D = gconv(
                        x,
                        local_mask,
                        self.Nfeat,
                        self.Nfeat,
                        self.stride,
                        self.stride,
                        compute_graph_bool=True,
                        return_graph=True,
                    )
                else:
                    x_nl = gconv(
                        x,
                        local_mask,
                        self.Nfeat,
                        self.Nfeat,
                        self.stride,
                        self.stride,
                        compute_graph_bool=False,
                        return_graph=False,
                        D=D,
                    )
                x_l = self.prox(x.reshape([-1, self.Nfeat, self.patch_size[0], self.patch_size[1]])).reshape(
                    [-1, self.N, self.Nfeat]
                )
                x = lnl_aggregation(x_l, x_nl, nn.init.zeros_(torch.empty(1, self.prox_fnet_Nfeat)).cuda())
                x = self.prox_bn(torch.unsqueeze(x, 0))
                x = torch.squeeze(nn.LeakyReLU(0.2)(x), 0)
            x = x + x_old
        return x

    def hpf_block(self, x, local_mask):
        x_hpf = x + 0.0
        x_hpf = self.hpf(x_hpf.reshape([-1, self.Nfeat, self.patch_size[0], self.patch_size[1]])).reshape(
            [-1, self.N, self.Nfeat]
        )
        x_hpf = self.hpf_bn(torch.unsqueeze(x_hpf, 0))
        x_hpf = torch.squeeze(nn.LeakyReLU(0.2)(x_hpf), 0)
        for i in range(self.hpf_n_layers):
            if i == 0:
                x_hpf_nl, D = gconv(
                    x_hpf,
                    local_mask,
                    self.Nfeat,
                    self.Nfeat,
                    self.stride,
                    self.stride,
                    compute_graph_bool=True,
                    return_graph=True,
                )
            else:
                x_hpf_nl = gconv(
                    x_hpf,
                    local_mask,
                    self.Nfeat,
                    self.Nfeat,
                    self.stride,
                    self.stride,
                    compute_graph_bool=False,
                    return_graph=False,
                    D=D,
                )
            x_hpf_l = self.hpf(x_hpf.reshape([-1, self.Nfeat, self.patch_size[0], self.patch_size[1]])).reshape(
                [-1, self.N, self.Nfeat]
            )
            x_hpf = lnl_aggregation(x_hpf_l, x_hpf_nl, nn.init.zeros_(torch.empty(1, self.hpf_fnet_Nfeat)).cuda())
            x_hpf = nn.LeakyReLU(0.2)(x_hpf)
        return x_hpf

    def last_block(self, x, local_mask):
        x_nl = gconv(
            x,
            local_mask,
            self.Nfeat,
            self.patch_size[2],
            self.stride,
            self.patch_size[2],
            compute_graph_bool=True,
            return_graph=False,
        )
        x_l = self.last(x.reshape([-1, self.Nfeat, self.patch_size[0], self.patch_size[1]])).reshape(
            [-1, self.N, self.patch_size[2]]
        )
        x = lnl_aggregation(x_l, x_nl, nn.init.zeros_(torch.empty(1, 1)).cuda())
        x = x.reshape([-1, self.patch_size[2], self.patch_size[0], self.patch_size[1]])
        return x


def pre(in_channels, out_channels, kernal, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernal, padding=padding, bias=False, padding_mode="reflect"),
        nn.LeakyReLU(0.2, inplace=True),
    )


def pregconv(in_channels, out_channels, kernal, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernal, padding=padding, bias=False, padding_mode="reflect"),
    )


def hpfconv(in_channels, out_channels, kernal, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernal, padding=padding, bias=True, padding_mode="reflect"),
    )


def hpfbn(ch, eps=1e-3):
    return nn.Sequential(
        torch.nn.BatchNorm2d(ch, eps),
    )


def proxconv(in_channels, out_channels, kernal, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernal, padding=padding, bias=True, padding_mode="reflect"),
    )


def proxbn(ch, eps=1e-3):
    return nn.Sequential(
        torch.nn.BatchNorm2d(ch, eps),
    )


def lastconv(in_channels, out_channels, kernal, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernal, padding=padding, bias=True, padding_mode="reflect"),
    )


def lnl_aggregation(h_l, h_nl, b):
    return torch.div(h_l + h_nl, 2) + b


def compute_graph(h, local_mask):
    id_mat = 2 * torch.eye(N)  # 1764 * 1764
    h = h.type(torch.DoubleTensor).detach()
    sq_norms = torch.sum(h * h, 2)  # (B,N), h*h is element wise product, so dimension of output same as h
    D = torch.abs(
        torch.unsqueeze(sq_norms, 2) + torch.unsqueeze(sq_norms, 1) - 2 * torch.matmul(h, torch.transpose(h, 1, 2))
    )  # (B, N, N)
    D = D.type(torch.FloatTensor)
    D = torch.mul(D, local_mask)
    D = D - id_mat
    h = h.type(torch.FloatTensor)
    return D.cuda()


def gconv(h, local_mask, in_feat, out_feat, stride_th1, stride_th2, compute_graph_bool=True, return_graph=False, D=[]):
    if compute_graph_bool:
        D = compute_graph(h, local_mask)
    _, top_idx = torch.topk(-D, min_nn + 1)  # (B, N, d+1)
    top_idx2 = torch.tensor(
        np.tile(torch.unsqueeze(top_idx[:, :, 0].cpu(), 2), [1, 1, min_nn - 8]).reshape([-1, N * (min_nn - 8)])
    ).cuda()
    top_idx = top_idx[:, :, 9:].reshape([-1, N * (min_nn - 8)])  # (B, N*d)

    x_tilde1 = []
    for i in range((top_idx[0].shape)[0]):
        x_tilde1.append(h[0][top_idx[0][i]])
    x_tilde1 = torch.stack(x_tilde1)
    x_tilde1 = x_tilde1.reshape(1, x_tilde1.shape[0], x_tilde1.shape[1])

    x_tilde2 = []
    for i in range((top_idx2[0].shape)[0]):
        x_tilde2.append(h[0][top_idx2[0][i]])
    x_tilde2 = torch.stack(x_tilde2)
    x_tilde2 = x_tilde2.reshape(1, x_tilde2.shape[0], x_tilde2.shape[1])

    labels = x_tilde1 - x_tilde2  # (B, K, dlm1)
    x_tilde1 = x_tilde1.reshape([-1, in_feat])  # (B*K, dlm1)
    labels = labels.reshape([-1, in_feat])  # (B*K, dlm1)
    d_labels = torch.sum(labels * labels, 1).reshape([-1, min_nn - 8])  # (B*N, d)

    W = nn.init.xavier_uniform_(torch.empty(in_feat, in_feat)).cuda()
    b = nn.init.xavier_uniform_(torch.empty(1, in_feat)).cuda()
    labels = torch.nn.LeakyReLU(0.2)(torch.matmul(labels, W) + b)  #  (B*K, F)

    labels_exp = torch.unsqueeze(labels, 1)  # (B*K, 1, F)
    labels1 = labels_exp + 0.0
    for ss in range(1, int(in_feat / stride_th1)):
        labels1 = torch.cat(
            [labels1, myroll(labels_exp, shift=(ss + 1) * stride_th1, axis=2)], axis=1
        )  # (B*K, dlm1/stride, dlm1)

    labels2 = labels_exp + 0.0
    for ss in range(1, int(out_feat / stride_th2)):
        labels2 = torch.cat(
            [labels2, myroll(labels_exp, shift=(ss + 1) * stride_th2, axis=2)], axis=1
        )  # (B*K, dl/stride, dlm1)

    W = nn.init.normal_(
        torch.empty(in_feat, stride_th1 * rank_theta), 0, 1.0 / (np.sqrt(in_feat + 0.0) * np.sqrt(in_feat + 0.0))
    ).cuda()
    theta1 = torch.matmul(labels1.reshape([-1, in_feat]), W)  # (B*K*dlm1/stride, R*stride)
    b = nn.init.zeros_(torch.empty(1, rank_theta, in_feat)).cuda()
    theta1 = theta1.reshape([-1, rank_theta, in_feat]) + b

    W = nn.init.normal_(
        torch.empty(in_feat, stride_th2 * rank_theta), 0, 1.0 / (np.sqrt(in_feat + 0.0) * np.sqrt(in_feat + 0.0))
    ).cuda()
    theta2 = torch.matmul(labels2.reshape([-1, in_feat]), W)  # (B*K*dl/stride, R*stride)
    b = nn.init.zeros_(torch.empty(1, rank_theta, out_feat)).cuda()
    theta2 = theta2.reshape([-1, rank_theta, out_feat]) + b
    W = nn.init.normal_(torch.empty(in_feat, rank_theta), 0, 1.0 / np.sqrt(rank_theta + 0.0)).cuda()
    b = nn.init.zeros_(torch.empty(1, rank_theta)).cuda()
    thetal = torch.unsqueeze((torch.matmul(labels, W) + b), 2)

    x = torch.matmul(theta1, torch.unsqueeze(x_tilde1, 2))  # (B*K, R, 1)
    x = torch.mul(x, thetal)  # (B*K, R, 1)
    x = torch.matmul(torch.transpose(theta2, 1, 2), x)[:, :, 0]  # (B*K, dl)
    x = x.reshape([-1, min_nn - 8, out_feat])  # (N, d, dl)
    x = torch.mul(x, torch.unsqueeze(torch.exp(-torch.div(d_labels, 10)), 2))  # (N, d, dl)
    x = torch.mean(x, 1)  # (N, dl)
    x = x.reshape([-1, N, out_feat])  # (B, N, dl)
    if return_graph:
        return x, D
    else:
        return x


def myroll(h, shift=0, axis=2):
    h_len = h.shape[2]
    return torch.cat([h[:, :, h_len - shift :], h[:, :, : h_len - shift]], axis=2)


def get_gaussian_filter(kernel_size, sigma, channels):

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)

    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2.0

    # Calculate the 2-dimensional gaussian kernel which is the product of two gaussian distributions
    # for two different variables (in this case called x and y)
    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    if kernel_size == 3:
        padding = 1
    elif kernel_size == 5:
        padding = 2
    else:
        padding = 0

    gaussian_filter = nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        groups=channels,
        bias=False,
        padding=padding,
    )
    gaussian_filter.weight.data = gaussian_kernel.cuda()
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


##################################################################### ARCHITECTURE 1 ########################################################################################
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1), BatchNorm2d(plane))

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b, c, h, w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q, node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return out


class DualGCN(nn.Module):
    """
    Feature GCN with coordinate GCN
    """

    def __init__(self, planes, ratio=4):
        super(DualGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
        )
        self.gcn_local_attention = SpatialGCN(planes)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False), BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat):
        # # # # Local # # # #
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode="bilinear", align_corners=True)
        spatial_local_feat = x * local + x

        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x + y)

        # cat or sum, nearly the same results
        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return out


class DualGCNHead(nn.Module):
    def __init__(self, inplanes, interplanes, num_classes):
        super(DualGCNHead, self).__init__()
        self.conva = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False), BatchNorm2d(interplanes), nn.ReLU(interplanes)
        )
        self.dualgcn = DualGCN(interplanes)
        self.convb = nn.Sequential(
            nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes + interplanes, interplanes, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            ## adding these on my own, had a normal conv2d(512,num_classes,k=1,p=0) here before, doing this to upsample my image and restore it
            nn.ConvTranspose2d(512, 128, 2, stride=2),
            BatchNorm2d(128),
            nn.ReLU(128),
            nn.ConvTranspose2d(128, 32, 2, stride=2),
            BatchNorm2d(32),
            nn.ReLU(32),
            nn.ConvTranspose2d(32, 3, 2, stride=2),
        )

    def forward(self, x):
        output = self.conva(x)
        output = self.dualgcn(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output


class ChannelPool(AvgPool1d):
    def forward(self, x):
        n, c, w, h = x.size()
        x_inter = x.view(n, c, w * h).permute(1, 2, 0)
        pooled = F.max_pool1d(x_inter, self.kernel_size)
        _, _, n = pooled.size()
        pooled = pooled.permute(2, 0, 1)
        return pooled.view(n, c, w, h)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            conv3x3(3, 64, stride=2),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 128),
        )

        self.bn1 = BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 2, 4)
        )  # (1,2048,64,64)
        self.agg = ChannelPool(5)
        # DualGCN
        self.head = DualGCNHead(2048, 512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                dilation=dilation,
                downsample=downsample,
                multi_grid=generate_multi_grid(0, multi_grid),
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid))
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.agg(x)
        x = self.head(x)
        # mean and std add
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation * multi_grid,
            dilation=dilation * multi_grid,
            bias=False,
        )
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def adjust_learning_rate(optimizer, i_iter, total_steps):
    lr = lr_poly(learning_rate, i_iter, total_steps, power)
    optimizer.param_groups[0]["lr"] = lr
    return lr


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def DualSeg_res101(num_classes):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model


##################################################################### Load Pretrained weights ########################################################################################
"""fixed configuraration details"""
num_classes = 1
lowest = 100000
# model_i = DualSeg_res101(num_classes).to(GPU)
# Model Save path
model_f = torch.load("model1.pt", map_location=GPU).to(GPU)
criterion = nn.MSELoss().to(GPU)
# Variables to be input
input_img = Variable(torch.FloatTensor(batch_size, channel_size, height, width)).to(GPU)
target_img = Variable(torch.FloatTensor(channel_size, height, width)).to(GPU)
##################################################################### eval ########################################################################################
def test(model, testing_data_loader, criterion, input_img, target_img, files):
    global img2

    loss_list = []
    ssim_list = []
    psnr_list = []
    begin = time.time()
    for i_iter, batch in enumerate(testing_data_loader):
        images, labels, file_path = batch
        input_img.data.copy_(torch.squeeze(images))
        target_img.data.copy_(torch.squeeze(labels))
        images = input_img
        labels = torch.squeeze(target_img)
        preds = model(images)
        preds = torch.squeeze(preds)
        loss = criterion(preds, labels)
        loss_list.append(loss.item())
        img1 = np.transpose(labels.detach().cpu().numpy(), (1, 2, 0)) * std + mean
        img2 = np.transpose(preds.detach().cpu().numpy(), (1, 2, 0)) * std + mean
        grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ssim_t, diff = ssim(grayA, grayB, full=True)
        psnr_t = psnr(img1, img2, data_range=1)
        # print("Test image no.: ",i_iter+1," - SSIM: ",ssim_t,", PSNR: ",psnr_t)
        ssim_list.append(ssim_t)
        psnr_list.append(psnr_t.item())
    mean_loss = np.mean(loss_list)
    mean_ssim = np.mean(ssim_list)
    mean_psnr = np.mean(psnr_list)
    end = time.time()
    print(
        "Model no.:",
        files,
        " MSE: ",
        mean_loss,
        " Testing SSIM: ",
        mean_ssim,
        " Testing PSNR: ",
        mean_psnr,
        " Time taken: ",
        (end - begin) / 60,
        " mins",
    )

weights = torch.load(folder + file_name, map_location=GPU)
model_f.load_state_dict(weights["model"])
print(model_f, folder + file_name)
# test(model_f, testing_data_loader, criterion, input_img, target_img, in_folders+"/"+str(file_name))

# folder = "/media/mmlab/data/OCT-Denoise/test/Final_Models"
# print("############### Testing Started ##################")
# for in_folders in os.listdir(folder):
#     in_dir = folder + "/" + str(in_folders)
#     weights = torch.load(in_dir,map_location=GPU)
#     model_f.load_state_dict(weights["model"])
#     print(model_f, in_folders)
#     # test(model_f, testing_data_loader, criterion, input_img, target_img, in_folders+"/"+str(in_folders))
# print("############### Testing ended ##################")


# print("############### Testing Started ##################")
# for in_folders in os.listdir(folder):
#     in_dir = folder + "/" + str(in_folders)
#     for files in os.listdir(in_dir):
#             dir = in_dir + "/" + str(files)
#             model_f = torch.load(dir).to(GPU)

#             test(model_f, testing_data_loader, criterion, input_img, target_img, in_folders+"/"+str(files))
# print("############### Testing ended ##################")

# w2 = torch.load("/media/mmlab/data/OCT-Denoise/test/model2.pt")
# folder = "/media/mmlab/data/OCT-Denoise/test/Main_models"
# for in_folders in os.listdir(folder):
#     in_dir = folder + "/" + str(in_folders)
#     w1 = torch.load(in_dir)
#     torch.save({
#         "model": w1.state_dict(),
#         "model_2_state_dict": w2.state_dict()
#     }, "/media/mmlab/data/OCT-Denoise/test/Final Models/"+str(in_folders))

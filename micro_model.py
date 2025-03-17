import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t, _ratio_3_t, _ratio_2_t
from torch.nn.modules.pooling import _MaxPoolNd
from torch import Tensor
from torch.autograd import Variable
import math
from functools import partial

LOWER_1, UPPER_1 = 1, 3
LOWER_2, UPPER_2 = 1.5, 3
LOWER_3, UPPER_3 = 3, 4.5

# def fuzzy_pooling_2d_vectorized(x, kernel_size, stride):
#     # x: B, T, C, H, W
#     # Stride the x first for each image (H x W)
#     x_strided = torch.as_strided(x, 
#                                 size=(x.shape[0], x.shape[1], x.shape[2], 
#                                     (x.shape[3] - kernel_size)//stride + 1, 
#                                     (x.shape[4] - kernel_size)//stride + 1, 
#                                     kernel_size, kernel_size), 
#                                 stride=(x.stride(0), x.stride(1), x.stride(2), 
#                                         x.stride(3)*stride, 
#                                         x.stride(4)*stride, 
#                                         x.stride(3), x.stride(4))
#                                 )

#     # Do 3 different sigmoid, results are 3 (B, T, C, H, W) --> u1, u2, u3
#     u1 = torch.sigmoid(UPPER_1 * (x_strided - LOWER_1))
#     u2 = torch.sigmoid(UPPER_2 * (x_strided - LOWER_2))
#     u3 = torch.sigmoid(UPPER_3 * (x_strided - LOWER_3))

#     # print("Before")
#     # print(x_strided[0, 0, 0, 0, 0])
    

#     # print("Res")
#     # print(u1[0, 0, 0, 0, 0])
#     # print(u2[0, 0, 0, 0, 0])
#     # print(u3[0, 0, 0, 0, 0])

#     # Do max for each sigmoid results, which results in 3 (B, T, C, H, W) with value 0 or 1 --> max_u1, max_u2, max_u3
#     # Step 1: Stack the tensors along a new dimension
#     stacked_tensors = torch.stack([u1, u2, u3], dim=-1)  # Shape will be [3, 3, 3]

#     # Step 2: Find the maximum value along the last dimension (axis=-1)
#     max_values, max_indices = torch.max(stacked_tensors, dim=-1)

#     # Step 3: Create binary masks for u1, u2, u3 based on max values
#     max_u1 = (max_indices == 0).float()  # 1 where u1 is the max, 0 otherwise
#     max_u2 = (max_indices == 1).float()  # 1 where u2 is the max, 0 otherwise
#     max_u3 = (max_indices == 2).float()  # 1 where u3 is the max, 0 otherwise

#     # print("Max")
#     # print(max_u1[0, 0, 0, 0, 0])
#     # print(max_u2[0, 0, 0, 0, 0])
#     # print(max_u3[0, 0, 0, 0, 0])

#     # For each sigmoid max results, do max_u1 * strided_x * u1 + max_u2 * strided_x * u2 + max_u3 * strided_x * u3
#     x = max_u1 * x_strided * u1 + max_u2 * x_strided * u2 + max_u3 * x_strided * u3

#     # Mean the result along the last 2 dimensions
#     x = x.mean(dim=(-1, -2))
#     return x

def fuzzy_pooling_2d_vectorized(x, kernel_size, stride):
    # x: B, T, C, H, W
    # Stride the x first for each image (H x W)
    x_strided = torch.as_strided(x, 
                                size=(x.shape[0], x.shape[1], x.shape[2], 
                                    (x.shape[3] - kernel_size)//stride + 1, 
                                    (x.shape[4] - kernel_size)//stride + 1, 
                                    kernel_size, kernel_size), 
                                stride=(x.stride(0), x.stride(1), x.stride(2), 
                                        x.stride(3)*stride, 
                                        x.stride(4)*stride, 
                                        x.stride(3), x.stride(4))
                                )

    # Do 3 different sigmoid, results are 3 (B, T, C, H, W) --> u1, u2, u3
    u1 = torch.sigmoid(UPPER_1 * (x_strided - LOWER_1))
    u2 = torch.sigmoid(UPPER_2 * (x_strided - LOWER_2))
    u3 = torch.sigmoid(UPPER_3 * (x_strided - LOWER_3))

    # print("Before")
    # print(x_strided[0, 0, 0, 0, 0])
    

    # print("Res")
    # print(u1[0, 0, 0, 0, 0])
    # print(u2[0, 0, 0, 0, 0])
    # print(u3[0, 0, 0, 0, 0])

    # Do max for each sigmoid results, which results in 3 (B, T, C, H, W) with value 0 or 1 --> max_u1, max_u2, max_u3
    # Step 1: Stack the tensors along a new dimension
    stacked_tensors = torch.stack([u1, u2, u3], dim=-1)  # Shape will be [3, 3, 3]

    # Step 2: Find the maximum value along the last dimension (axis=-1)
    max_values, max_indices = torch.max(stacked_tensors, dim=-1)

    # Step 3: Create binary masks for u1, u2, u3 based on max values
    max_u1 = (max_indices == 0).float()  # 1 where u1 is the max, 0 otherwise
    max_u2 = (max_indices == 1).float()  # 1 where u2 is the max, 0 otherwise
    max_u3 = (max_indices == 2).float()  # 1 where u3 is the max, 0 otherwise

    # print("Max")
    # print(max_u1[0, 0, 0, 0, 0])
    # print(max_u2[0, 0, 0, 0, 0])
    # print(max_u3[0, 0, 0, 0, 0])

    # For each sigmoid max results, do max_u1 * strided_x * u1 + max_u2 * strided_x * u2 + max_u3 * strided_x * u3
    x = max_u1 * x_strided * u1 + max_u2 * x_strided * u2 + max_u3 * x_strided * u3

    # Mean the result along the last 2 dimensions
    x = x.mean(dim=(-1, -2))
    return x

# test = fuzzy_pooling_2d_vectorized(test_tensor, 3, 2)
# visualize_video_frames(test, 3, 3)

# %%
class FuzzyPool2DVectorized(_MaxPoolNd):
    kernel_size: _size_2_t
    stride: _size_2_t

    def forward(self, input: Tensor) -> Tensor:
        return fuzzy_pooling_2d_vectorized(input, self.kernel_size, self.stride)


# %%
# __all__ = ['ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                    stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                            out.size(2), out.size(3),
                            out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                            padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, sample_size, sample_duration, fuzzy=False, 
                 shortcut_type='B', num_classes=400, last_fc=True, dropout=0, softmax=False, temperature=1.0):
        self.last_fc = last_fc

        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
                            padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, dropout=dropout)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2, dropout=dropout)
        
        self.fuzzy_logic = None
        last_duration = math.ceil(sample_duration / 16)
        last_size = math.ceil(sample_size / 32)
        
        if fuzzy:
            self.fuzzy_logic = FuzzyPool2DVectorized(kernel_size=3, stride=2)
            last_size = (last_size-3) // 2 + 1
            
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        
        # Use last layer to calculate linear input shape
        # dummy = torch.ones([batch_size, 3, sample_duration, sample_size, sample_size])
        # dense_input_shape = self._calculate_linear_input(dummy)
        # print(dummy.shape)
        # print(dense_input_shape)
        # self.fc = nn.Linear(dense_input_shape, num_classes)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.apply_softmax = softmax
        self.softmax_temp = temperature

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dropout=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                    planes=planes * block.expansion,
                                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def _calculate_linear_input(self, dummy_data):
        dummy_data = self.conv1(dummy_data)
        dummy_data = self.bn1(dummy_data)
        dummy_data = self.relu(dummy_data)
        dummy_data = self.maxpool(dummy_data)

        dummy_data = self.layer1(dummy_data)
        dummy_data = self.layer2(dummy_data)
        dummy_data = self.layer3(dummy_data)
        dummy_data = self.layer4(dummy_data)

        if self.fuzzy_logic:
            dummy_data = self.fuzzy_logic(dummy_data)
            dummy_data = dummy_data.unsqueeze(4)
        else:
            dummy_data = self.avgpool(dummy_data)
        
        return dummy_data.view(dummy_data.size(0), -1).shape[-1]


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # print(x.shape)
        # x = x[:, :, : ,-1]
        # print(x.shape)
        
        if self.fuzzy_logic:
            x = self.fuzzy_logic(x)
        
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        if self.last_fc:
            x = self.fc(x)

        if self.apply_softmax:
            x /= self.softmax_temp
            x = F.softmax(x, dim=1)
            
        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(ft_begin_index))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

class TSN(nn.Module):
    def __init__(self, backbone, num_classes):
        super(TSN, self).__init__()  # Initialize nn.Module

        self.backbone = backbone  # Backbone model
    
    def forward(self, x):
        # x input should have shape (n, input_shape), where n is the batch size     
        # Apply backbone to the batch and get the output
        batch_size = len(x)
        output_dim = self.backbone(x[0]).size(-1)  # Assuming backbone output dim is consistent
        batch_output = torch.empty(batch_size, output_dim, device=x.device)  # Preallocate tensor for batch outputs

        for i, batch_input in enumerate(x):
            # Apply backbone to the input
            output = self.backbone(batch_input)  # Shape: (n, output_dim)
            
            # Average across the sequence dimension (dim=0 for n)
            averaged_output = torch.mean(output, dim=0)  # Shape: (output_dim,)
            
            # Directly store averaged output in the preallocated tensor
            batch_output[i] = averaged_output

        return batch_output
    
    def check_process(self, x):
        # x input should have shape (n, input_shape), where n is the batch size     
        # Apply backbone to the batch and get the output
        batch_size = len(x)
        output_dim = self.backbone(x[0]).size(-1)  # Assuming backbone output dim is consistent
        batch_output = torch.empty(batch_size, output_dim, device=x.device)  # Preallocate tensor for batch outputs
        batch_model_output = []

        for i, batch_input in enumerate(x):
            # Apply backbone to the input
            output = self.backbone(batch_input)  # Shape: (n, output_dim)
            batch_model_output.append(output)
            
            # Average across the sequence dimension (dim=0 for n)
            averaged_output = torch.mean(output, dim=0)  # Shape: (output_dim,)
            
            # Directly store averaged output in the preallocated tensor
            batch_output[i] = averaged_output

        return batch_output, batch_model_output
    
models = {
    'fuzzy': resnet10,
    'resnet10': resnet10,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152
}
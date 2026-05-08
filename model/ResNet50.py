import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels * BottleNeck.expansion,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BottleNeck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.residual_function(x)
        shortcut = self.shortcut(x)
        out = out + shortcut
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.conv2_x = self._make_layer(BottleNeck, 64, 3, stride=1)
        self.conv3_x = self._make_layer(BottleNeck, 128, 4, stride=2)
        self.conv4_x = self._make_layer(BottleNeck, 256, 6, stride=2)
        self.conv5_x = self._make_layer(BottleNeck, 512, 3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleNeck.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []

        for stride in strides:
            layers.append(
                block(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride
                )
            )

            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight

        v, m = torch.var_mean(
            w,
            dim=[1, 2, 3],
            keepdim=True,
            unbiased=False
        )

        w = (w - m) / torch.sqrt(v + 1e-5)

        return F.conv2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return StdConv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=bias
    )

def conv3x3(in_channels, out_channels, stride=1, groups=1, bias=False):
    return StdConv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        groups=groups
    )

class BottleNeckV2(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        mid_channels = out_channels
        out_channels_expanded = out_channels * BottleNeckV2.expansion

        self.gn1 = nn.GroupNorm(32, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(in_channels, mid_channels)

        self.gn2 = nn.GroupNorm(32, mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_channels, mid_channels, stride=stride)

        self.gn3 = nn.GroupNorm(32, mid_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_channels, out_channels_expanded)

        self.shortcut = None

        if stride != 1 or in_channels != out_channels_expanded:
            self.shortcut = conv1x1(
                in_channels,
                out_channels_expanded,
                stride=stride
            )

    def forward(self, x):
        out = self.gn1(x)
        out = self.relu1(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(out)
        else:
            shortcut = x

        out = self.conv1(out)

        out = self.gn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.gn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out = out + shortcut

        return out
    

class ResNet50V2(nn.Module):
    def __init__(self, output_stride=16):
        super().__init__()

        assert output_stride in [16, 32], "output_stride must be 16 or 32"

        self.output_stride = output_stride
        self.in_channels = 64

        self.root = nn.Sequential(
            StdConv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(BottleNeckV2, 64, 3, stride=1)
        self.conv3_x = self._make_layer(BottleNeckV2, 128, 4, stride=2)
        self.conv4_x = self._make_layer(BottleNeckV2, 256, 6, stride=2)

        if output_stride == 32:
            self.conv5_x = self._make_layer(BottleNeckV2, 512, 3, stride=2)
            self.out_channels = 2048
        else:
            self.conv5_x = None
            self.out_channels = 1024

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride
                )
            )
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.root(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)

        if self.output_stride == 32:
            x = self.conv5_x(x)

        return x
    
if __name__ == "__main__":
    model = ResNet50V2(output_stride=32)

    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    print("Output shape:", out.shape) # output_stride=16: torch.Size([2, 1024, 14, 14]) output_stride=32: torch.Size([2, 2048, 7, 7])
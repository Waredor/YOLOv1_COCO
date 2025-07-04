import torch.nn as nn
import torch
import utils

# Import Network Architecture
# net_architecture = utils.read_json('./config.json')
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.bn(self.conv(x)))

class YOLOv1(nn.Module):
    r"""
    This YOLOv1 model is used for object detection and multiclass classification.
    It has total 30 layers: 24 convolutional, 4 pooling and 2 full connected.
    Dimension of input tensor is Nx448x448x1, where N is batch size.
    Dimension of output tensor is NxSxSx(2*B+C), where N is batch size,
    S is number of cells in each dimension (height and width),
    B is number of bounding boxes in each cell
    and C is number of classes.

    Args:
        c (int, optional): number of classes for classification in dataset (default: ``3``).
        s (int, optional): number of cells on each dimension (height and width) (default: ``7``).
        b (int, optional): number of bounding boxes centers in each cell (default: ``2``).

    """
    def __init__(self, in_channels=1, **kwargs):
        super(YOLOv1, self).__init__()

        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.classifiers = self._create_classifiers(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.classifiers(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == tuple:
                layers += [CNNBlock(
                    in_channels=in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                )]
                in_channels = x[1]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                repeat = x[2]

                for _ in range(repeat):
                    layers += [
                        CNNBlock(
                            in_channels=in_channels, out_channels=conv1[1], kernel_size=conv1[0],
                            stride=conv1[2], padding=conv1[3]
                        )
                    ]

                    layers += [
                        CNNBlock(
                            in_channels=conv1[1], out_channels=conv2[1], kernel_size=conv2[0],
                            stride=conv2[2], padding=conv2[3]
                        )
                    ]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_classifiers(self, grid_size, num_boxes, num_classes):
        s, b, c = grid_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * s * s, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=496, out_features=s * s * (c + 5 * b))
        )

if __name__ == '__main__':
    pass
from torch import nn
import torch.nn.functional as F
import torch

class FLNet(nn.Module):
    def __init__(self):
        super(FLNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
## Note: This model is taken from McMahan et al. FL paper
class CIFAR_Net(nn.Module):
    def __init__(self):
        super(CIFAR_Net, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # MaxPooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)  # 256 channels, 4x4 feature map
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)  # 10 output classes for CIFAR-10

    def forward(self, x):
        # Apply convolutions, ReLU, and pooling
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 64, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 128, 8, 8]
        x = self.pool(F.relu(self.conv3(x)))  # [batch, 256, 4, 4]
        
        # Use reshape() instead of view() to flatten the tensor
        x = x.reshape(-1, 256 * 4 * 4)  # [batch, 4096]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here since this is the output layer
        
        return x
    
class FMNIST_Net(nn.Module):
    def __init__(self):
        super(FMNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.reshape(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        out = F.relu(out)
        return out


class ResNet_FMNIST(nn.Module):
    def __init__(self):
        super(ResNet_FMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)

        # Use a dummy input to calculate the size of the feature map after conv layers
        self.flatten_size = self._get_flatten_size()

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 10)

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels, 1)
        )

    # Function to calculate the size of the flattened feature map
    def _get_flatten_size(self):
        with torch.no_grad():
            # Create a dummy input tensor of size (1, 1, 28, 28) for FMNIST (grayscale images)
            dummy_input = torch.zeros(1, 1, 28, 28)
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = self.layer1(x)
            x = self.layer2(x)
            return x.view(1, -1).size(1)  # Flatten the tensor and return its size

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        features = [x]  # Initial input as the first feature
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))  # Concatenate previous features along the channel dimension
            features.append(new_features)
        return torch.cat(features, 1)  # Final concatenation

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)

class DenseNet_FMNIST(nn.Module):
    def __init__(self, growth_rate=12, num_layers_per_block=4, num_classes=10):
        super(DenseNet_FMNIST, self).__init__()
        self.growth_rate = growth_rate

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(1, 2 * growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        # Dense blocks and transition layers
        self.block1 = DenseBlock(2 * growth_rate, growth_rate, num_layers_per_block)
        self.trans1 = TransitionLayer(2 * growth_rate + num_layers_per_block * growth_rate, 128)

        self.block2 = DenseBlock(128, growth_rate, num_layers_per_block)
        self.trans2 = TransitionLayer(128 + num_layers_per_block * growth_rate, 128)

        # Final batch norm
        self.bn = nn.BatchNorm2d(128)

        # Fully connected layer
        self.fc = nn.Linear(128 * 7 * 7, 512)
        self.out = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.trans1(x)

        x = self.block2(x)
        x = self.trans2(x)

        x = self.bn(x)

        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.out(x)
        return x
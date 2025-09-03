import torch
import torch.nn as nn
import torchvision.transforms as transforms

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x


# Define a BasicBlock for ResNet without BatchNorm, using Tanh
class BasicBlockNoBatchNorm(nn.Module):
    expansion = 1  # No change in expansion for ResNet18 and ResNet32

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockNoBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.tanh = nn.Tanh()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        out = self.tanh(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.tanh(out)
        return out
        
# Define ResNet18NoBatchNorm for BloodMNIST
class ResNet18Blood(nn.Module):
    def __init__(self, num_classes=8):  # Default for BloodMNIST
        super(ResNet18Blood, self).__init__()
        self.in_planes = 8  # Reduced from 32 to 8

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True)  # Reduced from 32 to 8
        self.tanh = nn.Tanh()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers with reduced channels
        self.layer1 = self._make_layer(BasicBlockNoBatchNorm, 8, 2, stride=1)   # Reduced from 32 to 8
        self.layer2 = self._make_layer(BasicBlockNoBatchNorm, 16, 2, stride=2)  # Reduced from 64 to 16
        self.layer3 = self._make_layer(BasicBlockNoBatchNorm, 32, 2, stride=2)  # Reduced from 128 to 32
        self.layer4 = self._make_layer(BasicBlockNoBatchNorm, 64, 2, stride=2)  # Reduced from 256 to 64

        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlockNoBatchNorm.expansion, num_classes)  # Reduced from 256 to 64

        # Initialize weights with Xavier initialization for Tanh
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicBlockNoBatchNormRelu(nn.Module):
    expansion = 1  # No change in expansion for ResNet18 and ResNet32

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockNoBatchNormRelu, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.relu = nn.ReLU()  # Changed from Tanh to ReLU
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        out = self.relu(self.conv1(x))  # Changed from Tanh to ReLU
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)  # Changed from Tanh to ReLU
        return out
        
# Define ResNet32NoBatchNorm for CIFAR-10
class ResNet32NoBatchNorm(nn.Module):
    def __init__(self, num_classes=10):  # Default for CIFAR-10
        super(ResNet32NoBatchNorm, self).__init__()
        self.in_planes = 32  # Starting with 16 channels

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()  # Changed from Tanh to ReLU

        # ResNet layers with 5 blocks per layer
        self.layer1 = self._make_layer(BasicBlockNoBatchNormRelu, 32, 5, stride=1)
        self.layer2 = self._make_layer(BasicBlockNoBatchNormRelu, 64, 5, stride=2)
        self.layer3 = self._make_layer(BasicBlockNoBatchNormRelu, 128, 5, stride=2)

        # Global max pooling and fully connected layer
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))  # Changed from AdaptiveAvgPool2d to AdaptiveMaxPool2d
        self.fc = nn.Linear(128 * BasicBlockNoBatchNorm.expansion, num_classes)

        # Initialize weights with Xavier initialization
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)  # Changed from Tanh to ReLU

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.maxpool(x)  # Changed from avgpool to maxpool
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class VGG11Light(nn.Module):
    def __init__(self, num_classes=11):
        super(VGG11Light, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 1 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        
# Check number of parameters
model = LeNet5()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total LeNet5 parameters: {total_params}")     

model = ResNet18Blood()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total ResNet18Blood parameters: {total_params}")   

model = VGG11Light()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total VGG11Light parameters: {total_params}")

model = ResNet32NoBatchNorm()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total ResNet32NoBatchNorm parameters: {total_params}")
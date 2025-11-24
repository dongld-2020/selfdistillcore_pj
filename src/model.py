#model.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        

        self.feature = nn.Sequential(

            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),

            nn.GroupNorm(num_groups=2, num_channels=6),
            #nn.BatchNorm2d(6),            
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            

            nn.Conv2d(6, 16, kernel_size=5, stride=1),

            nn.GroupNorm(num_groups=4, num_channels=16),
            #nn.BatchNorm2d(16),            
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        

        FLAT_SIZE = 16 * 5 * 5 


        self.classifier = nn.Sequential(
            nn.Flatten(),
            

            nn.Linear(FLAT_SIZE, 120),

            nn.GroupNorm(num_groups=1, num_channels=120), # Layer Norm cho 1D
            #nn.BatchNorm1d(120),
            nn.Tanh(),
            

            nn.Linear(120, 84),
            nn.GroupNorm(num_groups=1, num_channels=84),  # Layer Norm cho 1D
            #nn.BatchNorm1d(84),
            nn.Tanh(),
            
            # Lớp Output: (84 -> 10)
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x
        


# Define a BasicBlock for ResNet with GroupNorm, using ReLU
class BasicBlockGroupNorm(nn.Module):
    expansion = 1  
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockGroupNorm, self).__init__()
        

        num_groups = 8
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.gn1 = nn.GroupNorm(num_groups, planes)  
        self.relu = nn.ReLU()  
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn2 = nn.GroupNorm(num_groups, planes)  
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
                nn.GroupNorm(num_groups, self.expansion * planes)  
            )

    def forward(self, x):

        out = self.relu(self.gn1(self.conv1(x)))

        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# BasicBlock with GroupNorm
class BasicBlockWithGroupNorm(nn.Module):
    expansion = 1  

    def __init__(self, in_planes, planes, stride=1, num_groups=16):
        super(BasicBlockWithGroupNorm, self).__init__()
        

        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, planes)  
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, planes)  

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, self.expansion * planes)  
            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += identity  
        out = self.relu(out)
        return out

# ResNet32 with GroupNorm
class ResNet32WithGroupNorm(nn.Module):
    def __init__(self, num_classes=10, num_groups=32):  
        super(ResNet32WithGroupNorm, self).__init__()
        self.in_planes = 32
        self.num_groups = num_groups  


        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(self.num_groups, 32)  
        self.relu = nn.ReLU()

        
        self.layer1 = self._make_layer(BasicBlockWithGroupNorm, 32, 5, stride=1)
        self.layer2 = self._make_layer(BasicBlockWithGroupNorm, 64, 5, stride=2)
        self.layer3 = self._make_layer(BasicBlockWithGroupNorm, 128, 5, stride=2)


        self.global_pool=nn.AvgPool2d(kernel_size=8, stride=8)
        self.fc = nn.Linear(128 * BasicBlockWithGroupNorm.expansion, num_classes)


        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:

            layers.append(block(self.in_planes, planes, stride, num_groups=self.num_groups))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):

                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)  
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ResNet-20 with GroupNorm

class ResNet20WithGroupNorm(nn.Module):
    def __init__(self, num_classes=8, num_groups=16):
        super(ResNet20WithGroupNorm, self).__init__()
        self.in_planes = 32
        self.num_groups = num_groups

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(self.num_groups, 32)
        self.relu = nn.ReLU()


        self.layer1 = self._make_layer(BasicBlockWithGroupNorm, 32, 3, stride=1) 
        self.layer2 = self._make_layer(BasicBlockWithGroupNorm, 64, 3, stride=2) 
        self.layer3 = self._make_layer(BasicBlockWithGroupNorm, 128, 3, stride=2)

        self.maxpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * BasicBlockWithGroupNorm.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, num_groups=self.num_groups))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        


class VGG11Light(nn.Module):

    def __init__(self, num_classes=11, num_groups=8):
        """
        Khởi tạo model VGG11-Light với Group Normalization.
        
        Args:
            num_classes (int): Số lượng lớp đầu ra.
            num_groups (int): Số lượng nhóm cho GroupNorm. 
                               Giá trị này phải chia hết số channels (32, 64, 128).
                               Giá trị 8 là một lựa chọn an toàn.
        """
        super(VGG11Light, self).__init__()
        

        assert 32 % num_groups == 0, "num_groups must be a divisor of 32"
        assert 64 % num_groups == 0, "num_groups must be a divisor of 64"
        assert 128 % num_groups == 0, "num_groups must be a divisor of 128"

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),  
            nn.GroupNorm(num_groups, 32),                           
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), 
            nn.GroupNorm(num_groups, 64),                           
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, 128),                          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, 128),                          
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

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
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
        self.fc = nn.Linear(128 * BasicBlockNoBatchNormRelu.expansion, num_classes)

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
        
# Check number of parameters
"""
model = LeNet5()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total LeNet5 parameters: {total_params}")     

model = ResNet20WithGroupNorm()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total ResNet20WithGroupNorm parameters: {total_params}")   

model = VGG11Light()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total VGG11Light parameters: {total_params}")

model = ResNet32WithGroupNorm()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total ResNe32WithGroupNorm parameters: {total_params}")
"""
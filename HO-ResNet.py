import torch.nn.functional as func
import torch
import torch.nn as nn
import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

"""
Use Euler method, which is the stand ResNet
1 block = 2layers para and flops
"""

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            func.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):

        out = self.conv2(func.relu(self.bn2(self.conv1(func.relu(self.bn1(x))))))
        out += self.shortcut(x)
        out = func.relu(out)

        return out

"""
Use MidPoint method, shall have half blocks number Euler does
1 block = 4layers para and flops
"""

class MidBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(MidBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            func.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                     "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):

        out = self.conv2(func.relu(self.bn2(self.conv1(func.relu(self.bn1(x))))))
        out = self.conv4(func.relu(self.bn4(self.conv3(func.relu(self.bn3(0.5 * out + self.shortcut(x)))))))

        out += self.shortcut(x)
        out = func.relu(out)

        return out


"""
Use Improved Euler method, shall have half blocks number Euler does
1 block = 4layers para and flops
"""

class ImprovedEuler(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(ImprovedEuler, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            func.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                     "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        outx = 0 * x
        out = self.conv2(func.relu(self.bn2(self.conv1(func.relu(self.bn1(x))))))
        outx += 0.5 * out
        out = self.conv4(func.relu(self.bn4(self.conv3(func.relu(self.bn3(0.5 * out + self.shortcut(x)))))))
        outx += 0.5 * out

        outx += self.shortcut(x)
        out = func.relu(outx)

        return out

"""
Use RK2 method, shall have half blocks number Euler does
1 block = 4layers para and flops
"""

class RK2Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(RK2Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            func.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                     "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        outx = 0 * x
        out = self.conv2(func.relu(self.bn2(self.conv1(func.relu(self.bn1(x))))))
        outx += 0.25 * out
        out = self.conv4(func.relu(self.bn4(self.conv3(func.relu(self.bn3(0.666666 * out + self.shortcut(x)))))))
        outx += 0.75 * out

        outx += self.shortcut(x)
        out = func.relu(outx)

        return out


"""
Use Heun3 method, shall have 1/3. blocks number Euler does
1 block = 6layers para and flops
"""

class Heun3Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(Heun3Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            func.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                     "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        outx = 0 * x
        out = self.conv2(func.relu(self.bn2(self.conv1(func.relu(self.bn1(x))))))
        outx += 0.25 * out
        out = self.conv4(func.relu(self.bn4(self.conv3(func.relu(self.bn3(0.333333 * out + self.shortcut(x)))))))
        out = self.conv6(func.relu(self.bn6(self.conv5(func.relu(self.bn5(0.666666 * out + self.shortcut(x)))))))
        outx += 0.75 * out

        outx += self.shortcut(x)
        out = func.relu(outx)

        return out


"""
Use RK3 method, shall have 1/3. blocks number Euler does
1 block = 6layers para and flops
"""


class RK3Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(RK3Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            func.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                     "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        outx = 0 * x
        k1 = self.conv2(func.relu(self.bn2(self.conv1(func.relu(self.bn1(x))))))
        outx += 1/6. * k1
        k2 = self.conv4(func.relu(self.bn4(self.conv3(func.relu(self.bn3(0.5 * k1 + self.shortcut(x)))))))
        outx += 2/3. * k2
        k3 = self.conv6(func.relu(self.bn6(self.conv5(func.relu(self.bn5(2*k2-k1 + self.shortcut(x)))))))
        outx += 1/6. * k3

        outx += self.shortcut(x)
        out = func.relu(outx)

        return out

"""
Use RK4 method, shall have 1/4 blocks number Euler does
1 block = 8layers para and flops
"""

class RK4Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(RK4Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(planes)
        self.conv7 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(planes)
        self.conv8 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            func.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                     "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):

        out = self.conv2(func.relu(self.bn2(self.conv1(func.relu(self.bn1(x))))))
        outx = 1 / 6. * out
        out = self.conv4(func.relu(self.bn4(self.conv3(func.relu(self.bn3(0.5 * out + self.shortcut(x)))))))
        outx += 1 / 3. * out
        out = self.conv6(func.relu(self.bn6(self.conv5(func.relu(self.bn5(0.5 * out + self.shortcut(x)))))))
        outx += 1 / 3. * out
        out = self.conv8(func.relu(self.bn8(self.conv7(func.relu(self.bn7(out + self.shortcut(x)))))))
        outx = 1 / 6. * out
        out = outx + self.shortcut(x)
        out = func.relu(out)

        return out

"""
Use Gill 4 method, shall have 1/4 blocks number Euler does
1 block = 8layers para and flops
"""

class Gill4Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(Gill4Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(planes)
        self.conv7 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(planes)
        self.conv8 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            func.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                     "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):

        k1 = self.conv2(func.relu(self.bn2(self.conv1(func.relu(self.bn1(x))))))
        outx = 1 / 6. * k1
        k2 = self.conv4(func.relu(self.bn4(self.conv3(func.relu(self.bn3(0.5 * k1 + self.shortcut(x)))))))
        outx += 0.097631 * k2
        k3 = self.conv6(func.relu(self.bn6(self.conv5(func.relu(self.bn5(0.2071 * k1 + 0.29289 * k2 + self.shortcut(x)))))))
        outx += 0.569 * k3
        out = self.conv8(func.relu(self.bn8(self.conv7(func.relu(self.bn7(1.7071 * k3 - 0.7071 * k2 + self.shortcut(x)))))))
        outx = 1 / 6. * out
        out = outx + self.shortcut(x)
        out = func.relu(out)

        return out


"""
Use Kutta-Nystrom 5-6 method, shall have 1/6 blocks number Euler does
1 block = 12layers para and flops
"""

class KuttaNys56Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(KuttaNys56Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(planes)
        self.conv7 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(planes)
        self.conv8 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(planes)

        self.conv9 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(planes)
        self.conv10 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(planes)
        self.conv11 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(planes)
        self.conv12 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(planes)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            func.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                     "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):

        k1 = self.conv2(func.relu(self.bn2(self.conv1(func.relu(self.bn1(x))))))
        k2 = self.conv4(func.relu(self.bn4(self.conv3(func.relu(self.bn3(1/3. * k1 + self.shortcut(x)))))))
        k3 = self.conv6(func.relu(self.bn6(self.conv5(func.relu(self.bn5(1/25. * (4 * k1 + 6 * k2) + self.shortcut(x)))))))
        k4 = self.conv8(func.relu(self.bn8(self.conv7(func.relu(self.bn7(1/4. * (k1 - 12 * k2 + 15 * k3) + self.shortcut(x)))))))
        k5 = self.conv10(func.relu(self.bn10(self.conv9(func.relu(self.bn9(1/81. * (6 * k1 + 90 * k2 + - 50 * k3 + 8 * k4) + self.shortcut(x)))))))
        k6 = self.conv12(func.relu(self.bn12(self.conv11(func.relu(self.bn11(1/75. *(6 * k1 + 36 * k2 + 10 * k3 + 8 * k4) + self.shortcut(x)))))))
        out = 1/192. * (23 * k1 + 125 * k2 - 81 * k5 + 125 * k6)
        out = out + self.shortcut(x)
        out = func.relu(out)

        return out


"""
Use Huta 6-8 method, shall have 1/8 blocks number Euler does
1 block = 16layers para and flops
"""


class Huta68Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(Huta68Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(planes)
        self.conv7 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(planes)
        self.conv8 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(planes)

        self.conv9 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(planes)
        self.conv10 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(planes)
        self.conv11 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(planes)
        self.conv12 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(planes)
        self.conv13 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(planes)
        self.conv14 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(planes)
        self.conv15 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(planes)
        self.conv16 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(planes)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            func.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                     "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):

        k1 = self.conv2(func.relu(self.bn2(self.conv1(func.relu(self.bn1(x))))))
        k2 = self.conv4(func.relu(self.bn4(self.conv3(func.relu(self.bn3(1/9. * k1 + self.shortcut(x)))))))
        k3 = self.conv6(func.relu(self.bn6(self.conv5(func.relu(self.bn5(1/24. * (k1 + 3 * k2) + self.shortcut(x)))))))
        k4 = self.conv8(func.relu(self.bn8(self.conv7(func.relu(
            self.bn7(1/6. * (k1 - 3 * k2 + 4 * k3) + self.shortcut(x)))))))
        k5 = self.conv10(func.relu(self.bn10(self.conv9(func.relu(
            self.bn9(1/8. * (-5 * k1 + 27 * k2 - 24 * k3 + 6 * k4) + self.shortcut(x)))))))
        k6 = self.conv12(func.relu(self.bn12(self.conv11(func.relu(
            self.bn11(1/9. * (221 * k1 - 981 * k2 + 867 * k3 - 102*k4 + k5) + self.shortcut(x)))))))
        k7 = self.conv14(func.relu(self.bn14(self.conv13(func.relu(
            self.bn13(1/48. *(-183 * k1 + 678 * k2 - 472 * k3 -66 * k4 +80 * k5 + 3*k6) + self.shortcut(x)))))))
        k8 = self.conv16(func.relu(self.bn16(self.conv15(func.relu(
            self.bn15(1/82. * (716 * k1 - 2079 * k2 + 1002 * k3 + 834 * k4 -454 * k5 - 9*k6 + 72 * k7) + self.shortcut(x)))))))
        out = 1/840.*(41*k1+216*k3 +24*k4+ 272*k5 + 27*k6+ 216*k7+41*k8)
        # 8
        out = out + self.shortcut(x)
        out = func.relu(out)

        return out

"""
Use RK-Fehlberg 6-8 method, shall have 1/8 blocks number Euler does
1 block = 16layers para and flops
"""


class RKFehlberg(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(RKFehlberg, self).__init__()
        self.h = 1
        self.e = 9999
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(planes)
        self.conv7 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(planes)
        self.conv8 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(planes)

        self.conv9 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(planes)
        self.conv10 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(planes)
        self.conv11 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(planes)
        self.conv12 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(planes)
        self.conv13 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(planes)
        self.conv14 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(planes)
        self.conv15 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(planes)
        self.conv16 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(planes)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            func.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                     "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):

        k1 = self.conv2(func.relu(self.bn2(self.conv1(func.relu(self.bn1(x))))))
        k2 = self.conv4(func.relu(self.bn4(self.conv3(func.relu(self.bn3(1/4. * k1 + self.shortcut(x)))))))
        k3 = self.conv6(func.relu(self.bn6(self.conv5(func.relu(self.bn5(3/32. * (k1 + 3 * k2) + self.shortcut(x)))))))
        k4 = self.conv8(func.relu(self.bn8(self.conv7(func.relu(
            self.bn7((1932/2197*k1 - 7200/2197 * k2 + 7296/32 * k3) + self.shortcut(x)))))))
        k5 = self.conv10(func.relu(self.bn10(self.conv9(func.relu(
            self.bn9((439/216. * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4) + self.shortcut(x)))))))
        k6 = self.conv12(func.relu(self.bn12(self.conv11(func.relu(
            self.bn11((-8/27 * k1 - 2 * k2 + 3544/2565 * k3 + 1859/4194*k4 - 11/40 * k5) + self.shortcut(x)))))))
        k7 = self.conv14(func.relu(self.bn14(self.conv13(func.relu(
            self.bn13(1/48. *(-183 * k1 + 678 * k2 - 472 * k3 -66 * k4 +80 * k5 + 3*k6) + self.shortcut(x)))))))
        k8 = self.conv16(func.relu(self.bn16(self.conv15(func.relu(
            self.bn15(1/82. * (716 * k1 - 2079 * k2 + 1002 * k3 + 834 * k4 -454 * k5 - 9*k6 + 72 * k7) + self.shortcut(x)))))))
        y1 = x + 25/216 * k1 + 1408/2565 * k3 + 2197/4104*k4 - 1/5*k5
        y2 = x + 1/360 * k1 - 128/4275 * k3 + 2197/75240*k4 + 1/50*k5 + 2/55 * k6
        y2 = y2.abs()
        self.e = y2
        q = 0.84 * (self.h * self.e/y2)^0.25
        if y2/self.h > self.e:
            self.h *= q
        out = self.h * (1 / 840. * (41 * k1 + 216 * k3 + 24 * k4 + 272 * k5 + 27 * k6 + 216 * k7 + 41 * k8))

        # 8
        out = out + self.shortcut(x)
        out = func.relu(out)

        return out


"""
Use Verner 8-9 method, shall have 1/14 blocks number Euler does
1 block = 28layers para and flops
"""

class Verner89Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(Verner89Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(planes)
        self.conv7 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(planes)
        self.conv8 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(planes)

        self.conv9 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(planes)
        self.conv10 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(planes)
        self.conv11 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(planes)
        self.conv12 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(planes)
        self.conv13 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(planes)
        self.conv14 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(planes)
        self.conv15 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(planes)
        self.conv16 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(planes)


        self.conv17 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(planes)
        self.conv18 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18 = nn.BatchNorm2d(planes)
        self.conv19 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn19 = nn.BatchNorm2d(planes)
        self.conv20 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(planes)
        self.conv21 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn21 = nn.BatchNorm2d(planes)
        self.conv22 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(planes)
        self.conv23 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(planes)
        self.conv24 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn24 = nn.BatchNorm2d(planes)

        self.conv25 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn25 = nn.BatchNorm2d(planes)
        self.conv26 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.BatchNorm2d(planes)
        self.conv27 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn27 = nn.BatchNorm2d(planes)
        self.conv28 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn28 = nn.BatchNorm2d(planes)
        self.conv29 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn29 = nn.BatchNorm2d(planes)
        self.conv30 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn30 = nn.BatchNorm2d(planes)
        self.conv31 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(planes)
        self.conv32 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            func.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                     "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):

        k1 = self.conv2(func.relu(self.bn2(self.conv1(func.relu(self.bn1(x))))))
        outx = 103 / 1680. * k1
        k2 = self.conv4(func.relu(self.bn4(self.conv3(func.relu(self.bn3(1/12. * k1 + self.shortcut(x)))))))
        k3 = self.conv6(func.relu(self.bn6(self.conv5(func.relu(self.bn5(1/27. * (k1 + 2 * k2) + self.shortcut(x)))))))
        k4 = self.conv8(func.relu(self.bn8(self.conv7(func.relu(self.bn7(1/24. * (k1 + 3 * k3) + self.shortcut(x)))))))
        k5 = self.conv10(func.relu(self.bn10(self.conv9(func.relu(self.bn9(1/375. * (234.25203582132 * k1 - 899.27141518056 * k3 + 837.49386649824 * k4) + self.shortcut(x)))))))
        k6 = self.conv12(func.relu(self.bn12(self.conv11(func.relu(self.bn11((0.053333333333 * k1 + 0.2739534538729544 * k4 + 0.24567579393091227 * k5) + self.shortcut(x)))))))
        k7 = self.conv14(func.relu(self.bn14(self.conv13(func.relu(self.bn13((0.06162164740427197 * k1 + 0.1815318224097963 * k4 - 0.013477689611 * k5 + 0.007024903611899742 * k6) + self.shortcut(x)))))))
        k8 = self.conv16(func.relu(self.bn16(self.conv15(func.relu(self.bn15(1/54. * (4*k1+ 13.550510257220001 * k6 + 18.44948974278* k7) + self.shortcut(x)))))))
        outx -= -27 / 140. * k8
        #8
        k9 = self.conv18(func.relu(self.bn18(self.conv17(func.relu(self.bn17(1/512 * (38*k1 + 61.66173591606*k6 + 174.33826408394 * k7 - 18 * k8) + self.shortcut(x)))))))
        outx += 76 / 105. * k9
        k10 = self.conv20(func.relu(self.bn20(self.conv19(func.relu(self.bn19(11/144. * k1 + 0.30503531279770835 * k6 + 0.31070542794303235 * k7 - 1/16. * k8 -8/27. * k9 + self.shortcut(x)))))))
        outx -= 201 / 280. * k10
        k11 = self.conv22(func.relu(self.bn22(self.conv21(func.relu(self.bn21(0.07112936653168327*k1 + 0.37852828889059764 * k7 - 0.01174633003514941 * k8 + 0.07272054197227078*k9 - 0.26063186735940236* k10 + self.shortcut(x)))))))
        outx += 1024 / 1365. * k11
        k12 = self.conv24(func.relu(self.bn24(self.conv23(func.relu(self.bn23(-8.141639713845233 * k1 -574.4363925621823 * k6 + 847.8814814814815 * k7 + 113.71920186905155 * k8 + 626.9414848959715* k9 + 605.7315968367965 * k10 -328.69135802469134 * k11 + self.shortcut(x)))))))
        outx += 3 / 7280. * k12

        k13 = self.conv26(func.relu(self.bn26(self.conv25(func.relu(self.bn25(0.0878037592818966 * k1+0.6933735017296832*k6-1.9030978898036277*k7 + 0.22886338868515282*k8 -0.6904282483623702*k9 -0.07691188807394458* k10 +2624/ 1053.*k11 +3/1664.*k12 + self.shortcut(x)))))))
        outx += 12 / 35. * k13
        k14 = self.conv28(func.relu(self.bn28(self.conv27(func.relu(self.bn27(-137/1296.*k1 + 5.5746781906054865*k6 + 7.485506994579699 * k7 - 299/48. * k8 + 184/81. * k9 -44/9.* k10-5120/1053.*k11 - 11/468.* k12 +16/9.* k13 + self.shortcut(x)))))))
        outx += 9 / 280. * k14

        out = outx + self.shortcut(x)
        out = func.relu(out)

        return out


class CFResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(CFResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = func.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = func.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    net = CFResNet(Gill4Block, [1, 1, 1], num_classes=10)
    a = torch.rand(7, 3, 32, 32)

    print(net(a).shape)

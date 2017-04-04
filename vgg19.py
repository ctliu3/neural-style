import torch.nn as nn
import torch.nn.functional as F


def _alias(id):
    return 'features.%s' % id


class VGG19Net(nn.Module):

    def __init__(self):
        super(VGG19Net, self).__init__()

        # conv: (input channel, output channel, height, width)
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_1_alias = _alias(0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_alias = _alias(2)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_alias = _alias(5)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_alias = _alias(7)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_alias = _alias(10)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_alias = _alias(12)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_alias = _alias(14)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_4_alias = _alias(16)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_alias = _alias(19)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_alias = _alias(21)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_alias = _alias(23)
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_4_alias = _alias(25)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_alias = _alias(28)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_alias = _alias(30)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_alias = _alias(32)
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_4_alias = _alias(34)

    def run_with_style(self):
        self.is_style = True
        self.is_content = False

    def run_with_content(self):
        self.is_style = False
        self.is_content = True

    def run_with_training(self):
        self.is_style = self.is_content = True

    def forward(self, x):
        is_style, is_content = self.is_style, self.is_content
        internals = {}

        x = F.relu(self.conv1_1(x))
        if is_style:
            internals['conv1_1'] = x
        x = F.max_pool2d(F.relu(self.conv1_2(x)), 2, stride=2)

        x = F.relu(self.conv2_1(x))
        if is_style:
            internals['conv2_1'] = x
        x = F.max_pool2d(F.relu(self.conv2_2(x)), 2, stride=2)

        x = F.relu(self.conv3_1(x))
        if is_style:
            internals['conv3_1'] = x
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.relu(self.conv3_4(x))
        x = F.max_pool2d(F.relu(self.conv3_4(x)), 2, stride=2)

        x = F.relu(self.conv4_1(x))
        if is_style:
            internals['conv4_1'] = x
        x = F.relu(self.conv4_2(x))
        if is_content:
            internals['conv4_2'] = x
        x = F.relu(self.conv4_3(x))
        x = F.relu(self.conv4_4(x))
        x = F.max_pool2d(F.relu(self.conv4_4(x)), 2, stride=2)

        x = F.relu(self.conv5_1(x))
        if is_style:
            internals['conv5_1'] = x
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.relu(self.conv5_4(x))
        x = F.max_pool2d(F.relu(self.conv5_4(x)), 2, stride=2)

        self.internals = internals
        return x

    def feature_name_map(self):
        d = {}
        for name, _ in self.named_children():
            alias = getattr(self, '%s_alias' % name)
            d.update({
                '%s.weight' % alias: '%s.weight' % name,
                '%s.bias' % alias: '%s.bias' % name,
            })
        return d

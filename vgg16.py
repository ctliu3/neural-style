import torch.nn as nn
import torch.nn.functional as F


class ForwardType(object):
    Style = 0
    Content = 1
    Train = 2


def _alias(id):
    return 'features.%s' % id


class VGG16Net(nn.Module):

    def __init__(self):
        super(VGG16Net, self).__init__()

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
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_alias = _alias(17)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_alias = _alias(19)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_alias = _alias(21)

    def forward(self, x, typ):
        if typ == ForwardType.Content:
            is_style, is_content = False, True
        elif typ == ForwardType.Style:
            is_style, is_content = True, False
        elif typ == ForwardType.Train:
            is_style, is_content = True, True
        else:
            raise Exception('Unknown forward type, {}'.format(typ))

        internals = {}

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        if is_style:
            internals['conv1_2'] = x
        x = F.max_pool2d(x, 2, stride=2)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        if is_style or is_content:
            internals['conv2_2'] = x
        x = F.max_pool2d(x, 2, stride=2)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        if is_style:
            internals['conv3_3'] = x
        x = F.max_pool2d(x, 2, stride=2)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        if is_style:
            internals['conv4_3'] = x

        return internals

    def feature_name_map(self):
        d = {}
        for name, _ in self.named_children():
            alias = getattr(self, '%s_alias' % name)
            d.update({
                '%s.weight' % alias: '%s.weight' % name,
                '%s.bias' % alias: '%s.bias' % name,
            })
        return d

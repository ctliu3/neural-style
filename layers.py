import torch
import torch.nn as nn
from torch.autograd import Variable


STYLE_LAYER_WEIGHTS = {
    'conv1_1': 0.2,
    'conv2_1': 0.2,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2,
}


class StyleLoss(nn.Module):

    def __init__(self, a_out_dict):
        super(StyleLoss, self).__init__()
        self.a_out_dict = a_out_dict
        self.loss_fn = torch.nn.MSELoss()
        self.A = {}
        for layer, out in a_out_dict.items():
            a_out = a_out_dict[layer].detach()
            self.A[layer] = self.gram_matrix(a_out)

    def gram_matrix(self, X):
        n, c, h, w = X.size()
        X_flatten = X.view(n, c, -1)
        return X_flatten.bmm(X_flatten.transpose(1, 2)) / (c * h * w)

    def forward(self, x_out_dict):
        loss = []
        for layer, A in self.A.items():
            x_out = x_out_dict[layer]

            G = self.gram_matrix(x_out)
            weight = STYLE_LAYER_WEIGHTS[layer]
            layer_loss = weight * self.loss_fn(G, A)
            loss.append(layer_loss)
        return sum(loss)


class ContentLoss(nn.Module):

    def __init__(self, p_out):
        super(ContentLoss, self).__init__()
        self.p_out = p_out
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x_out):
        return self.loss_fn(x_out, self.p_out)


# Fast neural style

VGG16_STYLE_LAYER_SET = {
    'conv1_2', 'conv2_2', 'conv3_3', 'conv4_3'
}


class FastStyleLoss(nn.Module):

    def __init__(self):
        super(FastStyleLoss, self).__init__()
        self.loss_fn = torch.nn.MSELoss()

    def gram_matrix(self, X):
        n, c, h, w = X.size()
        X_flatten = X.view(n, c, -1)
        return X_flatten.bmm(X_flatten.transpose(1, 2)) / (c * h * w)

    def forward(self, x_out_dict, a_out_dict):
        loss = []
        for layer, _ in a_out_dict.items():
            if layer not in VGG16_STYLE_LAYER_SET:
                continue
            x_out = x_out_dict[layer]
            a_out = a_out_dict[layer]
            a_out = Variable(a_out.data, requires_grad=False)

            G = self.gram_matrix(x_out)
            A = self.gram_matrix(a_out)

            loss.append(self.loss_fn(G, A))

        return sum(loss)


class FastFeatureLoss(nn.Module):

    def __init__(self):
        super(FastFeatureLoss, self).__init__()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x_out_dict, p_out_dict):
        x_out = x_out_dict['conv3_3']
        p_out = p_out_dict['conv3_3']
        p_out = Variable(p_out.data, requires_grad=False)

        _, c, h, w = x_out.size()
        return self.loss_fn(x_out, p_out) / (c * h * w)

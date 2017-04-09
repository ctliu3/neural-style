from __future__ import print_function
from PIL import Image
import argparse
import torch
from vgg19 import VGG19Net
from utils import load_param
from torch.autograd import Variable, Function
import torch.nn as nn
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='Style transfer in pytorch')
parser.add_argument('--style-image', type=str, default='images/starry_night_google.jpg',
                    help="style image file")
parser.add_argument('--content-image', type=str, default='images/hoovertowernight.jpg',
                    help="content image file")
parser.add_argument('--size', type=str, default='600,400',
                    help="size (width, height) to precess, i.e., 600,400")
parser.add_argument('--niter', type=int, default=100,
                    help="training iteration number")
parser.add_argument('--loss-alpha', type=float, default=100.0,
                    help="ratio in style loss")
parser.add_argument('--loss-beta', type=float, default=5.0,
                    help="ratio in content loss")
parser.add_argument('--use-cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('--loss-interval', type=int, default=100,
                    help='print loss in each fixed interval')
parser.add_argument('--snapshot-prefix', type=str, default='neural-style',
                    help="snapshot prefix")
parser.add_argument('--snap-interval', type=int, default=1000,
                    help='save model in each fixed interval')
args = parser.parse_args()


MODEL_PATH = './models/vgg19-dcbb9e9d.pth'
# R, G, B
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

args.use_cuda = args.use_cuda and torch.cuda.is_available()


def preprocess(image_path, size):
    # PIL reads image in RGB format
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    transformer = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    image = Image.open(image_path)
    image = image.resize(size)
    image = Variable(transformer(image), requires_grad=False)
    image = image.unsqueeze(0)
    return image


def postprocess(output):

    def denormalize(image):
        for t in range(3):
            image[t, :, :] = (image[t, :, :] * STD[t]) + MEAN[t]
        return image

    transformer = transforms.Compose([
        transforms.ToPILImage()])

    image = output.cpu().data[0]
    image = torch.clamp(denormalize(image), min=0, max=1)
    return transformer(image)


def gen_noise_image(content_image):
    out = torch.randn(content_image.size()).type(torch.FloatTensor)
    return out


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
            N = a_out.size(1)
            a_out = a_out.view(N, -1)
            self.A[layer] = torch.mm(a_out, a_out.t())

    def forward(self, x_out_dict):
        loss = []
        for layer, A in self.A.items():
            x_out = x_out_dict[layer]

            _, N, W, H = x_out.size()
            M = W * H
            x_out = x_out.view(N, -1)
            G = torch.mm(x_out, x_out.t())
            weight = STYLE_LAYER_WEIGHTS[layer]
            layer_loss = weight * (1. / (4 * M * M * N * N)) * self.loss_fn(G, A)
            loss.append(layer_loss)
        return sum(loss)


class ContentLoss(nn.Module):

    def __init__(self, p_out):
        super(ContentLoss, self).__init__()
        self.p_out = p_out
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x_out):
        return self.loss_fn(x_out, self.p_out)


def main():
    width, height = map(int, args.size.split(','))
    size = (width, height)

    # Read ipunt image
    a = preprocess(args.style_image, size)
    p = preprocess(args.content_image, size)
    noise_image = p.data.clone()
    # noise_image = gen_noise_image(a)

    net = VGG19Net()
    load_param(net, MODEL_PATH)
    for param in net.parameters():
        param.requires_grad = False

    if args.use_cuda:
        net.cuda()
        a = a.cuda()
        p = p.cuda()
        noise_image = noise_image.cuda()

    x = Variable(noise_image, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=1e-2)

    net.run_with_style()
    net.forward(a)
    a_out_dict = net.internals

    net.run_with_content()
    net.forward(p)
    p_out_dict = net.internals

    style_loss = StyleLoss(a_out_dict)
    content_loss = ContentLoss(p_out_dict['conv4_2'].detach())
    for t in range(args.niter):
        net.run_with_training()
        net.forward(x)
        x_out_dict = net.internals

        loss1 = style_loss.forward(x_out_dict) * args.loss_beta
        loss2 = content_loss.forward(x_out_dict['conv4_2']) * args.loss_alpha
        loss = loss1 + loss2

        if t % args.loss_interval == 0:
            print('niter:%s, style loss: %s, content loss: %s' % (
                  t, loss1.data[0], loss2.data[0]))

        optimizer.zero_grad()
        loss.backward(retain_variables=True)

        optimizer.step()

        if t % args.snap_interval == 0:
            image = postprocess(x.clone())
            image.save('snap/{}_{}.jpg'.format(args.snapshot_prefix, t))


if __name__ == '__main__':
    main()

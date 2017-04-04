from __future__ import print_function
from PIL import Image
import argparse
import torch
from vgg19 import VGG19Net
from utils import load_param
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np


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
parser.add_argument('--use-cuda', action='store_true', default=True,
                    help='enables CUDA training')
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
    'conv1_1': 1.0,
    'conv2_1': 1.0,
    'conv3_1': 1.0,
    'conv4_1': 1.5,
    'conv5_1': 2.0,
}


class ContentLoss(nn.Module):
    pass


def get_style_loss(a_dict, x_dict):
    # loss = []
    loss_fn = torch.nn.MSELoss()
    for layer, out in a_dict.items():
        a_out = a_dict[layer].detach()
        x_out = x_dict[layer]
        weight = STYLE_LAYER_WEIGHTS[layer]

        _, N, W, H = a_out.size()
        M = W * H
        a_out = a_out.view(N, -1)
        x_out = x_out.view(N, -1)
        G = torch.mm(a_out, a_out.t())
        A = torch.mm(x_out, x_out.t())
        # G.mul_(255.)
        # A.mul_(255.)
        # G.div_(4 * N * M)
        # A.div_(4 * N * M)
        loss = args.loss_beta * weight * (1. / (4*M*M*N*N)) * loss_fn(A, G)
        # loss = args.loss_beta * 1 * loss_fn(A, G)
        # print(loss.data[0])
        loss.backward(retain_variables=True)
    # print([l.data[0] for l in loss])
    # return sum(loss)


def get_content_loss(p_dict, x_dict):
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(x_dict['conv4_2'], p_dict['conv4_2'].detach())
    loss.backward()
    # return loss


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

    for t in range(args.niter):

        net.run_with_training()
        net.forward(x)
        x_out_dict = net.internals

        optimizer.zero_grad()

        get_style_loss(a_out_dict, x_out_dict)
        get_content_loss(p_out_dict, x_out_dict)

        # style_loss = get_style_loss(a_out_dict, x_out_dict)
        # content_loss = get_content_loss(p_out_dict, x_out_dict)
        # loss = style_loss * args.loss_alpha + content_loss * args.loss_beta
        print(t)

        # loss.backward()
        optimizer.step()

    image = postprocess(x)
    image.save('after.jpg')


if __name__ == '__main__':
    main()

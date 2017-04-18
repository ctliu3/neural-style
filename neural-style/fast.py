from __future__ import print_function

import argparse
import torch
import logging
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.autograd import Variable

from image_transform_net import ImageTransformNet
from image_process import preprocess_caffe, postprocess_caffe, sub_mean_caffe
from layers import FastFeatureLoss, FastStyleLoss
from opts import add_common_args
from vgg16 import VGG16Net, ForwardType
from utils import load_param


FORMAT = "%(asctime)s %(levelname)s %(process)d %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, filename=None)

parser = argparse.ArgumentParser(description="fast neural style",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_common_args(parser)
parser.add_argument('--dataset', type=str,
                    help="training image directory")
parser.add_argument('--batch-size', type=int, default=100,
                    help="batch size in each iteration")
parser.add_argument('--epoch', type=int, default=10,
                    help="training epoch")
parser.add_argument('--num-worker', type=int, default=5,
                    help="number of worker to load images")
args = parser.parse_args()
use_cuda = args.use_cuda and torch.cuda.is_available()


def init_network():
    # Image transform network
    transform_net = ImageTransformNet()

    # Loss network
    vgg16_net = VGG16Net()
    load_param(vgg16_net, args.pretrained_model)
    for param in vgg16_net.parameters():
        param.requires_grad = False

    if use_cuda:
        transform_net.cuda()
        vgg16_net.cuda()

    return transform_net, vgg16_net


def main():
    width, height = map(int, args.size.split(','))
    size = (width, height)

    def process(x):
        x = x.resize(size)
        x = np.array(np.array(x)[..., ::-1])  # RGB -> BGR
        x = torch.from_numpy(x.transpose(2, 0, 1)).float()  # HWC -> CHW
        return x

    transform = transforms.Compose([transforms.Lambda(process)])
    dataset = ImageFolder(root=args.dataset, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             num_workers=5, shuffle=True)

    # Networks
    transform_net, vgg16_net = init_network()

    # Loss
    feature_loss = FastFeatureLoss()
    style_loss = FastStyleLoss()

    # Solver
    optimizer = torch.optim.Adam(transform_net.parameters(), lr=args.lr)

    a = preprocess_caffe(args.style_image, size, use_cuda)
    a = a.repeat(args.batch_size, 1, 1, 1)
    sub_mean_caffe(a)
    a_feat_dict = vgg16_net.forward(a, ForwardType.Style)

    # test image
    test_image = preprocess_caffe(args.content_image, size, use_cuda)
    sub_mean_caffe(test_image)

    for epoch in range(args.epoch):
        logging.info('#epoch: {}'.format(epoch))
        for index, (X, _) in enumerate(data_loader):
            if len(X) != args.batch_size:
                break

            p = Variable(X.clone(), volatile=True)
            if use_cuda:
                p = p.cuda()
            sub_mean_caffe(p)
            p_feat_dict = vgg16_net.forward(p, ForwardType.Content)

            X = Variable(X.clone(), requires_grad=True)
            if use_cuda:
                X = X.cuda()
            X_transform = transform_net.forward(X)
            sub_mean_caffe(X_transform)
            X_feat_dict = vgg16_net.forward(X_transform, ForwardType.Train)

            loss1 = feature_loss.forward(X_feat_dict, p_feat_dict) * args.loss_feature
            loss2 = style_loss.forward(X_feat_dict, a_feat_dict) * args.loss_style
            loss = loss1 + loss2

            if index % args.loss_interval == 0:
                logging.info('#epoch:{} #niter:{}, feature loss: {:.3f}, style loss: {:.3f}'.format(
                             epoch, index, loss1.data[0], loss2.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if index % args.styled_interval == 0:
                styled_image = transform_net(test_image)
                styled_image = postprocess_caffe(styled_image)
                styled_image.save('output/{}_epoch_{}_itr_{}.jpg'.format(
                                  args.styled_prefix, epoch, index))

            if index % args.snapshot_interval == 0:
                fn = 'snapshot/{}_epoch_{}_iter_{}.pth'.format(
                    args.snapshot_prefix,
                    epoch, index)
                torch.save(transform_net, fn)


if __name__ == '__main__':
    main()

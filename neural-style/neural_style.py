from __future__ import print_function

import logging
import argparse
import torch
from torch.autograd import Variable

from image_process import preprocess_caffe, postprocess_caffe, sub_mean_caffe
from layers import ContentLoss, StyleLoss
from vgg19 import VGG19Net, ForwardType
from opts import add_common_args
from utils import load_param


FORMAT = "%(asctime)s %(levelname)s %(process)d %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, filename=None)

parser = argparse.ArgumentParser(description="neural style",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_common_args(parser)
parser.add_argument('--niter', type=int, default=100,
                    help="training iteration number")
args = parser.parse_args()
use_cuda = args.use_cuda and torch.cuda.is_available()


def main():
    width, height = map(int, args.size.split(','))
    size = (width, height)

    # Read ipunt image
    a = preprocess_caffe(args.style_image, size)
    p = preprocess_caffe(args.content_image, size)
    init_image = p.data.clone()

    net = VGG19Net()
    load_param(net, args.pretrained_model)
    for param in net.parameters():
        param.requires_grad = False

    if args.use_cuda:
        net.cuda()
        a = a.cuda()
        p = p.cuda()
        init_image = init_image.cuda()

    x = Variable(init_image, requires_grad=True)
    sub_mean_caffe(x)

    sub_mean_caffe(a)
    a_out_dict = net.forward(a, ForwardType.Style)

    sub_mean_caffe(p)
    p_out_dict = net.forward(p, ForwardType.Content)

    optimizer = torch.optim.Adam([x], lr=args.lr)
    style_loss = StyleLoss(a_out_dict)
    content_loss = ContentLoss(p_out_dict['conv4_2'].detach())
    for itr in range(args.niter):
        x_out_dict = net.forward(x, ForwardType.Train)

        loss1 = style_loss.forward(x_out_dict) * args.loss_style
        loss2 = content_loss.forward(x_out_dict['conv4_2']) * args.loss_feature
        loss = loss1 + loss2

        if itr % args.loss_interval == 0:
            logging.info('#niter:%s, style loss: %s, content loss: %s' % (
                         itr, loss1.data[0], loss2.data[0]))

        optimizer.zero_grad()
        loss.backward(retain_variables=True)
        optimizer.step()

        if itr % args.styled_interval == 0:
            image = postprocess_caffe(x.clone())
            image.save('output/{}_itr_{}.jpg'.format(args.snapshot_prefix, itr))


if __name__ == '__main__':
    main()

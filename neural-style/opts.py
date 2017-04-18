from __future__ import print_function


def add_common_args(parser):
    parser = parser.add_argument_group('common', 'args')
    parser.add_argument('--style-image', type=str, default='images/starry_night_google.jpg',
                        help="style image file")
    parser.add_argument('--content-image', type=str, default='images/amsterdam_canal.jpg',
                        help="content image file")
    parser.add_argument('--size', type=str, default='600,400',
                        help="size (width, height) to precess, i.e., 600,400")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--loss-style', type=float, default=100.0,
                        help="ratio in style loss")
    parser.add_argument('--loss-feature', type=float, default=5.0,
                        help="ratio in feature loss")
    parser.add_argument('--use-cuda', action='store_true',
                        help='enables CUDA training')
    parser.add_argument('--pretrained-model', type=str,
                        help='pretrained model')

    parser.add_argument('--snapshot-prefix', type=str, default='neural-style',
                        help="snapshot prefix")
    parser.add_argument('--styled-prefix', type=str, default='neural-style',
                        help="styled image prefix")
    parser.add_argument('--styled-interval', type=int, default=1000,
                        help='get styled image in each fixed interval')

    parser.add_argument('--loss-interval', type=int, default=100,
                        help='print loss in each fixed interval')
    parser.add_argument('--snapshot-interval', type=int, default=1000,
                        help='save model in each fixed interval')

    parser.add_argument('--device-id', type=str, default='0',
                        help='device id used to train the model')
    return parser

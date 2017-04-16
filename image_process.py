from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable


# R, G, B for torch model
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# B, G, R for caffe model
CAFFE_MEAN_BGR = [103.939, 116.779, 123.68]


def sub_mean_caffe(x):
    mean = torch.FloatTensor(x.size())
    mean[:, 0, :, :] = CAFFE_MEAN_BGR[0]
    mean[:, 1, :, :] = CAFFE_MEAN_BGR[1]
    mean[:, 2, :, :] = CAFFE_MEAN_BGR[2]
    x.sub(Variable(mean.cuda(), requires_grad=False))


def _preprocess_caffe_tensor(image_path, size, use_cuda=False):
    image = Image.open(image_path)
    image = image.resize(size)

    image = np.array(np.array(image)[..., ::-1])
    image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    if use_cuda:
        image = image.cuda()
    return image


def preprocess_caffe(image_path, size, use_cuda=False, is_tensor=False):
    image = _preprocess_caffe_tensor(image_path, size, use_cuda)

    if is_tensor:
        return image

    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image


def postprocess_caffe(output):
    output = output.data.cpu().clamp(0, 255).numpy()
    output = output[0].transpose(1, 2, 0).astype('uint8')
    output = output[..., ::-1]
    output = Image.fromarray(output)
    return output


def preprocess_torch(image_path, size):
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


def postprocess_torch(output):

    # Should we?
    def denormalize(image):
        for t in range(3):
            image[t, :, :] = (image[t, :, :] * STD[t]) + MEAN[t]
        return image

    transformer = transforms.Compose([
        transforms.ToPILImage()])

    image = output.cpu().data[0]
    image = torch.clamp(denormalize(image), min=0, max=1)
    return transformer(image)

import torch


def load_param(net, model_path):
    model = torch.load(model_path)
    feature_dict = net.feature_name_map()
    pretrain_dict = {k: v for k, v in model.items() if k in feature_dict}
    net.load_state_dict({feature_dict[name]: param for name, param in pretrain_dict.items()})

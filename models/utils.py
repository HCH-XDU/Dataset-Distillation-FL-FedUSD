import pdb
import time
import logging
import torch
import torch.nn as nn
from models.networks import MLP, MLP_TINY, ConvNet, LeNet, AlexNet, VGG11BN, VGG11, ResNet18, ResNet18BN_AP, CharCNN, ViT_B_16, ViT_Tiny,ResNet50
from torchvision.models import resnet50,ResNet50_Weights
import optimizer_cust



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling

def get_default_charcnn_setting():
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    max_length, input_dim = 1014, len(alphabet)
    return max_length, input_dim


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


def get_network(model, channel, num_classes, im_size=(32, 32)):
    # torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
    max_length, input_dim = get_default_charcnn_setting()
    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'vit-b-16':
        net = ViT_B_16(num_classes=num_classes)
    elif model == 'vit-tiny':
        net = ViT_Tiny(class_num=num_classes)
    elif model == 'MLP_TINY':
        net = MLP_TINY(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet50':
        # net = ResNet50(num_classes=num_classes)
        net = ResNet50(channel=channel, num_classes=num_classes)
        # load_imagenet_pretrain_skip_classifier(net)
        # fill_shortcut_and_bn1(net)
        # state_dict = torch.load("E:\\project\\FedAF\\weight\\resnet50-19c8e357", map_location="cpu",weights_only=False)
        # net.load_state_dict(state_dict)  # load all parameters strictly
        # in_features = net.fc.in_features  # should be 2048
        # net.fc = nn.Linear(in_features, 200)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none')
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling')
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling')
    elif model == 'CharCNNSmall':
        net = CharCNN(input_length=max_length, n_classes=num_classes,
                      input_dim=input_dim,
                      n_conv_filters=256, n_fc_neurons=1024)
    elif model == 'CharCNNLarge':
        net = CharCNN(input_length=max_length, n_classes=num_classes,
                      input_dim=input_dim, 
                      n_conv_filters=1024, n_fc_neurons=2048)
    else:
        net = None
        exit('DC error: unknown model')

    gpu_num = torch.cuda.device_count()
    if gpu_num>0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net



def get_optimizer(parameters, opt_name, lr, weight_decay, rho, momentum):
    base_optimizer = optimizer_cust.SGD
    if opt_name == 'sgd':
        optimizer = base_optimizer(parameters, lr=lr, momentum=momentum)
    if opt_name == 'sam':
        optimizer = optimizer_cust.SAM(parameters, base_optimizer, rho=rho, adaptive=False,
                                       lr=lr, momentum=momentum, weight_decay=weight_decay)
    if opt_name == 'asam':
        optimizer = optimizer_cust.SAM(parameters, base_optimizer, rho=rho, adaptive=True,
                                       lr=lr, momentum=momentum, weight_decay=weight_decay)
    if opt_name == 'sam-rand':
        optimizer = optimizer_cust.SAM(parameters, base_optimizer, rho=rho, adaptive=False, rand=True,
                                       lr=lr, momentum=momentum, weight_decay=weight_decay)
    if opt_name == 'asam-rand':
        optimizer = optimizer_cust.SAM(parameters, base_optimizer, rho=rho, adaptive=True, rand=True,
                                       lr=lr, momentum=momentum, weight_decay=weight_decay)
    if opt_name == 'lbfgs':
        optimizer = torch.optim.LBFGS(parameters, lr=lr)
    optimizer.zero_grad()
    return optimizer



def get_lr_schedule(optimizer, opt_name, lr, Epoch):
    if opt_name == 'sgd':
        lr_schedule = optimizer_cust.StepLR_SGD(optimizer, lr=lr, total_epochs=Epoch)
    elif opt_name in ['sam', 'asam', 'sam-rand', 'asam-rand']:
        lr_schedule = optimizer_cust.StepLR_SAM(optimizer, lr=lr, total_epochs=Epoch)
    return lr_schedule
    


def copy_model(src, dst):
    for p, pp in zip(dst.parameters(), src.parameters()):
        p.data.copy_(pp.data)



def distance_wb(gwr, gws, dist_dropout=0):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0 ## TODO under dev ## here changed to tensor(0)
    if dist_dropout > 0:
        dout_mask = (torch.cuda.FloatTensor(size=gwr.shape).uniform_() > dist_dropout).float()
        gwr *= dout_mask
        gws *= dout_mask
    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis



def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).cuda()

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            try:
                dis += distance_wb(gwr, gws, dist_dropout=args.dist_dropout)
            except:
                dis += distance_wb(gwr, gws, dist_dropout=0)

    elif args.dis_metric == 'mse':
        assert not args.dist_dropout, 'not implemented for mse'
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        assert not args.dist_dropout, 'not implemented for cos'
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis



def theta_matching(net1, net2, args):
    net1_param = list(net1.parameters())
    net1_param_flat = torch.cat([p.flatten() for p in net1_param], dim=0)
    
    net2_param = list(net2.parameters())
    net2_param_flat = torch.cat([p.flatten() for p in net2_param], dim=0)

    l2 = (net1_param_flat - net2_param_flat).norm()
    dist = match_loss(net1_param, net2_param, args)
    norm1 = net1_param_flat.norm()
    norm2 = net2_param_flat.norm()
    return l2, dist, norm1, norm2



def compute_match_loss(imgs_real, labs_real, imgs_syn, labs_syn,
                        net, net_parameters, criterion, args, perturb=False):
    """ compute the gradient match loss for X, X~ """
    assert isinstance(net_parameters, list)
    output_real = net(imgs_real)
    loss_real = criterion(output_real, labs_real)
    if perturb:
        gw_real = torch.autograd.grad(loss_real, net_parameters, create_graph=True)
    else:
        gw_real = torch.autograd.grad(loss_real, net_parameters)
        gw_real = list((_.detach().clone() for _ in gw_real))
    
    output_syn = net(imgs_syn)
    loss_syn = criterion(output_syn, labs_syn)
    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
    loss = match_loss(gw_syn, gw_real, args)
    return loss



def cross_entropy_loss_cust(args):
    '''
    input: softmaxed output
    '''
    class xent:
        def __init__(self, args):
            self.logsoftmax = nn.LogSoftmax(dim=-1)
            self.torch_xent = nn.CrossEntropyLoss(reduction='mean')
            self.args = args
            try:
                self.normalize = (self.args.label == 'soft_norm')
                self.softmax   = (self.args.label == 'soft_sm')
            except:
                self.normalize = False
                self.softmax = False
        
        def __call__(self, output, target, weight=None):
            if len(target.shape) > 1: # soft-label
                if self.normalize:
                    target /= target.sum(dim=-1).view(-1, 1)
                if self.softmax:
                    target = torch.softmax(target, dim=-1)
                return torch.mean(torch.sum(- target * self.logsoftmax(output), 1))
            else:
                # all_loss = self.torch_xent(output, target)
                # print(all_loss)
                # weight /= weight.sum(dim=-1)
                # weight = weight
                # print(all_loss * weight)
                # return (all_loss * weight).sum()
                return self.torch_xent(output, target)

        def cuda(self):
            self.logsoftmax = self.logsoftmax.cuda()
            self.torch_xent = self.torch_xent.cuda()
            return self

    return xent(args)

# def soft_cross_entropy(pred, soft_targets):
#     """A method for calculating cross entropy with soft targets"""
#     logsoftmax = nn.LogSoftmax()
#     return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def tensor_list_add(tl1, tl2):
    if tl1 is None: # first addition
        return tl2
    else:
        assert len(tl1) == len(tl2)
        return [t1 + t2 for t1, t2 in zip(tl1, tl2)]


def tensor_list_div(tl, val):
    return [t / val for t in tl]


# class Bottleneck(nn.Module):
#     expansion = 4  # output channel multiplier

#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

# class ResNet50(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(ResNet50, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = self._make_layer(Bottleneck, 64, 3)
#         self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
#         self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
#         self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

#         # weight initialisation
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channels != out_channels * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, out_channels * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x
#     def embed(self,x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)

#         return x
def _center_crop_conv_weight(w_7x7, out_k=3):
    # [64, 3, 7, 7] -> [64, 3, 3, 3]
    s = (w_7x7.shape[-1] - out_k) // 2
    return w_7x7[:, :, s:s+out_k, s:s+out_k]

def load_imagenet_pretrain_skip_classifier(model: nn.Module, verbose=True):
    """
    Map torchvision ResNet50 (ImageNet1k) pretrained weights onto the target model:
      - stem 7x7 -> 3x3: centre-crop automatically
      - BN -> GN/IN: load weight/bias only; skip running_* / num_batches_tracked
      - classifier head: skip (target classifier is 200-dimensional)
    """
    off = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    off_sd = off.state_dict()
    my_sd  = model.state_dict()
    new_sd = {}

    # stem
    if 'conv1.weight' in my_sd and 'conv1.weight' in off_sd:
        w_off, w_my = off_sd['conv1.weight'], my_sd['conv1.weight']
        if w_my.shape[1] == 3 and w_my.shape[-1] == 3 and w_off.shape[-1] == 7:
            new_sd['conv1.weight'] = _center_crop_conv_weight(w_off, 3)
        elif w_off.shape == w_my.shape:
            new_sd['conv1.weight'] = w_off
        else:
            if verbose: print('[skip] conv1 mismatch:', w_off.shape, '->', w_my.shape)

    # backbone layers (layer1..4)
    for k, v in off_sd.items():
        if not any(k.startswith(s) for s in ['layer1', 'layer2', 'layer3', 'layer4']):
            continue
        if k not in my_sd:
            continue
        # skip BN running stats (GN/IN does not have these keys)
        if any(t in k for t in ['running_mean', 'running_var', 'num_batches_tracked']):
            continue
        if v.shape == my_sd[k].shape:
            new_sd[k] = v

    # skip classifier head (fc -> classifier) since num_classes=200
    # no mapping; keep the initialised classifier as-is

    # missing, unexpected = model.load_state_dict(new_sd, strict=False)
    # if verbose:
    #     print(f'Loaded {len(new_sd)} tensors. Missing={len(missing)}, Unexpected={len(unexpected)}')
    #     if missing:   print('Missing keys (kept init):', missing[:8], '...' if len(missing)>8 else '')
    #     if unexpected:print('Unexpected keys:', unexpected)
    return model

def fill_shortcut_and_bn1(model):
    off = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).state_dict()
    my_sd = model.state_dict()
    patched = {}

    # bn1 affine params: fill if shapes match (GN/IN only has weight/bias)
    for k_off in ['bn1.weight', 'bn1.bias']:
        if k_off in off and k_off in my_sd and off[k_off].shape == my_sd[k_off].shape:
            patched[k_off] = off[k_off]

    # downsample -> shortcut mapping
    for k_off, v_off in off.items():
        if '.downsample.' not in k_off:
            continue
        k_my = k_off.replace('.downsample.', '.shortcut.')
        # skip BN running_* and num_batches_tracked
        if any(t in k_off for t in ['running_mean', 'running_var', 'num_batches_tracked']):
            continue
        if k_my in my_sd and my_sd[k_my].shape == v_off.shape:
            patched[k_my] = v_off

    model.load_state_dict(patched, strict=False)
    print(f'Patched {len(patched)} tensors:', list(patched.keys())[:10], '...')
    return model
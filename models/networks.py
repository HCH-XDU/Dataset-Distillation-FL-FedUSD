import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTForImageClassification
# from torchvision.models import ResNet50
import timm
import numpy as np
from types import MethodType
# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,


''' MLP '''
class MLP(nn.Module):
    def __init__(self, channel, num_classes):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)
        self.modified = False

    def forward(self, x):
        if self.training: self.modified = True
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out

''' TINY MLP FOR GRADIENT MATCHING '''
class MLP_TINY(nn.Module):
    def __init__(self, channel, num_classes):
        super(MLP_TINY, self).__init__()
        self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
        # self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)        
        self.modified = False

    def forward(self, x):
        if self.training: self.modified = True
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = self.fc_3(out)
        return out


''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()
        if net_act == 'sigmoid':
            self.net_act = nn.Sigmoid()
        elif net_act == 'relu':
            self.net_act = nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            self.net_act = nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

        if net_pooling == 'maxpooling':
            self.net_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            self.net_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            self.net_pooling = None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)
        self.modified = False

    def forward(self, x, train=False, mode='dummy', normalize='dummy'):
        if self.training:
            self.modified = True
        out = self.features(x)
        inter_out = out.view(out.size(0), -1)
        out = self.classifier(inter_out)
        if train:
            return inter_out, out
        else:
            return out

    def embed(self, x):
        out0 = self.features(x)
        out = out0.view(out0.size(0), -1)
        logit = self.classifier(out)
        return out0, out, logit

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            norm = nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            norm = nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            norm = nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            norm = nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            norm = None
        else:
            norm = None
            exit('unknown net_norm: %s'%net_norm)
        return norm

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self.net_act]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self.net_pooling]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat



''' LeNet '''
class LeNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5, padding=2 if channel==1 else 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)
        self.modified = False

    def forward(self, x):
        if self.training: self.modified = True
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x



''' AlexNet '''
class AlexNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel==1 else 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * 4 * 4, num_classes)
        self.modified = False

    def forward(self, x):
        if self.training: self.modified = True
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



''' VGG '''
cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    def __init__(self, vgg_name, channel, num_classes, norm='instancenorm'):
        super(VGG, self).__init__()
        self.channel = channel
        self.features = self._make_layers(cfg_vgg[vgg_name], norm)
        self.classifier = nn.Linear(512 if vgg_name != 'VGGS' else 128, num_classes)
        self.modified = False

    def forward(self, x):
        if self.training: self.modified = True
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _make_layers(self, cfg, norm):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if self.channel==1 and ic==0 else 1),
                           nn.GroupNorm(x, x, affine=True) if norm=='instancenorm' else nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(channel, num_classes):
    return VGG('VGG11', channel, num_classes)
def VGG11BN(channel, num_classes):
    return VGG('VGG11', channel, num_classes, norm='batchnorm')
def VGG13(channel, num_classes):
    return VGG('VGG13', channel, num_classes)
def VGG16(channel, num_classes):
    return VGG('VGG16', channel, num_classes)
def VGG19(channel, num_classes):
    return VGG('VGG19', channel, num_classes)


''' ResNet_AP '''
# The conv(stride=2) is replaced by conv(stride=1) + avgpool(kernel_size=2, stride=2)

class BasicBlock_AP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # modification
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2), # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ViT_Tiny(nn.Module):
    def __init__(self, class_num=10, img_size=28, pre_path=None):
        pre_path = 'weight/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'
        super(ViT_Tiny, self).__init__()
        # create ViT Tiny model
        self.model = timm.create_model(
            'vit_tiny_patch16_224',
            num_classes=class_num,
            img_size=img_size,
            pretrained=False,
            # pretrained_cfg_overlay=dict(file=pre_path)
        )
        # init.trunc_normal_(self.model.patch_embed.proj.weight, std=0.02)
        # if self.model.patch_embed.proj.bias is not None:
        #     init.zeros_(self.model.patch_embed.proj.bias)
        # print("Patch embedding re-initialized.")
        # self.load_npz_without_pos_embed(pre_path)
        # extract key components
        self.patch_embed = self.model.patch_embed      # patch embedding module
        self.cls_token = self.model.cls_token          # classification token
        self.pos_embed = self.model.pos_embed          # positional encoding
        self.pos_drop = self.model.pos_drop            # dropout layer
        self.blocks = self.model.blocks                # transformer encoder blocks
        self.norm = self.model.norm                    # LayerNorm

    def embed(self, x):
        """
        Input: image x [B, 3, H, W]
        Output: feature map [B, N+1, D], where N is the number of patches and +1 is the cls token
        """
        B = x.shape[0]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)                        # [B, 1, H, W] -> [B, 3, H, W]
        x = self.model.patch_embed(x)                       # [B, N, D]
        cls_tokens = self.model.cls_token.expand(B, -1, -1) # [B, 1, D]

        x = torch.cat((cls_tokens, x), dim=1)         # [B, 1+N, D]
        x = x + self.model.pos_embed[:, :(x.size(1))]       # add positional encoding
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]  # extract final cls token
        # out = self.model.head(cls_tokens)
        return cls_token_final
    def caculate_embed(self, x):
        """
        Input: image x [B, 3, H, W]
        Output: feature map [B, N+1, D], where N is the number of patches and +1 is the cls token
        """
        B = x.shape[0]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)                        # [B, 1, H, W] -> [B, 3, H, W]
        x = self.model.patch_embed(x)                       # [B, N, D]
        cls_tokens = self.model.cls_token.expand(B, -1, -1) # [B, 1, D]

        x = torch.cat((cls_tokens, x), dim=1)         # [B, 1+N, D]
        x = x + self.model.pos_embed[:, :(x.size(1))]       # add positional encoding
        x = self.pos_drop(x)
        # out = self.model.head(cls_tokens)
        return x
    def forward(self, x):
        """
        Standard forward inference function.
        """
        x = self.caculate_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]  # extract final cls token
        out = self.model.head(cls_token_final)
        return out
    def load_npz_without_pos_embed(self, npz_path):
        npz = np.load(npz_path)
        state_dict = {}
        for name, param in self.model.named_parameters():
            print(f"{name}: {param.mean().item()}")
        for k in npz.files:
            if 'pos_embed' in k:
                continue  # skip positional encoding
            param = torch.from_numpy(npz[k])
            state_dict[k] = param
        self.model.load_state_dict(state_dict, strict=False)
        for name, param in self.model.named_parameters():
            print(f"{name}: {param.mean().item()}")
        print("Loaded pretrained weights (excluding pos_embed)")

class ViT_B_16(nn.Module):
    def __init__(self, num_classes=1000, merge_layer=1, merge_ratio=0.2):
        super().__init__()
        hf_model = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
        hf_state = hf_model.state_dict()

        # create timm model
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10)
        # for i, blk in enumerate(self.model.blocks):
        #     blk.attn = SelfAttentionWithSize(blk.attn)
        #     blk.forward = MethodType(block_forward, blk)
        # self.model.forward_features = MethodType(forward_features_with_token_sizes, self.model)
        # print("HF keys:", list(hf_state.keys())[:5])
        # print("Timm keys:", list(self.model.state_dict().keys())[:5])
        new_state = {}

        # 3. patch embedding
        new_state['patch_embed.proj.weight'] = hf_state['vit.embeddings.patch_embeddings.projection.weight']
        new_state['patch_embed.proj.bias'] = hf_state['vit.embeddings.patch_embeddings.projection.bias']

        # 4. cls_token and pos_embed
        new_state['cls_token'] = hf_state['vit.embeddings.cls_token']
        new_state['pos_embed'] = hf_state['vit.embeddings.position_embeddings']

        # 5. Transformer blocks
        for i in range(12):
            hf_blk = f'vit.encoder.layer.{i}'
            tm_blk = f'blocks.{i}'

            # norm1
            new_state[f'{tm_blk}.norm1.weight'] = hf_state[f'{hf_blk}.layernorm_before.weight']
            new_state[f'{tm_blk}.norm1.bias'] = hf_state[f'{hf_blk}.layernorm_before.bias']

            # attention: merge qkv
            q = hf_state[f'{hf_blk}.attention.attention.query.weight']
            k = hf_state[f'{hf_blk}.attention.attention.key.weight']
            v = hf_state[f'{hf_blk}.attention.attention.value.weight']
            new_state[f'{tm_blk}.attn.qkv.weight'] = torch.cat([q, k, v], dim=0)

            q = hf_state[f'{hf_blk}.attention.attention.query.bias']
            k = hf_state[f'{hf_blk}.attention.attention.key.bias']
            v = hf_state[f'{hf_blk}.attention.attention.value.bias']
            new_state[f'{tm_blk}.attn.qkv.bias'] = torch.cat([q, k, v], dim=0)

            new_state[f'{tm_blk}.attn.proj.weight'] = hf_state[f'{hf_blk}.attention.output.dense.weight']
            new_state[f'{tm_blk}.attn.proj.bias'] = hf_state[f'{hf_blk}.attention.output.dense.bias']

            # norm2
            new_state[f'{tm_blk}.norm2.weight'] = hf_state[f'{hf_blk}.layernorm_after.weight']
            new_state[f'{tm_blk}.norm2.bias'] = hf_state[f'{hf_blk}.layernorm_after.bias']

            # mlp
            new_state[f'{tm_blk}.mlp.fc1.weight'] = hf_state[f'{hf_blk}.intermediate.dense.weight']
            new_state[f'{tm_blk}.mlp.fc1.bias'] = hf_state[f'{hf_blk}.intermediate.dense.bias']
            new_state[f'{tm_blk}.mlp.fc2.weight'] = hf_state[f'{hf_blk}.output.dense.weight']
            new_state[f'{tm_blk}.mlp.fc2.bias'] = hf_state[f'{hf_blk}.output.dense.bias']

        # 6. final norm
        new_state['norm.weight'] = hf_state['vit.layernorm.weight']
        new_state['norm.bias'] = hf_state['vit.layernorm.bias']

        # 7. classifier head
        new_state['head.weight'] = hf_state['classifier.weight']
        new_state['head.bias'] = hf_state['classifier.bias']

        # 8. load state_dict
        missing, unexpected = self.model.load_state_dict(new_state, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        self.patch_embed = self.model.patch_embed
        self.blocks = self.model.blocks
        self.norm = self.model.norm
        self.head = self.model.head

    def embed_caculate(self, x):
        """
        Input: image x [B, 3, H, W]
        Output: feature map [B, N+1, D], where N is the number of patches and +1 is the cls token
        """
        B = x.shape[0]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)          # [B, 1, H, W] -> [B, 3, H, W]
        x = self.patch_embed(x)                # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+N, D]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                       # [B, 1+N, D]
        return x                               # returns the feature map (including cls token)
    def embed(self, x):
        """
        Input: image x [B, 3, H, W]
        Output: feature map [B, N+1, D], where N is the number of patches and +1 is the cls token
        """
        B = x.shape[0]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)          # [B, 1, H, W] -> [B, 3, H, W]
        x = self.patch_embed(x)                # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+N, D]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        cls_token = x[:, 0]                      # [B, 1+N, D]
        return cls_token                               # returns the cls token

    def forward(self, x):
        """
        Standard classification forward pass.
        """
        x = self.embed_caculate(x)
        cls_token = x[:, 0]                    # use only cls token for classification
        x = self.model.head(cls_token)
        return x
class SelfAttentionWithSize(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        # directly reuse the original weights
        self.qkv = original_attn.qkv
        self.proj = original_attn.proj
        self.num_heads = original_attn.num_heads
        self.scale = (original_attn.head_dim) ** -0.5 if hasattr(original_attn, 'head_dim') else None

    def forward(self, x, token_sizes=None):
        B, N, C = x.shape
        H = self.num_heads
        head_dim = C // H

        qkv = self.qkv(x).reshape(B, N, 3, H, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, N, H, D]
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        scale = self.scale or (head_dim ** -0.5)
        attn_scores = (q @ k.transpose(-2, -1)) * scale  # [B, H, N, N]

        if token_sizes is not None:
            # token_sizes: [B, N]
            log_s = torch.log(token_sizes + 1e-6)
            bias = log_s.unsqueeze(1).expand(-1, N, -1)  # [B, 1, N]
            bias = bias.unsqueeze(1)  # → [B, 1, 1, N]
            attn_scores = attn_scores + bias  # broadcasted to [B, H, N, N]

        attn = attn_scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)
class Bottleneck_AP(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # modification
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

        


class ResNet_AP(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet_AP, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion * 3 * 3 if channel==1 else 512 * block.expansion * 4 * 4, num_classes)  # modification
        self.modified = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training: self.modified = True
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1) # modification
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def ResNet18BN_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes)


''' ResNet '''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512*block.expansion, num_classes)
        self.modified = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training: self.modified = True
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)  
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        return out


def ResNet18BN(channel, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18(channel, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes)

def ResNet34(channel, num_classes):
    return ResNet(BasicBlock, [3,4,6,3], channel=channel, num_classes=num_classes)

def ResNet50(channel, num_classes):
    return ResNet(Bottleneck, [3,4,6,3], channel=channel, num_classes=num_classes)

def ResNet101(channel, num_classes):
    return ResNet(Bottleneck, [3,4,23,3], channel=channel, num_classes=num_classes)

def ResNet152(channel, num_classes):
    return ResNet(Bottleneck, [3,8,36,3], channel=channel, num_classes=num_classes)









###########################################################################################################
################################################### NLP ###################################################
###########################################################################################################

# -*- coding: utf-8 -*-
class CharCNN(nn.Module):
    def __init__(self, n_classes=14, input_length=1014, input_dim=68,
                 n_conv_filters=256,
                 n_fc_neurons=1024):
        dropout = 0 # was 0.5
        super(CharCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))

        dimension = int((input_length - 96) / 27 * n_conv_filters)
        self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.Dropout(dropout))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.Dropout(dropout))
        self.fc3 = nn.Linear(n_fc_neurons, n_classes)

        if n_conv_filters == 256 and n_fc_neurons == 1024:
            self._create_weights(mean=0.0, std=0.05)
        elif n_conv_filters == 1024 and n_fc_neurons == 2048:
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input, normalize='none'):
        if normalize == 'none':
            input = input
        elif normalize == 'softmax':
            input = torch.softmax(input, dim=-1)
        elif normalize == 'div':
            input = input / input.sum(dim=-1).unsqueeze(dim=-1)
        
        ## TODO under dev ##
        # assert (input > 1).sum() + (input < 0).sum() == 0
        input = input.transpose(1, 2)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output
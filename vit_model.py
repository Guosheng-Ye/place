#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   vit_model.py    
@Time    :   2022/11/3 21:49
@Author  :   Ye Guosheng
@Desc    :   None
"""

from functools import partial
from collections import OrderedDict
from torchsummary import summary
import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    随机深度的Drop方法
    
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        """
        call drop_path
        :param x:
        :return:
        """
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding for ViT/-Base
    kernel_size:16*16,
    stride:16
    """
    
    def __init__(self, img_size=256, patch_size=64, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)  # (256,256)
        patch_size = (patch_size, patch_size)  # (64,64)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # (4,4) => (16*768)的前半部分
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # (4*4=16)
        
        self.proj = nn.Conv2d(
                in_channels=in_c,
                out_channels=embed_dim,
                kernel_size=patch_size,  # 64
                stride=patch_size)  # 64
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        """
        patch_embedding(projection and norm)
        
        input_size:[3,256,256]
        output_size:[-1,16,768]
        
        :param x:input_img
        :return:norm(proj(x)),size:[B,HW,C]
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C] Note:交换两个维度时，permute 与 transpose 一样
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """
    Multi Self-Attention model
    """
    
    def __init__(self,
                 dim,  # 输入token的dim:768
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 768//8=96 每一个head的q,k,v对应的dim
        self.scale = qk_scale or head_dim ** -0.5  # \sqrt(1/head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv通过全连接实现 也可 3 * linear(dim,dim,...)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)  # 对应 Wo 矩阵
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    
    def forward(self, x):
        """

        :param x:size:[batch_size,196+1,768][batch_size , num_patches + 1 ,total_embedding_dim]
        :return:
        """
        
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torch script happy (cannot use tensor as tuple)
        
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        
        # q:[batch_size,  num_heads, num_patches + 1 , embed_dim_per_head]
        # k^T:[batch_size,num_heads, embed_dim_per_head , num_patches+1]
        # q * K^T =>[batch_size,num_heads,num_patches + 1 , num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # @ 只对两个张量的最后一个维度进行计算,self.scale 即 norm处理
        attn = attn.softmax(dim=-1)  # dim=-1对行进行softmax,dim=-2对列进行softmax
        attn = self.attn_drop(attn)
        
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim] # concat for each dim of heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()  # GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        """
        
        :param x:
        :return:
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Transformer Encoder Block
    """
    
    def __init__(self,
                 dim,  # 每个token的dim
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
    
    def forward(self, x):
        """
        forward in the Encoder Block
        :param x:
        :return:
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """
    ViT model
    """
    
    def __init__(self,
                 img_size=256,
                 patch_size=64,
                 in_c=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 distilled=False,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1  # default for 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # [batch_size,1,768]
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None  # None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  # [batch , 16+1 , 768]
        self.pos_drop = nn.Dropout(p=drop_ratio)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # drop_path ratio 以递增的方式传入
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop_ratio=drop_ratio,
                  attn_drop_ratio=attn_drop_ratio,
                  drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))  # OrderedDict() 有序字典
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        
        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:  # None
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        
        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
    
    def forward_features(self, x):
        """

        :param x: input_features
        :return:
        """
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B,3,256, 256] -> [B,16,768]
        
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [1, 1, 768] -> [B, 1, 768]
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B,16,768] -> [B, 17, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        x = self.pos_drop(x + self.pos_embed)  # position embedding + patch embedding
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:  # Extract the Class Token
            return self.pre_logits(x[:, 0])  # [batch,class_token] [768]
        # return self.pre_logits(x[:, :])  # [batch,17,embedding_num] [17,768]
        else:
            return x[:, 0], x[:, 1]
    
    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.forward_features(x)
        # return self.pre_logits(x[;,0]) [B,2,256,256] -> [B,768]
        # return self.pre_logits(x[:,:]) [B,2,256,256] -> [B,17,768]
        # print("forward_features_shape:{}".format(x.shape))
        
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)  # Linear Layer
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_for_image_composite(num_classes: int = 512):
    """
    
    :param num_classes: default for
    """
    vitps = VisionTransformer(
            img_size=256,
            patch_size=64,
            in_c=3,
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            representation_size=None,
            distilled=False,
            drop_ratio=0.,
            drop_path_ratio=0.,
            attn_drop_ratio=0,
            embed_layer=PatchEmbed,
            norm_layer=None, act_layer=None
    )
    return vitps


def summary_4_vit_base_16_224():
    """
    summary for vit base model
    """
    vit_base = vit_base_patch16_224(num_classes=1000)
    summary(
            model=vit_base,
            input_size=(3, 224, 224),
            batch_size=1,
    )
    """
    
    vit_base_patch16_224()
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [1, 768, 14, 14]         590,592
              Identity-2              [1, 196, 768]               0
            PatchEmbed-3              [1, 196, 768]               0
               Dropout-4              [1, 197, 768]               0
             LayerNorm-5              [1, 197, 768]           1,536
                Linear-6             [1, 197, 2304]       1,771,776
               Dropout-7          [1, 12, 197, 197]               0
                Linear-8              [1, 197, 768]         590,592
               Dropout-9              [1, 197, 768]               0
            Attention-10              [1, 197, 768]               0
             Identity-11              [1, 197, 768]               0
            LayerNorm-12              [1, 197, 768]           1,536
               Linear-13             [1, 197, 3072]       2,362,368
                 GELU-14             [1, 197, 3072]               0
              Dropout-15             [1, 197, 3072]               0
               Linear-16              [1, 197, 768]       2,360,064
              Dropout-17              [1, 197, 768]               0
                  Mlp-18              [1, 197, 768]               0
             Identity-19              [1, 197, 768]               0
                Block-20              [1, 197, 768]               0
            LayerNorm-21              [1, 197, 768]           1,536
               Linear-22             [1, 197, 2304]       1,771,776
              Dropout-23          [1, 12, 197, 197]               0
               Linear-24              [1, 197, 768]         590,592
              Dropout-25              [1, 197, 768]               0
            Attention-26              [1, 197, 768]               0
             Identity-27              [1, 197, 768]               0
            LayerNorm-28              [1, 197, 768]           1,536
               Linear-29             [1, 197, 3072]       2,362,368
                 GELU-30             [1, 197, 3072]               0
              Dropout-31             [1, 197, 3072]               0
               Linear-32              [1, 197, 768]       2,360,064
              Dropout-33              [1, 197, 768]               0
                  Mlp-34              [1, 197, 768]               0
             Identity-35              [1, 197, 768]               0
                Block-36              [1, 197, 768]               0
            LayerNorm-37              [1, 197, 768]           1,536
               Linear-38             [1, 197, 2304]       1,771,776
              Dropout-39          [1, 12, 197, 197]               0
               Linear-40              [1, 197, 768]         590,592
              Dropout-41              [1, 197, 768]               0
            Attention-42              [1, 197, 768]               0
             Identity-43              [1, 197, 768]               0
            LayerNorm-44              [1, 197, 768]           1,536
               Linear-45             [1, 197, 3072]       2,362,368
                 GELU-46             [1, 197, 3072]               0
              Dropout-47             [1, 197, 3072]               0
               Linear-48              [1, 197, 768]       2,360,064
              Dropout-49              [1, 197, 768]               0
                  Mlp-50              [1, 197, 768]               0
             Identity-51              [1, 197, 768]               0
                Block-52              [1, 197, 768]               0
            LayerNorm-53              [1, 197, 768]           1,536
               Linear-54             [1, 197, 2304]       1,771,776
              Dropout-55          [1, 12, 197, 197]               0
               Linear-56              [1, 197, 768]         590,592
              Dropout-57              [1, 197, 768]               0
            Attention-58              [1, 197, 768]               0
             Identity-59              [1, 197, 768]               0
            LayerNorm-60              [1, 197, 768]           1,536
               Linear-61             [1, 197, 3072]       2,362,368
                 GELU-62             [1, 197, 3072]               0
              Dropout-63             [1, 197, 3072]               0
               Linear-64              [1, 197, 768]       2,360,064
              Dropout-65              [1, 197, 768]               0
                  Mlp-66              [1, 197, 768]               0
             Identity-67              [1, 197, 768]               0
                Block-68              [1, 197, 768]               0
            LayerNorm-69              [1, 197, 768]           1,536
               Linear-70             [1, 197, 2304]       1,771,776
              Dropout-71          [1, 12, 197, 197]               0
               Linear-72              [1, 197, 768]         590,592
              Dropout-73              [1, 197, 768]               0
            Attention-74              [1, 197, 768]               0
             Identity-75              [1, 197, 768]               0
            LayerNorm-76              [1, 197, 768]           1,536
               Linear-77             [1, 197, 3072]       2,362,368
                 GELU-78             [1, 197, 3072]               0
              Dropout-79             [1, 197, 3072]               0
               Linear-80              [1, 197, 768]       2,360,064
              Dropout-81              [1, 197, 768]               0
                  Mlp-82              [1, 197, 768]               0
             Identity-83              [1, 197, 768]               0
                Block-84              [1, 197, 768]               0
            LayerNorm-85              [1, 197, 768]           1,536
               Linear-86             [1, 197, 2304]       1,771,776
              Dropout-87          [1, 12, 197, 197]               0
               Linear-88              [1, 197, 768]         590,592
              Dropout-89              [1, 197, 768]               0
            Attention-90              [1, 197, 768]               0
             Identity-91              [1, 197, 768]               0
            LayerNorm-92              [1, 197, 768]           1,536
               Linear-93             [1, 197, 3072]       2,362,368
                 GELU-94             [1, 197, 3072]               0
              Dropout-95             [1, 197, 3072]               0
               Linear-96              [1, 197, 768]       2,360,064
              Dropout-97              [1, 197, 768]               0
                  Mlp-98              [1, 197, 768]               0
             Identity-99              [1, 197, 768]               0
               Block-100              [1, 197, 768]               0
           LayerNorm-101              [1, 197, 768]           1,536
              Linear-102             [1, 197, 2304]       1,771,776
             Dropout-103          [1, 12, 197, 197]               0
              Linear-104              [1, 197, 768]         590,592
             Dropout-105              [1, 197, 768]               0
           Attention-106              [1, 197, 768]               0
            Identity-107              [1, 197, 768]               0
           LayerNorm-108              [1, 197, 768]           1,536
              Linear-109             [1, 197, 3072]       2,362,368
                GELU-110             [1, 197, 3072]               0
             Dropout-111             [1, 197, 3072]               0
              Linear-112              [1, 197, 768]       2,360,064
             Dropout-113              [1, 197, 768]               0
                 Mlp-114              [1, 197, 768]               0
            Identity-115              [1, 197, 768]               0
               Block-116              [1, 197, 768]               0
           LayerNorm-117              [1, 197, 768]           1,536
              Linear-118             [1, 197, 2304]       1,771,776
             Dropout-119          [1, 12, 197, 197]               0
              Linear-120              [1, 197, 768]         590,592
             Dropout-121              [1, 197, 768]               0
           Attention-122              [1, 197, 768]               0
            Identity-123              [1, 197, 768]               0
           LayerNorm-124              [1, 197, 768]           1,536
              Linear-125             [1, 197, 3072]       2,362,368
                GELU-126             [1, 197, 3072]               0
             Dropout-127             [1, 197, 3072]               0
              Linear-128              [1, 197, 768]       2,360,064
             Dropout-129              [1, 197, 768]               0
                 Mlp-130              [1, 197, 768]               0
            Identity-131              [1, 197, 768]               0
               Block-132              [1, 197, 768]               0
           LayerNorm-133              [1, 197, 768]           1,536
              Linear-134             [1, 197, 2304]       1,771,776
             Dropout-135          [1, 12, 197, 197]               0
              Linear-136              [1, 197, 768]         590,592
             Dropout-137              [1, 197, 768]               0
           Attention-138              [1, 197, 768]               0
            Identity-139              [1, 197, 768]               0
           LayerNorm-140              [1, 197, 768]           1,536
              Linear-141             [1, 197, 3072]       2,362,368
                GELU-142             [1, 197, 3072]               0
             Dropout-143             [1, 197, 3072]               0
              Linear-144              [1, 197, 768]       2,360,064
             Dropout-145              [1, 197, 768]               0
                 Mlp-146              [1, 197, 768]               0
            Identity-147              [1, 197, 768]               0
               Block-148              [1, 197, 768]               0
           LayerNorm-149              [1, 197, 768]           1,536
              Linear-150             [1, 197, 2304]       1,771,776
             Dropout-151          [1, 12, 197, 197]               0
              Linear-152              [1, 197, 768]         590,592
             Dropout-153              [1, 197, 768]               0
           Attention-154              [1, 197, 768]               0
            Identity-155              [1, 197, 768]               0
           LayerNorm-156              [1, 197, 768]           1,536
              Linear-157             [1, 197, 3072]       2,362,368
                GELU-158             [1, 197, 3072]               0
             Dropout-159             [1, 197, 3072]               0
              Linear-160              [1, 197, 768]       2,360,064
             Dropout-161              [1, 197, 768]               0
                 Mlp-162              [1, 197, 768]               0
            Identity-163              [1, 197, 768]               0
               Block-164              [1, 197, 768]               0
           LayerNorm-165              [1, 197, 768]           1,536
              Linear-166             [1, 197, 2304]       1,771,776
             Dropout-167          [1, 12, 197, 197]               0
              Linear-168              [1, 197, 768]         590,592
             Dropout-169              [1, 197, 768]               0
           Attention-170              [1, 197, 768]               0
            Identity-171              [1, 197, 768]               0
           LayerNorm-172              [1, 197, 768]           1,536
              Linear-173             [1, 197, 3072]       2,362,368
                GELU-174             [1, 197, 3072]               0
             Dropout-175             [1, 197, 3072]               0
              Linear-176              [1, 197, 768]       2,360,064
             Dropout-177              [1, 197, 768]               0
                 Mlp-178              [1, 197, 768]               0
            Identity-179              [1, 197, 768]               0
               Block-180              [1, 197, 768]               0
           LayerNorm-181              [1, 197, 768]           1,536
              Linear-182             [1, 197, 2304]       1,771,776
             Dropout-183          [1, 12, 197, 197]               0
              Linear-184              [1, 197, 768]         590,592
             Dropout-185              [1, 197, 768]               0
           Attention-186              [1, 197, 768]               0
            Identity-187              [1, 197, 768]               0
           LayerNorm-188              [1, 197, 768]           1,536
              Linear-189             [1, 197, 3072]       2,362,368
                GELU-190             [1, 197, 3072]               0
             Dropout-191             [1, 197, 3072]               0
              Linear-192              [1, 197, 768]       2,360,064
             Dropout-193              [1, 197, 768]               0
                 Mlp-194              [1, 197, 768]               0
            Identity-195              [1, 197, 768]               0
               Block-196              [1, 197, 768]               0
           LayerNorm-197              [1, 197, 768]           1,536
            Identity-198                   [1, 768]               0
              Linear-199                  [1, 1000]         769,000
    ================================================================
    Total params: 86,415,592
    Trainable params: 86,415,592
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 408.54
    Params size (MB): 329.65
    Estimated Total Size (MB): 738.77
    ----------------------------------------------------------------
    """


def summary_4_vit_for_image_composite():
    """
    summary for the vitps
    """
    vitps = vit_for_image_composite().cuda()
    summary(
            model=vitps,
            input_size=(3, 256, 256),
            batch_size=1
    )
    """
    vitps for image composite_patch64_256()
    ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1             [1, 768, 4, 4]       9,437,952
              Identity-2               [1, 16, 768]               0
            PatchEmbed-3               [1, 16, 768]               0
               Dropout-4               [1, 17, 768]               0
             LayerNorm-5               [1, 17, 768]           1,536
                Linear-6              [1, 17, 2304]       1,771,776
               Dropout-7            [1, 12, 17, 17]               0
                Linear-8               [1, 17, 768]         590,592
               Dropout-9               [1, 17, 768]               0
            Attention-10               [1, 17, 768]               0
             Identity-11               [1, 17, 768]               0
            LayerNorm-12               [1, 17, 768]           1,536
               Linear-13              [1, 17, 3072]       2,362,368
                 GELU-14              [1, 17, 3072]               0
              Dropout-15              [1, 17, 3072]               0
               Linear-16               [1, 17, 768]       2,360,064
              Dropout-17               [1, 17, 768]               0
                  Mlp-18               [1, 17, 768]               0
             Identity-19               [1, 17, 768]               0
                Block-20               [1, 17, 768]               0
            LayerNorm-21               [1, 17, 768]           1,536
               Linear-22              [1, 17, 2304]       1,771,776
              Dropout-23            [1, 12, 17, 17]               0
               Linear-24               [1, 17, 768]         590,592
              Dropout-25               [1, 17, 768]               0
            Attention-26               [1, 17, 768]               0
             Identity-27               [1, 17, 768]               0
            LayerNorm-28               [1, 17, 768]           1,536
               Linear-29              [1, 17, 3072]       2,362,368
                 GELU-30              [1, 17, 3072]               0
              Dropout-31              [1, 17, 3072]               0
               Linear-32               [1, 17, 768]       2,360,064
              Dropout-33               [1, 17, 768]               0
                  Mlp-34               [1, 17, 768]               0
             Identity-35               [1, 17, 768]               0
                Block-36               [1, 17, 768]               0
            LayerNorm-37               [1, 17, 768]           1,536
               Linear-38              [1, 17, 2304]       1,771,776
              Dropout-39            [1, 12, 17, 17]               0
               Linear-40               [1, 17, 768]         590,592
              Dropout-41               [1, 17, 768]               0
            Attention-42               [1, 17, 768]               0
             Identity-43               [1, 17, 768]               0
            LayerNorm-44               [1, 17, 768]           1,536
               Linear-45              [1, 17, 3072]       2,362,368
                 GELU-46              [1, 17, 3072]               0
              Dropout-47              [1, 17, 3072]               0
               Linear-48               [1, 17, 768]       2,360,064
              Dropout-49               [1, 17, 768]               0
                  Mlp-50               [1, 17, 768]               0
             Identity-51               [1, 17, 768]               0
                Block-52               [1, 17, 768]               0
            LayerNorm-53               [1, 17, 768]           1,536
               Linear-54              [1, 17, 2304]       1,771,776
              Dropout-55            [1, 12, 17, 17]               0
               Linear-56               [1, 17, 768]         590,592
              Dropout-57               [1, 17, 768]               0
            Attention-58               [1, 17, 768]               0
             Identity-59               [1, 17, 768]               0
            LayerNorm-60               [1, 17, 768]           1,536
               Linear-61              [1, 17, 3072]       2,362,368
                 GELU-62              [1, 17, 3072]               0
              Dropout-63              [1, 17, 3072]               0
               Linear-64               [1, 17, 768]       2,360,064
              Dropout-65               [1, 17, 768]               0
                  Mlp-66               [1, 17, 768]               0
             Identity-67               [1, 17, 768]               0
                Block-68               [1, 17, 768]               0
            LayerNorm-69               [1, 17, 768]           1,536
               Linear-70              [1, 17, 2304]       1,771,776
              Dropout-71            [1, 12, 17, 17]               0
               Linear-72               [1, 17, 768]         590,592
              Dropout-73               [1, 17, 768]               0
            Attention-74               [1, 17, 768]               0
             Identity-75               [1, 17, 768]               0
            LayerNorm-76               [1, 17, 768]           1,536
               Linear-77              [1, 17, 3072]       2,362,368
                 GELU-78              [1, 17, 3072]               0
              Dropout-79              [1, 17, 3072]               0
               Linear-80               [1, 17, 768]       2,360,064
              Dropout-81               [1, 17, 768]               0
                  Mlp-82               [1, 17, 768]               0
             Identity-83               [1, 17, 768]               0
                Block-84               [1, 17, 768]               0
            LayerNorm-85               [1, 17, 768]           1,536
               Linear-86              [1, 17, 2304]       1,771,776
              Dropout-87            [1, 12, 17, 17]               0
               Linear-88               [1, 17, 768]         590,592
              Dropout-89               [1, 17, 768]               0
            Attention-90               [1, 17, 768]               0
             Identity-91               [1, 17, 768]               0
            LayerNorm-92               [1, 17, 768]           1,536
               Linear-93              [1, 17, 3072]       2,362,368
                 GELU-94              [1, 17, 3072]               0
              Dropout-95              [1, 17, 3072]               0
               Linear-96               [1, 17, 768]       2,360,064
              Dropout-97               [1, 17, 768]               0
                  Mlp-98               [1, 17, 768]               0
             Identity-99               [1, 17, 768]               0
               Block-100               [1, 17, 768]               0
           LayerNorm-101               [1, 17, 768]           1,536
              Linear-102              [1, 17, 2304]       1,771,776
             Dropout-103            [1, 12, 17, 17]               0
              Linear-104               [1, 17, 768]         590,592
             Dropout-105               [1, 17, 768]               0
           Attention-106               [1, 17, 768]               0
            Identity-107               [1, 17, 768]               0
           LayerNorm-108               [1, 17, 768]           1,536
              Linear-109              [1, 17, 3072]       2,362,368
                GELU-110              [1, 17, 3072]               0
             Dropout-111              [1, 17, 3072]               0
              Linear-112               [1, 17, 768]       2,360,064
             Dropout-113               [1, 17, 768]               0
                 Mlp-114               [1, 17, 768]               0
            Identity-115               [1, 17, 768]               0
               Block-116               [1, 17, 768]               0
           LayerNorm-117               [1, 17, 768]           1,536
              Linear-118              [1, 17, 2304]       1,771,776
             Dropout-119            [1, 12, 17, 17]               0
              Linear-120               [1, 17, 768]         590,592
             Dropout-121               [1, 17, 768]               0
           Attention-122               [1, 17, 768]               0
            Identity-123               [1, 17, 768]               0
           LayerNorm-124               [1, 17, 768]           1,536
              Linear-125              [1, 17, 3072]       2,362,368
                GELU-126              [1, 17, 3072]               0
             Dropout-127              [1, 17, 3072]               0
              Linear-128               [1, 17, 768]       2,360,064
             Dropout-129               [1, 17, 768]               0
                 Mlp-130               [1, 17, 768]               0
            Identity-131               [1, 17, 768]               0
               Block-132               [1, 17, 768]               0
           LayerNorm-133               [1, 17, 768]           1,536
              Linear-134              [1, 17, 2304]       1,771,776
             Dropout-135            [1, 12, 17, 17]               0
              Linear-136               [1, 17, 768]         590,592
             Dropout-137               [1, 17, 768]               0
           Attention-138               [1, 17, 768]               0
            Identity-139               [1, 17, 768]               0
           LayerNorm-140               [1, 17, 768]           1,536
              Linear-141              [1, 17, 3072]       2,362,368
                GELU-142              [1, 17, 3072]               0
             Dropout-143              [1, 17, 3072]               0
              Linear-144               [1, 17, 768]       2,360,064
             Dropout-145               [1, 17, 768]               0
                 Mlp-146               [1, 17, 768]               0
            Identity-147               [1, 17, 768]               0
               Block-148               [1, 17, 768]               0
           LayerNorm-149               [1, 17, 768]           1,536
              Linear-150              [1, 17, 2304]       1,771,776
             Dropout-151            [1, 12, 17, 17]               0
              Linear-152               [1, 17, 768]         590,592
             Dropout-153               [1, 17, 768]               0
           Attention-154               [1, 17, 768]               0
            Identity-155               [1, 17, 768]               0
           LayerNorm-156               [1, 17, 768]           1,536
              Linear-157              [1, 17, 3072]       2,362,368
                GELU-158              [1, 17, 3072]               0
             Dropout-159              [1, 17, 3072]               0
              Linear-160               [1, 17, 768]       2,360,064
             Dropout-161               [1, 17, 768]               0
                 Mlp-162               [1, 17, 768]               0
            Identity-163               [1, 17, 768]               0
               Block-164               [1, 17, 768]               0
           LayerNorm-165               [1, 17, 768]           1,536
              Linear-166              [1, 17, 2304]       1,771,776
             Dropout-167            [1, 12, 17, 17]               0
              Linear-168               [1, 17, 768]         590,592
             Dropout-169               [1, 17, 768]               0
           Attention-170               [1, 17, 768]               0
            Identity-171               [1, 17, 768]               0
           LayerNorm-172               [1, 17, 768]           1,536
              Linear-173              [1, 17, 3072]       2,362,368
                GELU-174              [1, 17, 3072]               0
             Dropout-175              [1, 17, 3072]               0
              Linear-176               [1, 17, 768]       2,360,064
             Dropout-177               [1, 17, 768]               0
                 Mlp-178               [1, 17, 768]               0
            Identity-179               [1, 17, 768]               0
               Block-180               [1, 17, 768]               0
           LayerNorm-181               [1, 17, 768]           1,536
              Linear-182              [1, 17, 2304]       1,771,776
             Dropout-183            [1, 12, 17, 17]               0
              Linear-184               [1, 17, 768]         590,592
             Dropout-185               [1, 17, 768]               0
           Attention-186               [1, 17, 768]               0
            Identity-187               [1, 17, 768]               0
           LayerNorm-188               [1, 17, 768]           1,536
              Linear-189              [1, 17, 3072]       2,362,368
                GELU-190              [1, 17, 3072]               0
             Dropout-191              [1, 17, 3072]               0
              Linear-192               [1, 17, 768]       2,360,064
             Dropout-193               [1, 17, 768]               0
                 Mlp-194               [1, 17, 768]               0
            Identity-195               [1, 17, 768]               0
               Block-196               [1, 17, 768]               0
           LayerNorm-197               [1, 17, 768]           1,536
            Identity-198                   [1, 768]               0
              Linear-199                  [1, 1000]         769,000
    ================================================================
    Total params: 95,262,952
    Trainable params: 95,262,952
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.75
    Forward/backward pass size (MB): 31.89
    Params size (MB): 363.40
    Estimated Total Size (MB): 396.04
    ----------------------------------------------------------------
    Process finished with exit code 0

    """


def summary_4_patch_embed():
    """
    summary for patch embedding
    """
    patch_emb = PatchEmbed().cuda()
    summary(
            model=patch_emb,
            input_size=(3, 256, 256),
            batch_size=-1,
    )
    """
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 768, 4, 4]       9,437,952
              Identity-2              [-1, 16, 768]               0
    ================================================================
    Total params: 9,437,952
    Trainable params: 9,437,952
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.75
    Forward/backward pass size (MB): 0.19
    Params size (MB): 36.00
    Estimated Total Size (MB): 36.94
    ----------------------------------------------------------------

    """


def summary_4_block():
    """
    summary for Block
    """
    block = Block(dim=768, num_heads=12).cuda()
    summary(
            model=block,
            input_size=(16, 768),
            batch_size=-1
    )
    """
    ----------------------------------------------------------------
       Layer (type)               Output Shape         Param #
    ================================================================
             LayerNorm-1              [-1, 16, 768]           1,536
                Linear-2             [-1, 16, 2304]       1,769,472
               Dropout-3           [-1, 12, 16, 16]               0
                Linear-4              [-1, 16, 768]         590,592
               Dropout-5              [-1, 16, 768]               0
             Attention-6              [-1, 16, 768]               0
              Identity-7              [-1, 16, 768]               0
             LayerNorm-8              [-1, 16, 768]           1,536
                Linear-9             [-1, 16, 3072]       2,362,368
                 GELU-10             [-1, 16, 3072]               0
              Dropout-11             [-1, 16, 3072]               0
               Linear-12              [-1, 16, 768]       2,360,064
              Dropout-13              [-1, 16, 768]               0
                  Mlp-14              [-1, 16, 768]               0
             Identity-15              [-1, 16, 768]               0
    ================================================================
    Total params: 7,085,568
    Trainable params: 7,085,568
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.05
    Forward/backward pass size (MB): 2.37
    Params size (MB): 27.03
    Estimated Total Size (MB): 29.44
    ----------------------------------------------------------------
    """


def summary_4_mlp():
    """
    summary for mlp
    """
    mlp = Mlp(in_features=768, hidden_features=768 * 3, out_features=5).cuda()
    summary(
            model=mlp,
            input_size=(1, 768),
            batch_size=-1
    )
    """
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1              [-1, 1, 2304]       1,771,776
                  GELU-2              [-1, 1, 2304]               0
               Dropout-3              [-1, 1, 2304]               0
                Linear-4                 [-1, 1, 5]          11,525
               Dropout-5                 [-1, 1, 5]               0
    ================================================================
    """


if __name__ == '__main__':
    # summary_4_patch_embed()
    # summary_4_block()
    # summary_4_mlp()
    # summary_4_vit_for_image_composite()
    
    """
    3670 images were found in the dataset.
    2939 images for training.
    731 images for validation.
    Using 8 dataloader workers every process
    training head.weight
    training head.bias
    [train epoch 0] loss: 1.551, acc: 0.336: 100%|██████████| 368/368 [00:21<00:00, 17.40it/s]
    [valid epoch 0] loss: 1.425, acc: 0.390: 100%|██████████| 92/92 [00:15<00:00,  5.97it/s]
    
    [train epoch 1] loss: 1.464, acc: 0.391: 100%|██████████| 368/368 [00:18<00:00, 19.99it/s]
    [valid epoch 1] loss: 1.344, acc: 0.435: 100%|██████████| 92/92 [00:15<00:00,  5.97it/s]
    
    [train epoch 2] loss: 1.448, acc: 0.402: 100%|██████████| 368/368 [00:18<00:00, 19.94it/s]
    [valid epoch 2] loss: 1.413, acc: 0.416: 100%|██████████| 92/92 [00:15<00:00,  6.05it/s]
    
    [train epoch 3] loss: 1.379, acc: 0.430: 100%|██████████| 368/368 [00:18<00:00, 20.14it/s]
    [valid epoch 3] loss: 1.333, acc: 0.447: 100%|██████████| 92/92 [00:15<00:00,  6.00it/s]
    
    [train epoch 4] loss: 1.363, acc: 0.430: 100%|██████████| 368/368 [00:18<00:00, 20.12it/s]
    [valid epoch 4] loss: 1.400, acc: 0.423: 100%|██████████| 92/92 [00:15<00:00,  5.99it/s]
    
    [train epoch 5] loss: 1.307, acc: 0.462: 100%|██████████| 368/368 [00:18<00:00, 20.34it/s]
    [valid epoch 5] loss: 1.376, acc: 0.440: 100%|██████████| 92/92 [00:15<00:00,  6.08it/s]
    
    [train epoch 6] loss: 1.322, acc: 0.460: 100%|██████████| 368/368 [00:18<00:00, 19.91it/s]
    [valid epoch 6] loss: 1.371, acc: 0.430: 100%|██████████| 92/92 [00:15<00:00,  6.00it/s]
    
    [train epoch 7] loss: 1.295, acc: 0.457: 100%|██████████| 368/368 [00:18<00:00, 19.89it/s]
    [valid epoch 7] loss: 1.310, acc: 0.440: 100%|██████████| 92/92 [00:15<00:00,  6.01it/s]
    
    [train epoch 8] loss: 1.262, acc: 0.485: 100%|██████████| 368/368 [00:18<00:00, 20.15it/s]
    [valid epoch 8] loss: 1.385, acc: 0.417: 100%|██████████| 92/92 [00:15<00:00,  5.99it/s]
    
    [train epoch 9] loss: 1.264, acc: 0.475: 100%|██████████| 368/368 [00:18<00:00, 19.80it/s]
    [valid epoch 9] loss: 1.310, acc: 0.447: 100%|██████████| 92/92 [00:15<00:00,  5.76it/s]
    
    [train epoch 10] loss: 1.250, acc: 0.475: 100%|██████████| 368/368 [00:19<00:00, 18.72it/s]
    [valid epoch 10] loss: 1.276, acc: 0.471: 100%|██████████| 92/92 [00:15<00:00,  5.85it/s]
    
    [train epoch 11] loss: 1.241, acc: 0.487: 100%|██████████| 368/368 [00:19<00:00, 19.15it/s]
    [valid epoch 11] loss: 1.347, acc: 0.413: 100%|██████████| 92/92 [00:15<00:00,  6.02it/s]
    
    [train epoch 12] loss: 1.212, acc: 0.503: 100%|██████████| 368/368 [00:18<00:00, 20.21it/s]
    [valid epoch 12] loss: 1.262, acc: 0.488: 100%|██████████| 92/92 [00:15<00:00,  5.97it/s]
    
    [train epoch 13] loss: 1.194, acc: 0.514: 100%|██████████| 368/368 [00:18<00:00, 19.51it/s]
    [valid epoch 13] loss: 1.298, acc: 0.446: 100%|██████████| 92/92 [00:15<00:00,  5.83it/s]
    
    [train epoch 14] loss: 1.176, acc: 0.521: 100%|██████████| 368/368 [00:19<00:00, 19.33it/s]
    [valid epoch 14] loss: 1.258, acc: 0.477: 100%|██████████| 92/92 [00:16<00:00,  5.67it/s]
    
    [train epoch 15] loss: 1.176, acc: 0.524: 100%|██████████| 368/368 [00:19<00:00, 18.85it/s]
    [valid epoch 15] loss: 1.236, acc: 0.499: 100%|██████████| 92/92 [00:15<00:00,  5.92it/s]
    
    [train epoch 16] loss: 1.159, acc: 0.537: 100%|██████████| 368/368 [00:18<00:00, 19.85it/s]
    [valid epoch 16] loss: 1.236, acc: 0.479: 100%|██████████| 92/92 [00:15<00:00,  5.94it/s]
    
    [train epoch 17] loss: 1.155, acc: 0.537: 100%|██████████| 368/368 [00:18<00:00, 19.91it/s]
    [valid epoch 17] loss: 1.233, acc: 0.484: 100%|██████████| 92/92 [00:15<00:00,  5.99it/s]
    
    [train epoch 18] loss: 1.150, acc: 0.544: 100%|██████████| 368/368 [00:19<00:00, 19.14it/s]
    [valid epoch 18] loss: 1.233, acc: 0.492: 100%|██████████| 92/92 [00:15<00:00,  5.96it/s]
    
    [train epoch 19] loss: 1.143, acc: 0.541: 100%|██████████| 368/368 [00:18<00:00, 20.23it/s]
    [valid epoch 19] loss: 1.233, acc: 0.487: 100%|██████████| 92/92 [00:15<00:00,  6.00it/s]
    """

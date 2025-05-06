#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/9/23 5:58 PM
# @Author  : zhangchao
# @File    : __init__.py.py
# @Email   : zhangchao5@genomics.cn
from .domain_specific_batch_norm import DomainSpecificBN1d
from .embed_layer import EmbeddingLayer
from .graph_vae import GraphVAE
from .residual_bottleneck import ResidualLayer
from .embed_model import FeatEmbed, ResidualEmbed
from .dgi import DGI
from .losses import scale_mse, contrast_loss, cross_instance_loss, trivial_entropy, kl_loss


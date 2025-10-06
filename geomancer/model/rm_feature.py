import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from geoopt.manifolds.stereographic.math import project
from geoopt.manifolds.stereographic import StereographicExact
from geoopt import ManifoldTensor
from geoopt import ManifoldParameter
EPS = 1e-6
class RiemannianFeatures(nn.Module):
    def __init__(self, num_nodes, dimensions, init_curvature, num_factors, device ='cuda', learnable=False):
        super(RiemannianFeatures, self).__init__()
        self.manifolds = nn.ModuleList()
        self.features = nn.ParameterList()
        self.Ws = nn.ParameterList()
        self.bias = nn.ParameterList()
        self.device = device
        for i in range(num_factors):
            d = dimensions
            k = init_curvature[i] 
            manifold = StereographicExact(k=k, learnable=learnable)
            features = ManifoldParameter(ManifoldTensor(torch.empty(num_nodes, d), manifold=manifold))
            
            if k != 0:
                self.init_weights(features)
            self.manifolds.append(manifold)
            self.features.append(features)
        for i in range(num_factors):
            d = dimensions
            pre = torch.randn(d, d)
            w = pre / (torch.norm(pre, dim=-1, keepdim=True) + EPS)
            w = w
            self.Ws.append(w)
            self.bias.append(2 * torch.pi * torch.rand(d))
        self.products = nn.ParameterList()
        for manifold, features in zip(self.manifolds, self.features):
            self.products.append(project(features, k=manifold.k))
    @staticmethod
    def init_weights(w, scale=1e-2):
        w.data.uniform_(-scale, scale)
        w_norm = w.data.norm(p=2, dim=-1, keepdim=True) + EPS
        w.data = w.data / w_norm * w.manifold.radius * 0.9 * torch.rand(1)


    
    def random_mapping_multi(self):
        # print('success')
        out = []
        rm_list =self.products
        for i in range(len(self.manifolds)):
            manifold = self.manifolds[i]
            x = rm_list[i]
            assert not torch.isnan(x).any(), "Tensor x contains NaN values"
            w = self.Ws[i]
            b = self.bias[i]
            k = manifold.k
            # print(x.device)
            # print(w.device)
            if k == 0:
                distance = x @ w.t()
            else:
                div = torch.sum((x[:, None] - w[None]) ** 2, dim=-1)
                distance = torch.log((1 + k * torch.sum(x * x, -1, keepdim=True)) / (div + EPS) + EPS)
            n = x.shape[-1]
            z = torch.exp((n - 1) * distance / 2) * torch.cos(distance + b)
            assert not torch.isnan(z).any(), "Tensor z contains NaN values"
            out.append(z)
        return out
    def random_mapping_multi_node(self):
        print('success')
        out = []
        rm_list =self.products
        for i in range(len(self.manifolds)):
            manifold = self.manifolds[i]
            x = rm_list[i]
            print(x)
            assert not torch.isnan(x).any(), "Tensor x contains NaN values"
            w = self.Ws[i]
            b = self.bias[i]
            k = manifold.k
            # print(x.device)
            # print(w.device)
            if k == 0:
                distance = x @ w.t()
            else:
                div = torch.sum((x[:, None] - w[None]) ** 2, dim=-1)
                distance = torch.log((1 + k * torch.sum(x * x, -1, keepdim=True)) / (div + EPS) + EPS)
            n = x.shape[-1]
            z = torch.exp((n - 1) * distance / 2) * torch.cos(distance + b)
            assert not torch.isnan(z).any(), "Tensor z contains NaN values"
            out.append(z)
        return out
    

a=RiemannianFeatures(32,32,[0.5,-0.5],2)
a = a.to('cuda')
# print(a.products[0])
out = a.random_mapping_multi( )
print(out[0].device)
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
import torch.nn as nn
from torch_geometric.graphgym.register import register_network
from .utils import *
from .rm_feature import RiemannianFeatures
from geoopt.manifolds.stereographic.math import project
from geoopt.manifolds.stereographic import StereographicExact
from geoopt import ManifoldTensor
from geoopt import ManifoldParameter
from torch_geometric.graphgym.config import cfg
# epsilon = 1e-15
EPS = 1e-10
class MPNNs(torch.nn.Module):
    # def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3, 
    #              dropout=0.5, heads=1, pre_ln=False, pre_linear=False, res=False, ln=False, bn=False, jk=False, gnn='gcn'):
    def __init__(self, dim_in=cfg.share.dim_in, dim_out=cfg.share.dim_out, cfg=None):
        super(MPNNs, self).__init__()

        self.dropout = cfg.dropout
        self.pre_ln = cfg.pre_ln

        self.pre_linear = cfg.pre_linear
        self.res = cfg.res
        self.ln = cfg.ln
        self.bn = cfg.bn
        self.jk = cfg.jk
        
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        gnn = cfg.gnn
        in_channels = cfg.in_dim
        hidden_channels = cfg.hid_dim
        out_channels = cfg.out_dim
        local_layers = cfg.local_layers
        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)
        
        if not self.pre_linear:
            if gnn=='gat':
                self.local_convs.append(GATConv(in_channels, hidden_channels, heads=cfg.heads,
                    concat=True, add_self_loops=False, bias=False))
            elif gnn=='sage':
                self.local_convs.append(SAGEConv(in_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(in_channels, hidden_channels,
                        cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(in_channels))
            local_layers = local_layers - 1
            
        for _ in range(local_layers):
            if gnn=='gat':
                self.local_convs.append(GATConv(hidden_channels, hidden_channels, heads=cfg.heads,
                    concat=True, add_self_loops=False, bias=False))
            elif gnn=='sage':
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels,
                        cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(hidden_channels))
                
        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()

    def forward(self,batch,label=None):
        x = batch.x
        edge_index = batch.edge_index
        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_final = 0
        
        for i, local_conv in enumerate(self.local_convs):
            if self.res:
                x = local_conv(x, edge_index) + self.lins[i](x)
            else:
                x = local_conv(x, edge_index)
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            else:
                pass
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x
            batch.x = x_final
        return batch
    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()

    def encode(self,batch,label=None):
        return self.forward(batch)
    
    def decode(self,batch,task=None,split=None):
        mask = batch.get(split+'_mask', None)
        x_final = batch.x[mask]
        x = self.pred_local(x_final)
        
        return x

class MPNNs_curv(torch.nn.Module):
    # def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3, 
    #              dropout=0.5, heads=1, pre_ln=False, pre_linear=False, res=False, ln=False, bn=False, jk=False, gnn='gcn'):
    def __init__(self, dim_in=cfg.share.dim_in, dim_out=cfg.share.dim_out, cfg=None):
        super(MPNNs_curv, self).__init__()

        self.dropout = cfg.dropout
        self.pre_ln = cfg.pre_ln

        self.pre_linear = cfg.pre_linear
        self.res = cfg.res
        self.ln = cfg.ln
        self.bn = cfg.bn
        self.jk = cfg.jk
        
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        gnn = cfg.gnn
        in_channels = cfg.in_dim
        hidden_channels = cfg.hid_dim
        out_channels = cfg.out_dim
        local_layers = cfg.local_layers
        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)
        
        if not self.pre_linear:
            if gnn=='gat':
                self.local_convs.append(GATConv(in_channels, hidden_channels, heads=cfg.heads,
                    concat=True, add_self_loops=False, bias=False))
            elif gnn=='sage':
                self.local_convs.append(SAGEConv(in_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(in_channels, hidden_channels,
                        cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(in_channels))
            local_layers = local_layers - 1
            
        for _ in range(local_layers):
            if gnn=='gat':
                self.local_convs.append(GATConv(hidden_channels, hidden_channels, heads=cfg.heads,
                    concat=True, add_self_loops=False, bias=False))
            elif gnn=='sage':
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels,
                        cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(hidden_channels))

        learnable = cfg.learnable
        self.out_dim =hidden_channels
        if cfg.rm_encode:
            #init
            self.curv1 = cfg.encode_curv
            self.rm_encode_manifolds = nn.ModuleList()
            for i,k in enumerate(self.curv1):
                manifold = StereographicExact(k=k, learnable=learnable)
                self.rm_encode_manifolds.append(manifold)
            self.encode_num_factors = cfg.encode_factor
            num_factors = self.encode_num_factors
            d = hidden_channels
            #构建node_kernel
            self.rm_encode_feature = nn.ParameterList()
            for i,k in enumerate(self.curv1):
                features = ManifoldParameter(ManifoldTensor(torch.empty(d, d), manifold=self.rm_encode_manifolds[i]))
                self.init_weights(features)
                self.rm_encode_feature.append(project(features, k=manifold.k))

            self.Ws_encode = nn.ParameterList()
            self.bias_encode = nn.ParameterList()
            self.lambda_encode = nn.ParameterList()
            for i in range(len(self.curv1)):
                pre = torch.randn(d, d)
                w = pre / (torch.norm(pre, dim=-1, keepdim=True) + EPS)
                self.Ws_encode.append(w)
                self.bias_encode.append(2 * torch.pi * torch.rand(d,d))
                self.lambda_encode.append(torch.rand(d,d))

            
       
            self.encode_kernel_layer_node = nn.Linear((len(self.curv1)+1)*self.out_dim,self.out_dim)
            if cfg.encode_layer_norm:
                self.encode_kernel_norm_node = nn.LayerNorm(self.out_dim)
            if cfg.encode_batch_norm:
                self.encode_kernel_norm_node = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                  momentum=self.bn_momentum)
        
        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

    def init_weights(self, w, scale=1e-2):
        w.data.uniform_(-scale, scale)
        w_norm = w.data.norm(p=2, dim=-1, keepdim=True) + EPS
        w.data = w.data / w_norm * w.manifold.radius * 0.9 * torch.rand(1)

    def general_fourier_mapping(self):
        # print('success')
        out = []
        
        for i,rm_feature in enumerate(self.rm_encode_feature):

            k = self.rm_encode_manifolds[i].k
            
            x = rm_feature
            assert not torch.isnan(x).any(), "Tensor x contains NaN values"
            w = self.Ws_encode[i]
            b = self.bias_encode[i]
            lamda = self.lambda_encode[i]
            # print(x.device)
            # print(w.device)
            if k == 0:
                distance = x @ w.t()
            else:
                div = torch.sum((x[:, None] - w[None]) ** 2, dim=-1)
                distance = torch.log((1 + k * torch.sum(x * x, -1, keepdim=True)) / (div + EPS) + EPS)
            n = x.shape[-1]
            z = torch.exp((n - 1) * distance / 2) * torch.cos(lamda*distance + b)
            assert not torch.isnan(z).any(), "Tensor z contains NaN values"
            out.append(z)
        return out
    
    def encode_node_curv(self,node_attr):
        embeds = self.general_fourier_mapping()
        emb_list = []
        for emb in embeds:
            tmp_x = node_attr
            tmp_x = tmp_x@emb
            emb_list.append(tmp_x)
        emb_list.append(node_attr)
        embeds_tensor = torch.concat(emb_list,-1)
      
        embeds_tensor = self.encode_kernel_layer_node(embeds_tensor)
        embeds_tensor = self.encode_kernel_norm_node(embeds_tensor)
        return embeds_tensor
    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()

    def forward(self,batch,label=None, **kwargs):
        x = batch.x
        edge_index = batch.edge_index
        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_final = 0
        
        for i, local_conv in enumerate(self.local_convs):
            x = batch.x
            if self.res:
                x = local_conv(x, edge_index) + self.lins[i](x)
            else:
                x = local_conv(x, edge_index)
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            else:
                pass
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x
            batch.x = x_final
        batch.x = self.encode_node_curv(batch.x)
        return batch
    
    def encode(self,batch,label=None,**kwargs):
        return self.forward(batch,**kwargs)
    
    def decode(self,batch,task=None,split=None,**kwargs):
        mask = batch.get(split+'_mask', None)
        x_final = batch.x[mask]
        x = self.pred_local(x_final)
        
        return x
register_network('MPNNs', MPNNs)
register_network('MPNNs_curv',MPNNs_curv)
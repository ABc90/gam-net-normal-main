import torch
import torch.nn as nn

from .base import knn_group_v2, get_knn_idx, Conv1D


class Aggregator(nn.Module):
    def __init__(self, oper):
        super().__init__()
        assert oper in ('mean', 'sum', 'max')
        self.oper = oper

    def forward(self, x, dim=2):
        if self.oper == 'mean':
            return x.mean(dim=dim, keepdim=False)
        elif self.oper == 'sum':
            return x.sum(dim=dim, keepdim=False)
        elif self.oper == 'max':
            ret, _ = x.max(dim=dim, keepdim=False)
            return ret



class GraphConv2(nn.Module):
    def __init__(self, knn, in_channels, growth_rate, num_fc_layers, aggr='max', with_bn=True, with_relu=True, relative_feat_only=False):
        super().__init__()
        self.knn = knn
        self.in_channels = in_channels
        assert num_fc_layers > 2
        self.num_fc_layers = num_fc_layers
        self.growth_rate = growth_rate
        self.relative_feat_only = relative_feat_only
        # self.act = nn.Sigmoid()
        # self.act_mlp = nn.Linear(2, 1)

        if knn > 1:
            if relative_feat_only:
                self.layer_first = Conv1D(in_channels+3, growth_rate, with_bn=with_bn, with_relu=with_relu)
            else:
                self.layer_first = Conv1D(in_channels*3, growth_rate, with_bn=with_bn, with_relu=with_relu)
        else:
            self.layer_first = Conv1D(in_channels, growth_rate, with_bn=with_bn, with_relu=with_relu)

        self.layers_mid = nn.ModuleList()
        for i in range(1, num_fc_layers-1):
            self.layers_mid.append(Conv1D(in_channels + i * growth_rate, growth_rate, with_bn=with_bn, with_relu=with_relu))

        self.layer_last = Conv1D(in_channels + (num_fc_layers - 1) * growth_rate, growth_rate, with_bn=with_bn, with_relu=with_relu)

        dim = in_channels + num_fc_layers * growth_rate
        self.layer_out = Conv1D(dim, dim, with_bn=False, with_relu=False)

        if knn > 1:
            self.aggr = Aggregator(aggr)

    @property
    def out_channels(self):
        return self.in_channels + self.num_fc_layers * self.growth_rate

    def get_edge_feature(self, x, pos, knn_idx):
        """
        x: (B, C, N)
        pos: (B, 3, N)
        knn_idx: (B, N, K)
        return: (B, C, N, K)
        """
        knn_feat = knn_group_v2(x, knn_idx)                  # (B, C, N, K)
        x_tiled = x.unsqueeze(-1).expand_as(knn_feat)
        '''######*Test_GAM*#####
        # gradient only: 9.96 -> 9.95
        B, D, N, M = knn_feat.size()  # (B, C, N, K)
        knn_pos = knn_group_v2(pos, knn_idx)
        pos_tiled = pos.unsqueeze(-1)
        pos_norm = knn_pos - pos_tiled
        pos_norm = pos_norm.reshape(-1, 3)
        distance = torch.clamp(pos_norm.norm(dim=-1, keepdim=True), min=1e-16)
        xy_distance = torch.clamp(pos_norm[:, :2].norm(dim=-1, keepdim=True), min=1e-16)
        gradient = (pos_norm[:, 2].unsqueeze(1) / distance) * (
                pos_norm[:, 0].unsqueeze(1) + pos_norm[:, 1].unsqueeze(1)) / xy_distance
        #attention = self.act((pos_norm[:, 2].unsqueeze(1) / distance) * (pos_norm[:, 0].unsqueeze(1) + pos_norm[:, 1].unsqueeze(1)) / xy_distance) * 0.5
        attention = self.act(self.act_mlp(torch.cat([gradient, distance], dim=-1)))
        knn_feat = (knn_feat.reshape(-1, D) * attention + knn_feat.reshape(-1, D)).reshape(B, D, N, M)
        ####################'''
        if self.relative_feat_only:
            knn_pos = knn_group_v2(pos, knn_idx)
            pos_tiled = pos.unsqueeze(-1)
            edge_feat = torch.cat([knn_pos - pos_tiled, knn_feat - x_tiled], dim=1)  # (B, 3+C, N, K)
        else:
            edge_feat = torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=1)    # (B, 3*C, N, K)

        return edge_feat

    def forward(self, x, pos, knn_idx):
        """
        x: (B, C, N)
        pos: (B, 3, N)
        return y: (B, C, N)
          knn_idx: (B, N, K)
        """
        B, C, N = x.shape
        K = self.knn

        if knn_idx is None and self.knn > 1:
            pos_t = pos.transpose(2, 1)                                   # (B, N, 3)
            knn_idx = get_knn_idx(pos_t, pos_t, k=self.knn, offset=1)     # (B, N, K)

        if K > 1:
            edge_feat = self.get_edge_feature(x, pos, knn_idx=knn_idx)    # (B, C, N, K)
            edge_feat = edge_feat.view(B, -1, N * K)                      # (B, C, N*K)
            x = x.unsqueeze(-1).repeat(1, 1, 1, K).view(B, C, N * K)      # (B, C, N*K)
        else:
            edge_feat = x

        # First Layer
        y = torch.cat([
            self.layer_first(edge_feat),     # (B, c, N*K)
            x,                               # (B, d, N*K)
        ], dim=1)                            # (B, c+d, N*K)

        # Intermediate Layers
        for layer in self.layers_mid:
            y = torch.cat([
                layer(y),                    # (B, c, N*K)
                y                            # (B, c+d, N*K)
            ], dim=1)                        # (B, d+(L-1)*c, N*K)

        # Last Layer
        y = torch.cat([
            self.layer_last(y),              # (B, c, N*K)
            y                                # (B, d+(L-1)*c, N*K)
        ], dim=1)                            # (B, d+L*c, N*K)

        # y = torch.cat([y, x], dim=1)
        y = self.layer_out(y)                # (B, C, N*K)

        # Pooling Layer
        if K > 1:
            y = y.reshape(B, -1, N, K)
            y = self.aggr(y, dim=-1)             # (B, C, N)

        return y, knn_idx


class GraphConv(nn.Module):
    def __init__(self, knn, in_channels, growth_rate, num_fc_layers, aggr='max', with_bn=True, with_relu=True, relative_feat_only=False):
        super().__init__()
        self.knn = knn
        self.in_channels = in_channels
        assert num_fc_layers > 2
        self.num_fc_layers = num_fc_layers
        self.growth_rate = growth_rate
        self.relative_feat_only = relative_feat_only

        if knn > 1:
            if relative_feat_only:
                self.layer_first = Conv1D(in_channels+3, growth_rate, with_bn=with_bn, with_relu=with_relu)
            else:
                self.layer_first = Conv1D(in_channels*3, growth_rate, with_bn=with_bn, with_relu=with_relu)
        else:
            self.layer_first = Conv1D(in_channels, growth_rate, with_bn=with_bn, with_relu=with_relu)

        self.layers_mid = nn.ModuleList()
        for i in range(1, num_fc_layers-1):
            self.layers_mid.append(Conv1D(in_channels + i * growth_rate, growth_rate, with_bn=with_bn, with_relu=with_relu))

        self.layer_last = Conv1D(in_channels + (num_fc_layers - 1) * growth_rate, growth_rate, with_bn=with_bn, with_relu=with_relu)

        dim = in_channels + num_fc_layers * growth_rate
        self.layer_out = Conv1D(dim, dim, with_bn=False, with_relu=False)

        if knn > 1:
            self.aggr = Aggregator(aggr)

    @property
    def out_channels(self):
        return self.in_channels + self.num_fc_layers * self.growth_rate

    def get_edge_feature(self, x, pos, knn_idx):
        """
        x: (B, C, N)
        pos: (B, 3, N)
        knn_idx: (B, N, K)
        return: (B, C, N, K)
        """
        knn_feat = knn_group_v2(x, knn_idx)                  # (B, C, N, K)
        x_tiled = x.unsqueeze(-1).expand_as(knn_feat)
        if self.relative_feat_only:
            knn_pos = knn_group_v2(pos, knn_idx)
            pos_tiled = pos.unsqueeze(-1)
            edge_feat = torch.cat([knn_pos - pos_tiled, knn_feat - x_tiled], dim=1)
        else:
            edge_feat = torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=1)
        return edge_feat

    def forward(self, x, pos, knn_idx):
        """
        x: (B, C, N)
        pos: (B, 3, N)
        return y: (B, C, N)
          knn_idx: (B, N, K)
        """
        B, C, N = x.shape
        K = self.knn

        if knn_idx is None and self.knn > 1:
            pos_t = pos.transpose(2, 1)                                   # (B, N, 3)
            knn_idx = get_knn_idx(pos_t, pos_t, k=self.knn, offset=1)     # (B, N, K)

        if K > 1:
            edge_feat = self.get_edge_feature(x, pos, knn_idx=knn_idx)    # (B, C, N, K)
            edge_feat = edge_feat.view(B, -1, N * K)                      # (B, C, N*K)
            x = x.unsqueeze(-1).repeat(1, 1, 1, K).view(B, C, N * K)      # (B, C, N*K)
        else:
            edge_feat = x

        # First Layer
        y = torch.cat([
            self.layer_first(edge_feat),     # (B, c, N*K)
            x,                               # (B, d, N*K)
        ], dim=1)                            # (B, c+d, N*K)

        # Intermediate Layers
        for layer in self.layers_mid:
            y = torch.cat([
                layer(y),                    # (B, c, N*K)
                y                            # (B, c+d, N*K)
            ], dim=1)                        # (B, d+(L-1)*c, N*K)

        # Last Layer
        y = torch.cat([
            self.layer_last(y),              # (B, c, N*K)
            y                                # (B, d+(L-1)*c, N*K)
        ], dim=1)                            # (B, d+L*c, N*K)

        # y = torch.cat([y, x], dim=1)
        y = self.layer_out(y)                # (B, C, N*K)

        # Pooling Layer
        if K > 1:
            y = y.reshape(B, -1, N, K)
            y = self.aggr(y, dim=-1)             # (B, C, N)

        return y, knn_idx


class EncodeNet(nn.Module):
    def __init__(self,
        knn,
        num_convs=1,
        in_channels=3,
        conv_channels=24,
        growth_rate=12,
        num_fc_layers=3,
    ):
        super().__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels

        self.trans = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            tran = Conv1D(in_channels, conv_channels, with_bn=True, with_relu=True)
            conv = GraphConv(
                knn=knn,
                in_channels=conv_channels,
                growth_rate=growth_rate,
                num_fc_layers=num_fc_layers,
                relative_feat_only=(i == 0),
            )
            self.trans.append(tran)
            self.convs.append(conv)
            in_channels = conv.out_channels

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def forward(self, x, pos, knn_idx):
        """
        x: (B, C, N)
        pos: (B, 3, N)
        return: (B, C, N), C = conv_channels+num_fc_layers*growth_rate
        """
        for i in range(self.num_convs):
            x = self.trans[i](x)
            x, knn_idx = self.convs[i](x, pos=pos, knn_idx=knn_idx)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SubFold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()
        self.in_channel = in_channel
        self.step = step
        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x, c):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = c.to(x.device) # b 3 n2
        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        return fd2


class GeoCrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=1, qkv_bias=False, qk_scale=1, attn_drop=0., proj_drop=0.,
                 aggregate_dim=16):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Identity()  # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Identity()  # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.x_map = nn.Identity()  # nn.Linear(aggregate_dim, 1)

    def forward(self, q, k, v):
        B, N, _ = q.shape
        C = self.out_dim
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B, NK, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, 3)

        x = self.x_map(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncodeNet2(nn.Module):
    def __init__(self,
        knn,
        num_convs=1,
        in_channels=3,
        conv_channels=24,
        growth_rate=12,
        num_fc_layers=3,
    ):
        super().__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels

        self.trans = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            tran = Conv1D(in_channels, conv_channels, with_bn=True, with_relu=True)
            conv = GraphConv2(
                knn=knn,
                in_channels=conv_channels,
                growth_rate=growth_rate,
                num_fc_layers=num_fc_layers,
                relative_feat_only=(i == 0),
            )
            self.trans.append(tran)
            self.convs.append(conv)
            in_channels = conv.out_channels

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def forward(self, x, pos, knn_idx):
        """
        x: (B, C, N)
        pos: (B, 3, N)
        return: (B, C, N), C = conv_channels+num_fc_layers*growth_rate
        """
        for i in range(self.num_convs):
            x = self.trans[i](x)
            x, knn_idx = self.convs[i](x, pos=pos, knn_idx=knn_idx)
        return x


class GraphConv3(nn.Module):
    def __init__(self, knn, in_channels, growth_rate, num_fc_layers, aggr='max', with_bn=True, with_relu=True, relative_feat_only=False):
        super().__init__()
        self.knn = knn
        self.in_channels = in_channels
        assert num_fc_layers > 2
        self.num_fc_layers = num_fc_layers
        self.growth_rate = growth_rate
        self.relative_feat_only = relative_feat_only
        self.act = nn.Sigmoid()
        self.act_mlp = nn.Linear(2, 1)

        if knn > 1:
            if relative_feat_only:
                self.layer_first = Conv1D(in_channels+3, growth_rate, with_bn=with_bn, with_relu=with_relu)
            else:
                self.layer_first = Conv1D(in_channels*3, growth_rate, with_bn=with_bn, with_relu=with_relu)
        else:
            self.layer_first = Conv1D(in_channels, growth_rate, with_bn=with_bn, with_relu=with_relu)

        self.layers_mid = nn.ModuleList()
        for i in range(1, num_fc_layers-1):
            self.layers_mid.append(Conv1D(in_channels + i * growth_rate, growth_rate, with_bn=with_bn, with_relu=with_relu))

        self.layer_last = Conv1D(in_channels + (num_fc_layers - 1) * growth_rate, growth_rate, with_bn=with_bn, with_relu=with_relu)

        dim = in_channels + num_fc_layers * growth_rate
        self.layer_out = Conv1D(dim, dim, with_bn=False, with_relu=False)

        if knn > 1:
            self.aggr = Aggregator(aggr)

    @property
    def out_channels(self):
        return self.in_channels + self.num_fc_layers * self.growth_rate

    def get_edge_feature(self, x, pos, knn_idx):
        """
        x: (B, C, N)
        pos: (B, 3, N)
        knn_idx: (B, N, K)
        return: (B, C, N, K)
        """
        knn_feat = knn_group_v2(x, knn_idx)                  # (B, C, N, K)
        x_tiled = x.unsqueeze(-1).expand_as(knn_feat)
        ######*Test_GAM*#####
        # gradient only: 9.96 -> 9.95
        B, D, N, M = knn_feat.size()  # (B, C, N, K)
        knn_pos = knn_group_v2(pos, knn_idx)
        pos_tiled = pos.unsqueeze(-1)
        pos_norm = knn_pos - pos_tiled
        pos_norm = pos_norm.reshape(-1, 3)
        distance = torch.clamp(pos_norm.norm(dim=-1, keepdim=True), min=1e-16)
        xy_distance = torch.clamp(pos_norm[:, :2].norm(dim=-1, keepdim=True), min=1e-16)
        gradient = (pos_norm[:, 2].unsqueeze(1) / distance) * (
                pos_norm[:, 0].unsqueeze(1) + pos_norm[:, 1].unsqueeze(1)) / xy_distance
        #attention = self.act((pos_norm[:, 2].unsqueeze(1) / distance) * (pos_norm[:, 0].unsqueeze(1) + pos_norm[:, 1].unsqueeze(1)) / xy_distance) * 0.5
        attention = self.act(self.act_mlp(torch.cat([gradient, distance], dim=-1)))
        knn_feat = (knn_feat.reshape(-1, D) * attention + knn_feat.reshape(-1, D)).reshape(B, D, N, M)
        ####################
        if self.relative_feat_only:
            knn_pos = knn_group_v2(pos, knn_idx)
            pos_tiled = pos.unsqueeze(-1)
            edge_feat = torch.cat([knn_pos - pos_tiled, knn_feat - x_tiled], dim=1)  # (B, 3+C, N, K)
        else:
            edge_feat = torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=1)    # (B, 3*C, N, K)

        return edge_feat, attention.reshape(B, 1, N, M), gradient.reshape(B, 1, N, M),knn_pos

    def forward(self, x, pos, knn_idx):
        """
        x: (B, C, N)
        pos: (B, 3, N)
        return y: (B, C, N)
          knn_idx: (B, N, K)
        """
        B, C, N = x.shape
        K = self.knn

        if knn_idx is None and self.knn > 1:
            pos_t = pos.transpose(2, 1)                                   # (B, N, 3)
            knn_idx = get_knn_idx(pos_t, pos_t, k=self.knn, offset=1)     # (B, N, K)

        if K > 1:
            edge_feat, aa, gg, knn_pos = self.get_edge_feature(x, pos, knn_idx=knn_idx)    # (B, C, N, K)
            edge_feat = edge_feat.view(B, -1, N * K)                      # (B, C, N*K)
            x = x.unsqueeze(-1).repeat(1, 1, 1, K).view(B, C, N * K)      # (B, C, N*K)
        else:
            edge_feat = x

        # First Layer
        y = torch.cat([
            self.layer_first(edge_feat),     # (B, c, N*K)
            x,                               # (B, d, N*K)
        ], dim=1)                            # (B, c+d, N*K)

        # Intermediate Layers
        for layer in self.layers_mid:
            y = torch.cat([
                layer(y),                    # (B, c, N*K)
                y                            # (B, c+d, N*K)
            ], dim=1)                        # (B, d+(L-1)*c, N*K)

        # Last Layer
        y = torch.cat([
            self.layer_last(y),              # (B, c, N*K)
            y                                # (B, d+(L-1)*c, N*K)
        ], dim=1)                            # (B, d+L*c, N*K)

        # y = torch.cat([y, x], dim=1)
        y = self.layer_out(y)                # (B, C, N*K)

        # Pooling Layer
        if K > 1:
            y = y.reshape(B, -1, N, K)
            y = self.aggr(y, dim=-1)             # (B, C, N)

        return y, knn_idx, aa, gg, knn_pos

class EncodeNet3(nn.Module):
    def __init__(self,
        knn,
        num_convs=1,
        in_channels=3,
        conv_channels=24,
        growth_rate=12,
        num_fc_layers=3,
    ):
        super().__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels

        self.trans = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            tran = Conv1D(in_channels, conv_channels, with_bn=True, with_relu=True)
            conv = GraphConv3(
                knn=knn,
                in_channels=conv_channels,
                growth_rate=growth_rate,
                num_fc_layers=num_fc_layers,
                relative_feat_only=(i == 0),
            )
            self.trans.append(tran)
            self.convs.append(conv)
            in_channels = conv.out_channels

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def forward(self, x, pos, knn_idx):
        """
        x: (B, C, N)
        pos: (B, 3, N)
        return: (B, C, N), C = conv_channels+num_fc_layers*growth_rate
        """
        aa_out = []
        gg_out = []
        knn_pos_out = []
        for i in range(self.num_convs):
            x = self.trans[i](x)
            x, knn_idx , aa, gg, knn_pos = self.convs[i](x, pos=pos, knn_idx=knn_idx)
            aa_out.append(aa)
            gg_out.append(gg)
            knn_pos_out.append(knn_pos)
        return x, aa_out, gg_out, knn_pos_out
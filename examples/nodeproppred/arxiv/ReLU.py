import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot
import torch
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch import Tensor
from torch_sparse import SparseTensor, set_diag
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree


class EdgeReluV2(MessagePassing):
    def __init__(self, channels,
                 k=2, reduction=2,
                 add_self_loops=True,
                 negative_slope=0.2,
                 **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(EdgeReluV2, self).__init__(node_dim=0, **kwargs)
        self.negative_slope = negative_slope
        self.channels = channels
        self.k = k
        self.add_self_loops = add_self_loops
        self.fc1 = GCNConv(channels,  channels // reduction)

        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor(
            [1.] + [0.]*(2*k - 1)).float())

        self.att_l = Parameter(torch.Tensor(1, channels))
        self.att_r = Parameter(torch.Tensor(1, channels))
        self.reset_parameters()

    def get_relu_coefs(self, x, edge_index):
        theta = x
        theta = self.fc1(theta, edge_index)
        theta = theta.mean(dim=0)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        # theta = 2 * self.sigmoid(theta) - 1
        theta = F.tanh(theta)
        return theta

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)

    def forward(self, x, edge_index, size=None):
        x_l = x_r = x
        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)
        # print(alpha_l.shape)
        # print(alpha_r.shape)
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)
        
        _, col = edge_index[0], edge_index[1]
        self.degree = degree(col)

        theta = self.get_relu_coefs(x, edge_index)
        self.theta = theta.view(-1, self.channels, 2 * self.k)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)
        return out

    def message(self,  x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)

        gamma = self.degree[index]/3
        alpha = alpha / 10
        alpha = softmax(alpha, index, ptr, size_i) * gamma

        alpha = torch.min(alpha, torch.ones_like(alpha))

        alpha = alpha.view(-1, 1, 1)
        # relu_coefs = (alpha * self.theta) * self.lambdas + self.init_v
        relu_coefs = (self.theta * self.lambdas + self.init_v) * alpha
        # relu_coefs = F.dropout(relu_coefs, 0.2, training=self.training)

        x = x_j
        x = x.unsqueeze(-1)
        x_perm = x.permute(2, 0, 1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        result = torch.max(output, dim=-1)[0].permute(1, 2, 0).squeeze()
        return result


class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.fc1 = GCNConv(channels,  channels // reduction)
#         self.fc1 = GraphConv(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor(
            [1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x, edge_index):
        theta = x
        theta = self.fc1(theta, edge_index)
        theta = theta.mean(dim=0)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x, edge_index):
        raise NotImplementedError


class DyReLUC(DyReLU):
    def __init__(self, channels, reduction=4, k=2):
        super(DyReLUC, self).__init__(channels, reduction, k)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)
#         self.pos = GraphConv(channels, 1)
        self.pos = GCNConv(channels,  1)
        self.dropout = nn.Dropout(0.2)

    def pos_coefs(self, x, edge_index):
        x = self.pos(x, edge_index)
        x = x.squeeze()
        x = x / 10
        x_norm = F.softmax(x).view(-1, 1)
        x_norm = x_norm * (x.shape[0]/3)
        return torch.min(x_norm, torch.ones_like(x_norm))

    def forward(self, x, edge_index):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x, edge_index)
        relu_coefs = theta.view(-1, self.channels, 2 * self.k)
        pos_norm_coefs = self.pos_coefs(x, edge_index).view(-1, 1, 1)
        relu_coefs = relu_coefs * pos_norm_coefs * self.lambdas + self.init_v
        # relu_coefs = F.dropout(relu_coefs, 0.2, training=self.training)
        x = x.unsqueeze(-1)
        x_perm = x.permute(2, 0, 1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        result = torch.max(output, dim=-1)[0].permute(1, 2, 0).squeeze()
        self.coefs = relu_coefs
        return result

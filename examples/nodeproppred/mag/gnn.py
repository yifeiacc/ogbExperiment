import argparse

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
from ReLU import EdgeReluV2, DyReLUC


class xReLU(torch.nn.Module):
    def __init__(self, kind):
        super(xReLU, self).__init__()
        self.kind = kind
        self.PReLU = torch.nn.PReLU()

    def forward(self, x, edge_index):
        if self.kind == "ReLU":
            return F.relu(x)
        elif self.kind == "PReLU":
            return self.PReLU(x)
        elif self.kind == "ELU":
            return F.elu(x, alpha=1)
        elif self.kind == "LReLU":
            return F.leaky_relu(x, negative_slope=0.01)
        else:
            return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))

        self.ReLU = torch.nn.ModuleList()
        if self.kind == "ReLU":
            self.ReLU.append(xReLU("ReLU"))
        elif self.kind == "ELU":
            self.ReLU.append(xReLU("ELU"))
        elif self.kind == "PReLU":
            self.ReLU.append(xReLU("PReLU"))
        elif self.kind == "LReLU":
            self.ReLU.append(xReLU("LReLU"))
        elif self.kind == "GraphReLUNode":
            self.ReLU.append(DyReLUC(hidden_channels))
        elif self.kind == "GraphReLUEdge":
            self.ReLU.append(EdgeReluV2(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
            if self.kind == "ReLU":
                self.ReLU.append(xReLU("ReLU"))
            elif self.kind == "ELU":
                self.ReLU.append(xReLU("ELU"))
            elif self.kind == "PReLU":
                self.ReLU.append(xReLU("PReLU"))
            elif self.kind == "LReLU":
                self.ReLU.append(xReLU("LReLU"))
            elif self.kind == "GraphReLUNode":
                self.ReLU.append(DyReLUC(hidden_channels))
            elif self.kind == "GraphReLUEdge":
                self.ReLU.append(EdgeReluV2(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.ReLU[i](x, adj_t)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.ReLU = torch.nn.ModuleList()
        if self.kind == "ReLU":
            self.ReLU.append(xReLU("ReLU"))
        elif self.kind == "ELU":
            self.ReLU.append(xReLU("ELU"))
        elif self.kind == "PReLU":
            self.ReLU.append(xReLU("PReLU"))
        elif self.kind == "LReLU":
            self.ReLU.append(xReLU("LReLU"))
        elif self.kind == "GraphReLUNode":
            self.ReLU.append(DyReLUC(hidden_channels))
        elif self.kind == "GraphReLUEdge":
            self.ReLU.append(EdgeReluV2(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

            if self.kind == "ReLU":
                self.ReLU.append(xReLU("ReLU"))
            elif self.kind == "ELU":
                self.ReLU.append(xReLU("ELU"))
            elif self.kind == "PReLU":
                self.ReLU.append(xReLU("PReLU"))
            elif self.kind == "LReLU":
                self.ReLU.append(xReLU("LReLU"))
            elif self.kind == "GraphReLUNode":
                self.ReLU.append(DyReLUC(hidden_channels))
            elif self.kind == "GraphReLUEdge":
                self.ReLU.append(EdgeReluV2(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.ReLU[i](x, adj_t)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-MAG (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-mag')
    rel_data = dataset[0]

    # We are only interested in paper <-> paper relations.
    data = Data(
        x=rel_data.x_dict['paper'],
        edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
        y=rel_data.y_dict['paper'])

    data = T.ToSparseTensor()(data)
    data.adj_t = data.adj_t.to_symmetric()

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']['paper'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    data = data.to(device)

    evaluator = Evaluator(name='ogbn-mag')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()

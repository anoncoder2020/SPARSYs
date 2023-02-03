import os
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
import networkx as nx
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from itertools import combinations
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.utils import remove_self_loops, add_self_loops, to_networkx, to_undirected
import math


def remove_in_clique(clique_list, graph, pecentage, pencent_cliques):
    total_remove = 0
    removed_edegs=[]
    num_clique = int(len(clique_list)*pencent_cliques)
    total_cliques = 0
    list_idx = 0
    while total_cliques < num_clique:
        if list_idx == len(clique_list):
            break
        edges_list = []
        clique = clique_list[list_idx]
        for c in combinations(clique, 2):
            edges_list.append(c)
        overlap = list(set(removed_edegs).intersection(set(edges_list)))
        if len(overlap) > 0:
            list_idx+=1
            continue
        num_remove_edges = int(len(edges_list) * pecentage)
        if num_remove_edges > 0:
            total_remove += num_remove_edges
            idx = np.arange(len(edges_list))
            remove_idx = np.random.choice(idx, num_remove_edges, replace=False)
            remove_edges_tuple = np.array(edges_list)[remove_idx]
            for i in range(remove_edges_tuple.shape[0]):
                removed_edegs.append(tuple(remove_edges_tuple[i]))
            graph.remove_edges_from([tuple(remove_edges_tuple[re_id]) for re_id in range(remove_edges_tuple.shape[0])])
        total_cliques += 1
        list_idx += 1
    return graph, nx.number_of_edges(graph)


def remove_in_cycle(graph, percentage, percent_cycle):
    cycle_edge_list = nx.cycle_basis(graph)
    num_cycle = math.floor(len(cycle_edge_list) * percent_cycle)
    total_cycle = 0
    while total_cycle < num_cycle:
        try:
            cycle = nx.find_cycle(graph, orientation='ignore')
        except Exception as NetworkXNoCycle:
            break
        idx = np.arange(len(cycle))
        remove_edges = np.random.choice(idx, int(len(cycle)*percentage), replace=False)
        #for loop remove
        for r in remove_edges.tolist():
            graph.remove_edge(cycle[r][0], cycle[r][1])
        total_cycle += 1
    return graph, nx.number_of_edges(graph)


def remove_graph_edges(graph, num_top_k, pruned_ratio):
    node_degree = [(n, d) for n, d in graph.degree()]
    node_degree.sort(key=lambda x: (x[1], x[0]), reverse=True)
    cut_edge_nodes = node_degree[0:num_top_k]
    for (node, _) in cut_edge_nodes:
        neighs = [n for n in graph.neighbors(node)]
        degree = list(graph.degree(neighs))
        degree.sort(key=lambda x: (x[1], x[0]), reverse=True)
        cut_num_edges = int(graph.degree[node]*pruned_ratio)
        for (neigh, _) in degree[0: cut_num_edges]:
            graph.remove_edge(node, neigh)
    return graph, nx.number_of_edges(graph)


def remove_between_clusters(cluster, graph, percentage):
    cluster1 = list(np.where(cluster == 0)[0])
    cluster2 = list(np.where(cluster == 1)[0])
    C1 = nx.subgraph(graph, cluster1)
    C2 = nx.subgraph(graph, cluster2)
    overlap = [e for e in graph.edges
               if (e[0] in C1 and e[1] in C2)
               or (e[0] in C2 and e[1] in C1)]
    overlap_idx = np.arange(len(overlap))
    print('vol in c1: ', nx.volume(C1, cluster1))
    print('vol in c2: ', nx.volume(C2, cluster2))
    print('2*overlap: ', 2*len(overlap))
    remove_idx = list(np.random.choice(overlap_idx, int(len(overlap) * percentage), replace=False))
    for idx in remove_idx:
        graph.remove_edge(*overlap[idx])
    return graph, nx.number_of_edges(graph)



def load_data(dataset, which_run, method, percent_edges, size_clique, top_k_degree, percent_cycle, pencent_cliques):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', dataset)

    if dataset in ["Citeseer"]:
        data = Planetoid(path, dataset, split='public', transform=T.NormalizeFeatures())[0]
        if dataset == 'Citeseer':
            data.num_classes = 6
        if percent_edges != 0:
            graph = to_networkx(data, to_undirected=True)
            num_edge = nx.number_of_edges(graph)
            num_node = nx.number_of_nodes(graph)
            if method == 'metis_sp':
                print('metis_sp')
                cluster = np.load('../metis/'+dataset+'_metis.npy')
                new_graph, num_update_edge = remove_between_clusters(cluster, graph, percent_edges)
            elif method == 'km_sp':
                print('km_sp')
                cluster = np.load('../km/'+dataset + '_km.npy')
                new_graph, num_update_edge = remove_between_clusters(cluster, graph, percent_edges)
            elif method == 'bm_sp':
                print('bm_sp')
                cluster = np.load('../bm/'+dataset + '_bm.npy')
                new_graph, num_update_edge = remove_between_clusters(cluster, graph, percent_edges)
            elif method == 'cycle_sp':
                print('cycle sparsy')
                new_graph, num_update_edge = remove_in_cycle(graph, percent_edges, percent_cycle)
            elif method == 'node_sp':
                print('node sparsy')
                new_graph, num_update_edge = remove_graph_edges(graph, num_top_k=int(num_node * top_k_degree),
                                                                pruned_ratio=percent_edges)
            elif method == 'clique_sp':
                print('clique sparsy')
                clique_list = [clique for clique in nx.find_cliques(graph) if len(clique) == size_clique]
                new_graph, num_update_edge = remove_in_clique(clique_list, graph, percent_edges, pencent_cliques)
            else:
                raise NotImplementedError("please specify `sparsy_type`")
            print("remove #: ", num_update_edge)
            print("sparsity: ", ((num_edge - num_update_edge) / num_edge) * 100)
            new_adj = nx.to_scipy_sparse_matrix(new_graph)
            data.edge_index = from_scipy_sparse_matrix(new_adj)[0]
            data = T.ToSparseTensor()(data)
            row, col, edge_attr = data.adj_t.t().coo()
            data.edge_index = torch.stack([row, col], dim=0)

    else:
        raise Exception('the dataset of {} has not been implemented'.format(dataset))

    num_nodes = data.x.size(0)
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
    if isinstance(edge_index, tuple):
        data.edge_index = edge_index[0]
    else:
        data.edge_index = edge_index
    return data


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for bn in self.bns:
        #     bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            # x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, optimizer):
    model.train()
    raw_logits = model(data.x, data.edge_index)
    logits = F.log_softmax(raw_logits[data.train_mask], 1)
    loss = F.nll_loss(logits, data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()

@torch.no_grad()
def test(model, data):
    model.eval()

    logits = model(data.x, data.edge_index)
    logits = F.log_softmax(logits, 1)
    acc_train = evaluate(logits, data.y, data.train_mask)
    acc_val = evaluate(logits, data.y, data.val_mask)
    acc_test = evaluate(logits, data.y, data.test_mask)
    return acc_train, acc_val, acc_test


def main():
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="Citeseer", required=False,
                        help="The input dataset.")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--top_k_degree', type=float, default=0.0)
    parser.add_argument('--size_clique', type=int, default=0)
    parser.add_argument('--percent_edges', type=float, default=0.0)
    parser.add_argument('--percent_cycle', type=float, default=0.0)
    parser.add_argument('--pencent_cliques', type=float, default=0.0)
    parser.add_argument('--method', type=str, default='node_sp')#cycle_sp, clique_sp
    args = parser.parse_args()
    print(args)

    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    # device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    max_acc = 0
    for run in range(args.runs):
        data = load_data(args.dataset, run, args.method, args.percent_edges, args.size_clique,
                         args.top_k_degree, args.percent_cycle, args.pencent_cliques)

        data = data.to(device)

        # seperate nodes into different clusters in train_idx based on the cluster result
        model = GCN(data.num_features, args.hidden_channels,
                    data.num_classes, args.num_layers,
                    args.dropout).to(device)
        model.reset_parameters()
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1, 1 + args.epochs):
            #for different training cluster: get subgraph
            loss = train(model, data, optimizer)
            result = test(model, data)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                if max_acc < test_acc:
                    max_acc = test_acc
                print('Run: {:02d}, Epoch: {:02d}, Loss: {:.4f}, Train: {:.2f}%, Valid: {:.2f}%, Test: {:.2f}%'.format(run + 1, epoch, loss, 100 * train_acc, 100 * valid_acc, 100 * test_acc))
    print("max accuracy: ", max_acc)

if __name__ == "__main__":
    main()

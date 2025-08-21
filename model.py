import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.parameter import Parameter
import math
from torch_geometric.utils import add_self_loops,remove_self_loops,softmax

from torch_geometric.nn import SGConv, sequential

EPS = 1e-15
class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.network(x)




class GraphEncode(nn.Module):
    def __init__(self,in_dim,out_dim,layer = 1 ):
        super(GraphEncode, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer = layer

        self.graphEnc = nn.ModuleList([sequential.Sequential("x,edge_index,edge_weight",[
            (SGConv(self.in_dim,self.out_dim),'x,edge_index,edge_weight -> x1'),
            nn.ReLU()
        ])for _ in range(self.layer)])

    def forward(self,x,edge_index,edge_weight):

        embbded = x.clone()
        for i in range(self.layer):
            embbded = self.graphEnc[i](embbded,edge_index,edge_weight)
        return embbded



class Model(torch.nn.Module):

    def __init__(self, args):
        """
        :param args: arguments dictionary
        """

        super(Model, self).__init__()
        self.args = args
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.pert_emb_lambda = 0.2


        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])


        self.G_sim = args['G_go'].to(args['device'])

        self.G_sim_weight = args['G_go_weight'].to(args['device'])


        # gene/globel perturbation embedding dictionary lookup
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)

        self.ctrl_enc = MLP([self.num_genes, hidden_size, hidden_size], last_layer_act="ReLU")


        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse_cell = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.fnn_de = MLP([hidden_size * 2, hidden_size * 2, hidden_size], last_layer_act='ReLU')

        self.coexp_Enc = GraphEncode(hidden_size,hidden_size)
        self.go_Enc = GraphEncode(hidden_size,hidden_size)

        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)


        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_cross = nn.BatchNorm1d(hidden_size * 2)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)

    def forward(self, data):
        """
        Forward pass of the model
        """
        x, pert_idx, y = data.x, data.pert_idx, data.y
        if self.no_perturb:
            out = x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique())

            x = x.view(num_graphs, -1)

            ctrl_emb = self.ctrl_enc(x)

            ## get base gene embeddings
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).to(self.args['device']))
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)

            base_emb = self.coexp_Enc(base_emb,self.G_coexpress,self.G_coexpress_weight)

            ## get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T

            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))

            pert_global_emb = self.go_Enc(pert_global_emb,self.G_sim,self.G_sim_weight)


            base_emb = self.emb_trans_v2(base_emb)

            base_emb = base_emb.repeat(num_graphs, 1)

            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)


            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track_cell = {}
                # non_ctrl = []
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track_cell:
                        pert_track_cell[j.item()] = pert_track_cell[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track_cell[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track_cell.values())) > 0:
                    if len(list(pert_track_cell.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track_cell.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track_cell.values())))

                    for idx, j in enumerate(pert_track_cell.keys()):

                        ctrl_emb[j] = ctrl_emb[j] + emb_total[idx]
                        base_emb[j] = base_emb[j] + emb_total[idx]



            ctrl_emb = ctrl_emb.repeat(1, self.num_genes)
            ctrl_emb = ctrl_emb.view(num_graphs, self.num_genes, -1)


            cross_emb = torch.cat([ctrl_emb, base_emb], dim=2)

            cross_emb = cross_emb.reshape(num_graphs * self.num_genes, -1)

            cross_emb = self.bn_cross(cross_emb)

            cross_emb = self.transform(cross_emb)


            out = self.fnn_de(cross_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis=2)
            out = w + self.indv_b1

            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1, 1)

            out = torch.split(torch.flatten(out), self.num_genes)


            return torch.stack(out)


import numpy as np
import torch
import torch.nn as nn
from stdmae.stdmae_arch.graphwavenet.net.AGCRNCell_v8 import AGCRNCell_v8


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell_v8(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell_v8(node_num, dim_out, dim_out, cheb_k, embed_dim))


    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN_v10(nn.Module):
    # def __init__(self, args):
    def __init__(self, node_num,input_dim,rnn_units,output_dim,cheb_k,embed_dim,num_layers,horizon,default_graph):

        super(AGCRN_v10, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers

        self.default_graph = default_graph

        # self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)
        self.node_embeddings = nn.Parameter(torch.randn(self.node_num, embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(node_num, input_dim, rnn_units, cheb_k,
                                embed_dim, num_layers)

        self.fc_his_t = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.fc_his_s = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())

        self.leckrule = nn.LeakyReLU()
        #predictor
        # self.end_conv = nn.Conv2d(1, horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        print("------output_dim-------",output_dim)
        print("------horizon * self.output_dim-------",horizon * self.output_dim)
        self.encoder_conv = nn.Conv2d(horizon, 256, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv = nn.Conv2d(256, horizon * self.output_dim, kernel_size=(1, 1), bias=True)

    # def forward(self, source, targets, teacher_forcing_ratio=0.5):
    def forward(self, source,hidden_states):

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        # print("----------output-----------", output.shape)
        #
        # print("----------hidden_states[:, :, :96]-----------",hidden_states[:, :, :96].shape)
        # hidden_states_t,hidden_states_s = self.cross(hidden_states[:, :, :96],hidden_states[:, :, 96:])


        # hidden_states_t = self.fc_his_t(hidden_states[:, :, :96])  # B, N, D
        # print("----------hidden_states_t-----------",hidden_states_t.shape)
        # hidden_states_t = hidden_states_t.transpose(1, 2).unsqueeze(-1)
        #
        # hidden_states_s = self.fc_his_s(hidden_states[:, :, 96:])  # B, N, D
        # hidden_states_s = hidden_states_s.transpose(1, 2).unsqueeze(-1)

        hidden_states_t = self.fc_his_t(hidden_states[:, :, :96])  # B, N, D
        # print("----------hidden_states_t-----------",hidden_states_t.shape)
        hidden_states_t = hidden_states_t.transpose(1, 2).unsqueeze(-1)

        hidden_states_s = self.fc_his_s(hidden_states[:, :, 96:])  # B, N, D
        hidden_states_s = hidden_states_s.transpose(1, 2).unsqueeze(-1)



        output = self.encoder_conv((output))  # B, T*C, N, 1
        # print("-------output------",output.shape)
        # output = self.tca(output, hidden_states_t)
        # output = self.sca(output, hidden_states_s)
        output = self.leckrule(output * hidden_states_t) + output
        output = self.leckrule(output * hidden_states_s) + output



        output = self.end_conv(output)
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.node_num)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C
        output = output.squeeze(-1).transpose(1, 2)
        print("---------检测----运行---------", output.shape)
        return output


if __name__ == "__main__":

    net = AGCRN_v10(
    node_num=307,
    input_dim=2,
    rnn_units=64,
    output_dim=1,
    cheb_k=2,
    embed_dim=10,
    num_layers=2,
    horizon=12,
    default_graph=True,

    )
    input = torch.rand(4, 12, 307, 2)
    hidden = torch.rand(1, 307, 192)

    x = net(input,hidden)
    print("----------------最后输出----------------",x.shape)
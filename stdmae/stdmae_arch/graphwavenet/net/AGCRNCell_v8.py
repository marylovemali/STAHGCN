import torch
import torch.nn as nn
from .AGCN import AVWGCN

class AGCRNCell_v8(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell_v8, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)
        self.Linear = nn.Linear(128, 64)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)

        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))

        z_r = self.Linear(z_r)



        # candidate = torch.cat((x, z*state), dim=-1)
        candidate = torch.cat((x, z_r), dim=-1)

        hc = torch.tanh(self.update(candidate, node_embeddings))

        # h = r*state + (1-r)*hc


        return hc

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
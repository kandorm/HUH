import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from torch_geometric.data import Batch, Data
from torch_geometric.nn import RGCNConv, GraphConv
from submodels.context_feature_extractor import CNN_Embedding
torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from transformers import BertModel

class EdgeAttention(nn.Module):
    def __init__(self, input_dim, max_seq_len):
        super(EdgeAttention, self).__init__()
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.init_weight()
    def init_weight(self):
        init.xavier_normal_(self.scalar.weight)
    def forward(self, M, edge_ind):
        # M (batch, seq_len, input_dim)
        seq_len = M.size(1)
        scale = self.scalar(M) # (batch, seq_len, max_seq_len)
        scale = scale[:, :, :seq_len] # (batch, seq_len, seq_len)
        alpha = F.softmax(scale, dim=1).permute(0, 2, 1)
        """
        eye_ = []
        for b_i in range(len(edge_ind)):
            for e_i in range(seq_len):
                eye_.append([b_i, e_i, e_i])
        eye_ = np.array(eye_, dtype=np.int).transpose()
        alpha[eye_] = 0.0
        alpha = alpha / torch.clamp(alpha.sum(-1, keepdim=True), 1e-10, 1.0)
        """
        """
        mask = Variable(torch.zeros(alpha.size())).to(device)
        edge_ind_ = []
        for b_i in range(len(edge_ind)):
            for e_i in range(len(edge_ind[b_i][0])):
                edge_ind_.append([b_i, edge_ind[b_i][0][e_i], edge_ind[b_i][1][e_i]])
        edge_ind_ = np.array(edge_ind_, dtype=np.int).transpose()
        mask[edge_ind_] = 1
        masked_alpha = alpha * mask
        _sums = torch.clamp(masked_alpha.sum(-1, keepdim=True), 1e-10, 1.0)
        scores = masked_alpha.div(_sums)
        """
        return alpha


class GraphNetwork(nn.Module):
    def __init__(self, input_size, num_relations, hidden_size=800):
        super(GraphNetwork, self).__init__()
        self.conv1 = RGCNConv(input_size, hidden_size, num_relations, num_bases=8)
        #self.conv2 = GraphConv(hidden_size, hidden_size)
    def forward(self, x, edge_index, edge_type, edge_norm=None):
        out = self.conv1(x, edge_index, edge_type)
        #out = self.conv2(out, edge_index)
        return out

def euclidean_dist(x, y):
    # x (batch, out_size, utt, hidden)
    # y (batch, out_size, max_his, hidden)
    m, n = x.size(-2), y.size(-2)
    xx = torch.pow(x, 2).sum(-1, keepdim=True).repeat(1,1,1,n) # (batch, out, utt, max_his)
    yy = torch.pow(y, 2).sum(-1, keepdim=True).repeat(1,1,1,m).transpose(-1,-2) 
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.transpose(-1, -2))
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

class HUH(nn.Module):
    def __init__(self, hidden_size, output_size, weights_matrix, n_filters, filter_sizes, c_dropout, l_dropout, utt_weight_matrix=None, max_utterance_num=10000):
        super(HUH, self).__init__()

        self.hidden_size = hidden_size
        self.his_hidden_size = hidden_size * 2
        self.graph_hidden_size = hidden_size
        self.output_size = output_size
        self.n_relations = 5
        self.max_utt_num = 90

        self.cxt_embed = CNN_Embedding(weights_matrix, n_filters, filter_sizes, c_dropout, 1)
        input_size = len(filter_sizes)*n_filters

        # self.pretrained_weights = './bert/'
        # self.cxt_embed = BertModel.from_pretrained(self.pretrained_weights)
        # input_size = 768

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=l_dropout, bidirectional=True, batch_first=True)
        bi_output_size = hidden_size * 2

        self.utt_embedding = nn.Embedding(max_utterance_num+1, self.his_hidden_size)
        if utt_weight_matrix is not None:
            self.utt_embedding.load_state_dict({'weight': utt_weight_matrix})
        self.utt_embedding.weight.requires_grad = True

        #self.cur_w = nn.Linear(bi_output_size, hidden_size)
        #self.his_w = nn.Linear(self.his_hidden_size, hidden_size)
        #self.user_att = nn.Linear(hidden_size, 1)
        self.sim_w = nn.Linear(bi_output_size, self.his_hidden_size)

        self.edge_att = EdgeAttention(bi_output_size, max_seq_len=self.max_utt_num)
        self.graph_net = GraphNetwork(bi_output_size, self.n_relations, self.graph_hidden_size)

        bi_output_size += self.graph_hidden_size

        bi_output_size += input_size

        self.h2o = nn.Linear(bi_output_size, output_size)
        self.init_weight()

    def init_weight(self):
        for weights in [self.rnn.weight_hh_l0, self.rnn.weight_ih_l0, self.rnn.weight_ih_l0_reverse, self.rnn.weight_hh_l0_reverse,
                        self.rnn.weight_hh_l1, self.rnn.weight_ih_l1, self.rnn.weight_ih_l1_reverse, self.rnn.weight_hh_l1_reverse]:
            init.orthogonal_(weights)
        #init.xavier_normal_(self.his_w.weight)
        #init.xavier_normal_(self.cur_w.weight)
        #init.xavier_normal_(self.user_att.weight)
        init.xavier_normal_(self.sim_w.weight)
        init.xavier_normal_(self.h2o.weight)

    def user_feature(self, current_usr, cls_att, usr_his, usr_his_mask):
        # current_usr (batch, utt, hidden * 2)
        # cls_att (batch, utt, out_size)
        # usr_his (batch, out_size, max_his_utt)
        # usr_his_mask (batch, out_size, max_his_utt)
        # out (batch, out_size, hidden * 2)
        usr_utt = self.utt_embedding(usr_his) # (batch, out_size, max_his_utt, hidden * 2)
        batch_size, out_size, max_his_utt, his_hidden = usr_utt.size()
        current_usr = current_usr.unsqueeze(1).repeat(1, out_size, 1, 1) # (batch, out_size, utt, hidden * 2)
        current_usr = self.sim_w(current_usr)
        cur_his_sim = torch.matmul(current_usr, usr_utt.transpose(-1, -2)) # (batch, out_size, utt, max_his_utt)
        #cur_his_sim = torch.exp(-torch.pow(euclidean_dist(current_usr, usr_utt), 2))
        cls_att = cls_att.transpose(-1, -2).unsqueeze(3) # (batch, out_size, utt, 1)
        all_att = cls_att * cur_his_sim # (batch, out_size, utt, max_his_utt)
        all_att, _ = torch.max(all_att, 2) # (batch, out_size, max_his_utt)
        all_att = F.softmax(all_att, dim=-1)
        all_att = all_att * usr_his_mask
        all_att = all_att / torch.clamp(all_att.sum(-1, keepdim=True), 1e-10, 1.0)
        all_att = all_att.unsqueeze(2)
        out = torch.matmul(all_att, usr_utt).squeeze(2) # (batch, out_size, hidden * 2)
        return out

    def forward(self, dialogue, user_tree, user_tag, user_history, user_history_mask, targets=None):
        # print(dialogue.size())
        # user_history (batch, out_size, max_his_utt)

        batch_size, timesteps, sent_len = dialogue.size()

        c_out = self.cxt_embed(dialogue) # (batch size * timesteps, n_filters * len(filter_sizes)*k)
        r_in = c_out.view(batch_size, timesteps, -1)

        # c_out = self.cxt_embed(dialogue.view(-1, sent_len))[0][:, 0]
        # r_in = c_out.view(batch_size, timesteps, -1)

        self.rnn.flatten_parameters()
        r_out, _ = self.rnn(r_in) # (batch, timesteps, hidden * 2)

        f_predict_vec = torch.cat((r_out, Variable(torch.zeros(batch_size, timesteps, self.graph_hidden_size)).to(device)), dim=2)
        f_predict_vec = torch.cat((r_in, f_predict_vec), dim=2)

        f_predicts = self.h2o(f_predict_vec)

        node_att = F.softmax(f_predicts, dim=-1)
        cls_att = torch.sigmoid(f_predicts)

        user_node = self.user_feature(r_out, cls_att, user_history, user_history_mask) # (batch, out_size, hidden*2)
        scores = self.edge_att(r_out, user_tree)
        graph_input = torch.cat((r_out, user_node), dim=1)
        #graph_input = r_out
        edge_norm = []
        for b_i in range(batch_size):
            b_edg_norm = []
            for e_i in range(len(user_tree[b_i][0])-timesteps*self.output_size):
                b_edg_norm.append(scores[b_i, user_tree[b_i][0][e_i], user_tree[b_i][1][e_i]])
            b_edg_norm.extend(node_att[b_i].view(-1))
            edge_norm.extend(b_edg_norm)
        edge_norm = torch.tensor(edge_norm).to(device)

        data_list = []
        edge_type = []
        for idx in range(batch_size):
            u_t = torch.from_numpy(np.array(user_tree[idx][:2])).long().to(device)
            edge_type.extend(user_tree[idx][2])
            data_list.append(Data(x=graph_input[idx], edge_index=u_t, y=1))
        data = Batch.from_data_list(data_list)
        nodes = data.x # (batch * timesteps, hidden * 3)
        edge_index = data.edge_index # (2, edge_num * batch)
        edge_type = torch.from_numpy(np.array(edge_type)).long().to(device)

        graph_out = self.graph_net(nodes, edge_index, edge_type, edge_norm)
        graph_out = graph_out.view(batch_size, timesteps+self.output_size, -1) # (batch, timesteps+out_size, hidden)

        predict_vec = torch.cat((r_out, graph_out[:, :timesteps]), dim=2)
        predict_vec = torch.cat((r_in, predict_vec), dim=2)

        predicts = self.h2o(predict_vec)

        return r_out, torch.sigmoid(f_predicts), torch.sigmoid(predicts)

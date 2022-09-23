import torch
from torch import nn
import torch.nn.functional as F
#from node2vec import N2V
import pickle
from torch_geometric.nn import Node2Vec

class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff = 32):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        self.net_dropped = torch.nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net_dropped(x)


class Pigat(nn.Module):

    def __init__(self, feature_dim, embedding_dim, num_heads, output_dimension,n2v_dim,window_size,attention_dim = 64):
        super(Pigat, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.n2v = N2V('node2vec.mdl')
        open_file = open("edge_index.pkl", "rb")
        edge_index = pickle.load(open_file)
        open_file.close()
        self.n2v = Node2Vec(edge_index, embedding_dim=32, walk_length=20,
                              context_size=10, walks_per_node=10,
                              num_negative_samples=1, p=1, q=1, sparse=True)

        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.total_embed_dim = self.feature_dim + sum(self.embedding_dim)+ n2v_dim
        self.output_dimension = output_dimension
        self.num_heads = num_heads
        # embedding layers for 7 categorical features
        # "road_type", "time_stage", "week_day", "lanes", "bridge", "endpoint_u", "endpoint_v", "trip_id"
        # 0 represents Unknown
        # 0-21
        self.embedding_road_type = nn.Embedding(22, self.embedding_dim[0])
        # 0-6
        self.embedding_time_stage = nn.Embedding(7, self.embedding_dim[1])
        # 0-7
        self.embedding_week_day = nn.Embedding(8, self.embedding_dim[2])
        # 0-8
        self.embedding_lanes = nn.Embedding(9, self.embedding_dim[3])
        # 0-1
        self.embedding_bridge = nn.Embedding(2, self.embedding_dim[4])
        # 0-16
        self.embedding_endpoint_u = nn.Embedding(17, self.embedding_dim[5])
        self.embedding_endpoint_v = nn.Embedding(17, self.embedding_dim[6])
        # self.linearq = nn.Linear(self.total_embed_dim, self.attention_dim)
        # self.linearx = nn.Linear(self.total_embed_dim, self.attention_dim)
        self.attention_dim = self.total_embed_dim
        self.selfattn = nn.MultiheadAttention(embed_dim= self.attention_dim, num_heads= self.num_heads)
        self.norm = LayerNorm(self.attention_dim )
        self.feed_forward = PositionwiseFeedForward(self.attention_dim)
        self.linear = nn.Linear(self.attention_dim,self.output_dimension)
        self.activate = nn.Softplus()
        self.middleOfTheWindow = window_size//2

    def forward(self, x, c, id):
        # x -> [ batch, window size, feature dimension]
        # c -> [batch, number of categorical features, window size]

        # [batch_sz, window_sz, embedding dim]
        '''
        embedded_road_type = self.embedding_road_type(c[:,0,:])
        embedded_time_stage = self.embedding_time_stage(c[:, 1, :])
        embedded_week_day = self.embedding_week_day(c[:, 2, :])
        embedded_lanes = self.embedding_lanes(c[:, 3, :])
        embedded_bridge = self.embedding_bridge(c[:, 4, :])
        embedded_endpoint_u = self.embedding_endpoint_u(c[:, 5, :])
        embedded_endpoint_v = self.embedding_endpoint_v(c[:, 6, :])
        '''

        #print(id.view(id.shape[0], id.shape[1]*id.shape[2]).shape)
        # segmentEmbed_0 = self.n2v(id[:, 0]).unsqueeze(1)
        # segmentEmbed_1 = self.n2v(id[:, 1]).unsqueeze(1)
        # segmentEmbed_2 = self.n2v(id[:, 2]).unsqueeze(1)
        # [batch, window size, node2vec]
        segmentEmbed  = torch.cat([self.n2v(id[:, i]).unsqueeze(1) for i in range(id.shape[1])], dim=1)
        #print('segmentEmbed', segmentEmbed.shape)

        # [batch_sz, window_sz, embedding dim
        embedded0 = self.embedding_road_type(c[:,0,:])
        embedded1 = self.embedding_time_stage(c[:, 1, :])
        embedded2 = self.embedding_week_day(c[:, 2, :])
        embedded3 = self.embedding_lanes(c[:, 3, :])
        embedded4 = self.embedding_bridge(c[:, 4, :])
        embedded5 = self.embedding_endpoint_u(c[:, 5, :])
        embedded6 = self.embedding_endpoint_v(c[:, 6, :])

        # embedded = torch.cat([embedded, self.embedding_time_stage(c[:, 1, :])], dim=-1)
        # embedded = torch.cat([embedded, self.embedding_week_day(c[:, 2, :])], dim=-1)
        # embedded = torch.cat([embedded, self.embedding_lanes(c[:, 3, :])], dim=-1)
        # embedded = torch.cat([embedded, self.embedding_bridge(c[:, 4, :])], dim=-1)
        # embedded = torch.cat([embedded, self.embedding_endpoint_u(c[:, 5, :])], dim=-1)
        # embedded_6 = self.embedding_endpoint_v(c[:, 6, :])
        #embedded = torch.cat([embedded0,embedded1,embedded2,embedded3,embedded4,embedded5,embedded6], dim=-1)

        # [ batch, window size, feature dimension+ sum embedding dimension + node2vec]
        x = torch.cat([x, embedded0,embedded1,embedded2,embedded3,embedded4,embedded5,embedded6, segmentEmbed], dim=-1)
        # [ window size, batch,  feature dimension+ sum embedding dimension]
        x = x.transpose(0, 1).contiguous()
        # q -> [1, batch, feature dimension+ sum embedding dimension]
        # middle of the window
        q = x[self.middleOfTheWindow, :, :].unsqueeze(0)
        # q = F.relu(self.linearq(q))
        # x = F.relu(self.linearx(x))
        # x -> [windowsz, batch, feature dimension+ sum embedding dimension]

        x_output, output_weight = self.selfattn(q,x,x)
        # x_output -> [1, batchsz, feature dimension+ sum embedding dimension]
        x_output = self.norm(q+x_output)

        x_output_ff = self.feed_forward(x_output.squeeze(0))
        x_output = self.norm(x_output.squeeze(0) + x_output_ff)
        x_output = self.linear(x_output)
        # offset for velocity profile estimation to avoid 0 velocity
        # return F.relu(x_output)+0.1
        x_output = self.activate(x_output)
        return x_output

def testNet():
    # test nets
    net = Pigat(feature_dim=6, embedding_dim=[4, 2, 2, 2, 2, 4, 4], num_heads=1,
                  output_dimension=60, n2v_dim=32,window_size=3)
    p = sum(map(lambda p: p.numel(), net.parameters()))
    print("number of parameters:", p)
    net.n2v.embedding.weight.requires_grad = False
    print(dict(net.named_parameters()))
    # [batch size, window length,  feature dimension]
    tmp = torch.randn(1, 10,3, 6)
    # [batch, categorical_dim, window size]
    c = torch.randint(1, (1, 10, 7, 3))
    #[batch,  window size]
    id = torch.randint(1, (1, 10, 3))
    print(tmp.shape,c.shape,id.shape)
    # [batch size, output dimension]
    out = net(tmp,c,id)

    print(net)
    print("tmp",tmp)
    print("out", out)
    print("fc out:", out.shape)



if __name__ == "__main__":
    testNet()
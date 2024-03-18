import networkx as nx
import osmnx as ox
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx
import pickle

import os.path as osp
import sys

class N2V:
    '''
    :param
        'dual': segment -> node; intersection -> edge
    :return:
    '''

    def readGraph(self, fileName):
        graph = ox.io.load_graphml(fileName)
        # convert the graph to the dual graph
        lineGraph = nx.line_graph(graph)
        # transfer the multiDilineGraph to 2 dimension
        eNew = [(x[0],x[1]) for x in lineGraph.edges]
        graph = nx.Graph()
        graph.update(edges=eNew, nodes=lineGraph.nodes)
        return graph
    
    def __init__(self, graphFile, ckpt_path, newModel = False):
        self.dualGraph = self.readGraph(graphFile)
        file_name = "../results/dualGraphNodes.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(list(self.dualGraph.nodes), open_file)
        open_file.close()
        #print(list(self.dualGraph.nodes))
        self.data = from_networkx(self.dualGraph)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device: {}'.format(self.device))
        open_file = open("../results/edge_index.pkl", "wb")
        pickle.dump(self.data.edge_index, open_file)
        open_file.close()
        open_file = open("../results/edge_index.pkl", "rb")
        edge_index = pickle.load(open_file)
        open_file.close()
        self.model = Node2Vec(edge_index, embedding_dim=32, walk_length=20,
                         context_size=10, walks_per_node=10,
                         num_negative_samples=1, p=1, q=1, sparse=True).to(self.device)

        self.loader = self.model.loader(batch_size=128, shuffle=True, num_workers=0)
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)
        if newModel:
            for epoch in range(1, 301):
                loss = self.train()
#                 acc = self.test()
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            self.saveTo(ckpt_path)
        else:
            self.loadFrom(ckpt_path)

    def train(self):
        self.model.train()
        total_loss = 0
        for pos_rw, neg_rw in self.loader:
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.loader)
    
#     @torch.no_grad()
#     def test(self):
#         self.model.eval()
#         z = self.model()
#         acc = self.model.test(
#             train_z=z[data.train_mask],
#             train_y=data.y[data.train_mask],
#             test_z=z[data.test_mask],
#             test_y=data.y[data.test_mask],
#             max_iter=150,
#         )
#         return acc

    def saveTo(self,ckpt_path):
        torch.save(self.model.state_dict(), ckpt_path)
        return

    def loadFrom(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        return

    def embed(self, segment):
        id = list(self.dualGraph.nodes).index((segment[0],segment[1],0))
        print(id)
        id_tensor = torch.tensor([id]) if isinstance(id, int) else torch.tensor(id)
        return self.model(id_tensor.to(self.device))
    
    
    

if __name__ == '__main__':
    graphFile = '../data/maps/minneapolis.graphml'
    modelFile = '../pretrainedModels/node2vec.mdl'
    n2v = N2V(graphFile, modelFile, newModel=True)
    print(n2v.embed((187849834, 598269571)))











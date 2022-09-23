import networkx as nx
import osmnx as ox
import geopandas as gpd
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx
import pickle


def readGraph():
    '''
    :param
        'dual': segment -> node; intersection -> edge
    :return:
    '''
    percentile = '005'
    filefold = r'D:/cygwin64/home/26075/workspace/'
    network_gdf = gpd.read_file(filefold+'network_'+percentile+'/edges.shp')
    nodes_gdf = gpd.read_file(filefold+'network_'+percentile+'/nodes.shp')
    graph = ox.utils_graph.graph_from_gdfs(nodes_gdf, network_gdf)
    # convert the graph to the dual graph
    lineGraph = nx.line_graph(graph)
    # transfer the multiDilineGraph to 2 dimension
    eNew = [(x[0],x[1]) for x in lineGraph.edges]
    graph = nx.Graph()
    graph.update(edges=eNew, nodes=lineGraph.nodes)
    return graph


class N2V:
    def __init__(self, ckpt_path, newModel = False):
        self.dualGraph = readGraph()
        file_name = "dualGraphNodes.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(list(self.dualGraph.nodes), open_file)
        open_file.close()
        #print(list(self.dualGraph.nodes))
        self.data = from_networkx(self.dualGraph)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        open_file = open("edge_index.pkl", "wb")
        pickle.dump(self.data.edge_index, open_file)
        open_file.close()
        open_file = open("edge_index.pkl", "rb")
        edge_index = pickle.load(open_file)
        open_file.close()
        self.model = Node2Vec(edge_index, embedding_dim=32, walk_length=20,
                         context_size=10, walks_per_node=10,
                         num_negative_samples=1, p=1, q=1, sparse=True).to(self.device)

        self.loader = self.model.loader(batch_size=128, shuffle=True, num_workers=0)
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)
        if newModel:
            self.train(epochs=20)
            self.saveTo(ckpt_path)
        else:
            self.loadFrom(ckpt_path)

    def train(self, epochs):
        for epoch in range(1, epochs):

            loss = self.trainForOnePace()
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        return

    def trainForOnePace(self):
        self.model.train()
        total_loss = 0
        for pos_rw, neg_rw in self.loader:
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.loader)

    def saveTo(self,ckpt_path):
        torch.save(self.model.state_dict(), ckpt_path)
        return

    def loadFrom(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        return

    def embed(self, segment):
        id = list(self.dualGraph.nodes).index((segment[0],segment[1],0))
        print(id)
        return self.model(id)


if __name__ == '__main__':
    n2v = N2V('node2vec.mdl',newModel=False)
    print(n2v.embed((187849834, 598269571)))








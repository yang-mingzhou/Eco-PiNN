import torch
import os
import csv
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import geopandas as gpd
import torch
import pickle
from torch.utils.data.dataset import ConcatDataset
import math
import torch
from torch.utils.data.sampler import RandomSampler

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

class ObdData(Dataset):

    def __init__(self, root="data_normalized", mode="train", fuel = False, percentage=20, window_size=5, path_length=10\
                 , label_dimension=1, pace=5, withoutElevation = False):
        """

        :param root: root of the data(normalized)
        :param mode: "train" / "val" / "test"
        :param percentage: the percentage of data used for validation/test
            "20" -> 60/20/20 train/val/test; "10" -> 80/10/10 train/val/test
        :param window_size: length of window for attention
        :param path_length: length of path
        :param label_dimension: if == 2 -> label_list = [energy, time]; if == 1 -> label_list = [energy]
        """
        super(ObdData, self).__init__()
        self.percentage = str(percentage)
        self.root = os.path.join(root, self.percentage)
        self.mode = mode
        self.fuel = fuel
        self.windowsz = window_size
        self.path_length = path_length
        self.label_dimension = label_dimension
        self.pace = pace
        self.len_of_index = 0
        self.withoutElevation = withoutElevation
        #self.dualGraphNode = list(readGraph().nodes)
        file_name = "dualGraphNodes.pkl"
        open_file = open(file_name, "rb")
        self.dualGraphNode = pickle.load(open_file)
        open_file.close()
        #print(self.dualGraphNode)


        if self.withoutElevation == True:
            self.numerical_dimension = 5
        else:
            self.numerical_dimension = 6
        #self.data_list_w, self.label_list_w = self.load_csv(self.mode + "_data.csv")
        if fuel == True:
            self.data = self.load_csv(self.mode + "_data_fuel.csv")
        else:
            self.data = self.load_csv(self.mode + "_data.csv")
        #print(self.root)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            # print('Using GPU..')
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        #print(self.data[0])
        self.data_x = torch.Tensor([x[0] for x in self.data]).to(device)
        #print(self.data_x[0,...].shape)
        self.data_y = torch.Tensor([x[1] for x in self.data]).to(device)
        self.data_c = torch.LongTensor([x[5:12] for x in self.data]).to(device)
        self.id = torch.LongTensor([x[-1] for x in self.data]).to(device)
        print('number of paths(mode={m},fuel={f})'.format(m=self.mode,f=self.fuel), self.__len__())


    def load_csv(self, filename):
        """
        :param filename: -> str: filename to be read
        :return: data_list -> list(list(float)) -> list of data in the window
                 label_list -> list(list(float)) -> list of labels in the window
        """
        # [index, 13 dimensions, path length, windows size, feature dimension]
        data_list_path= []
        # {trip-id: [13 dimensions, windowsz, feature]}
        data_dict_trip_id = dict()
        data_list = []
        #print(os.path.join(self.root, filename))
        if not os.path.exists(os.path.join(self.root, filename)):
            print("Warning: Wrong File Directory")
        else:
            with open(os.path.join(self.root, filename)) as f:
                reader = csv.reader(f)

                for row in reader:
                    data_row = row
                    # [data, label, segment_id, length, position, road_type, time_stage,
                    # week_day, lanes, bridge, endpoint_u, endpoint_v, trip_id]
                    if int(data_row[3]) >= self.path_length:

                        # length -> the number of segments in the trip
                        # 6 numerical attributes
                        # speed_limit,mass,elevation_change,previous_orientation,length,direction_angle
                        if self.withoutElevation == True:
                            data_row[0] = list(map(float, data_row[0][1:-1].split(", ")))[:2]+list(map(float, data_row[0][1:-1].split(", ")))[3:]
                        else:
                            data_row[0] = list(map(float, data_row[0][1:-1].split(", ")))
                            #data_row[0][1] = 0
                        # label
                        if self.label_dimension == 2:
                            data_row[1] = list(map(float, data_row[1][1:-1].split(", ")))
                        elif self.label_dimension == 1:
                            # [0] for fuel *100; [1] for time
                            if self.fuel:
                                data_row[1] = list(map(float, data_row[1][1:-1].split(", ")))[0]*100
                            else:
                                data_row[1] = list(map(float, data_row[1][1:-1].split(", ")))[1]

                        data_row[2:] = map(int, map(float, data_row[2:]))
                        #data_row[-1] = map(int, data_row[-1].split(", "))
                        #data_row[-1] =self.dualGraphNode.index((data_row[-1][0], data_row[-1][1], 0))
                        #print('data_row',data_row)
                        data_list.append(data_row)
            if not len(data_list):
                print("No qualified data")
                return None
            for i in range(len(data_list)):
                position = data_list[i][4]
                length = data_list[i][3]
                trip_id = data_list[i][-2]
                # construct a feature matrix for each window (windowsz * feature_dimension),
                # each row of the matrix is a feature(or label) of a segment in the window
                left = position - self.windowsz // 2  # [left, right) in the trip
                right = position + self.windowsz // 2 + 1
                left_idx = i - self.windowsz // 2  # [left_idx, right_idx) in the data_list
                right_idx = i + self.windowsz // 2 + 1
                if self.label_dimension ==1:
                    data_zero_line = [[0]*self.numerical_dimension]+[0]*13
                else:
                    data_zero_line = [[0] * self.numerical_dimension] + [[0,0]] + [0] * 12
                if left > 0 and right <= length:
                    #print(left,right,left_idx,right_idx)
                    data_sub = data_list[left_idx:right_idx]
                elif left <= 0 and right <= length:
                    #print(left, right, left_idx, right_idx)
                    data_sub = [data_zero_line]*(1 - left)+data_list[left_idx+1 - left:right_idx]
                elif left > 0 and right > length+ 1:
                    # print(left, right, left_idx, right_idx)
                    data_sub = data_list[left_idx:(i+(length-position)+1)] + [data_zero_line] * (right - length-1)
                else:
                    data_sub = [data_zero_line] * self.windowsz
                # [dimension, windowsz, feature]
                data_w = [[x] for x in data_sub[0]]
                for j in range(1, len(data_sub)):
                    data_j = [[x] for x in data_sub[j]]
                    data_w = [x + y for x, y in zip(data_w, data_j)]
                data_dict_trip_id[trip_id] = data_dict_trip_id.get(trip_id,[])+[data_w]
            # print(len(data_dict_trip_id))
            for i in sorted(data_dict_trip_id.keys()):
                data_trip = data_dict_trip_id[i]
                for j in range(0,len(data_trip)-self.path_length+1,self.pace):
                    data_trip_j = [[x] for x in data_trip[j]]
                    for k in range(1, self.path_length):
                        data_trip_k = [[x] for x in data_trip[j+k]]
                        data_trip_j = [x + y for x, y in zip(data_trip_j, data_trip_k)]
                    data_list_path.append(data_trip_j)
        return data_list_path

    def __len__(self):
        """

        :return: the length of the db
        """
        return len(self.data)

    def __getitem__(self, idx):

        return self.data_x[idx,...],self.data_y[idx,...],self.data_c[idx,...],self.id[idx,...]




def testDataloader():
    batch_size = 8
    # test dataloader
    db_time = ObdData("model_data_newOct", "train", fuel=False, percentage=10, label_dimension=1,withoutElevation=False)
    db_fuel = ObdData("model_data_newOct", "train", fuel=True, percentage=10, label_dimension=1, withoutElevation=False)
    sampler = torch.utils.data.RandomSampler(db_fuel, replacement=True, num_samples=len(db_time),
                                             generator=None)
    #concat_dataset = ConcatDataset([db_time, db_fuel])



    # print(x,y,d)
    loader_time = DataLoader(db_time, batch_size= batch_size, shuffle= False, num_workers=0)
    loader_fuel = DataLoader(db_fuel, batch_size= batch_size, sampler = sampler, shuffle=False, num_workers=0)
    for (x,y,c,id),(x_f,y_f,c_f,id_f) in zip(loader_time,loader_fuel):
        # x: numerical features [batch, path length, window size, feature dimension]
        # y: label [batch, path length, window size, (label dimension)]
        # c: categorical features [batch, number of categorical features, path length, window size]
        # id: node2vec [batch, path length, window size]
        print(x.shape, y.shape,c.shape,id.shape)
        print(x_f.shape, y_f.shape,c_f.shape,id_f.shape)
        print(id)
        print(id_f)
        print(y)
        #t = torch.tensor([1, 0.01]).unsqueeze(0).to("cuda")
        #print(t.shape)
        label = y[:, 0, 5 // 2]
        print(label.shape)
        print(label)
        labelfuel = y_f[:, 0, 5 // 2]
        print(labelfuel.shape)
        print(labelfuel)
        #print(label * t)
        # c[:,0,:,:] the first categorical features
        # "road_type", "time_stage", "week_day", "lanes", "bridge", "endpoint_u", "endpoint_v"
        print(c[:,:,0,:])
        print(c[:, :, 0, :].shape)
        # [batch, path length, window size, feature dimension]
        # data shape: torch.Size([32, 10, 5, 8]) label shape: torch.Size([32, 10, 5, 2])
        print("data shape:", x.shape, "label shape:", y.shape)
        # [batch, window size, feature dimension]
        x_one_path = x[:,0,:,:]
        # [window size, batch ,feature dimension]
        x_one_path = x_one_path.transpose(0,1).contiguous()
        print(x_one_path.shape)
        # print(x_one_path)
        break
    
    




if __name__ == "__main__":
    testDataloader()

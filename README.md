# [Eco-PiNN: A Physics-informed Neural Network for Eco-toll](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch94)

## Abstract:

The eco-toll estimation problem quantifies the expected environmental cost (e.g., energy consumption, exhaust emissions) for a vehicle to travel along a path. This problem
is important for societal applications such as eco-routing, which aims to find paths with the lowest exhaust emission or energy need. The challenges of this problem are threefold: 
(1) the dependence of a vehicle’s eco-toll on its physical parameters; (2) the lack of access to data with eco-toll
information; and (3) the influence of contextual information (i.e. the connections of adjacent segments in the path) on the eco-toll of road segments. 
Prior work on eco-toll estimation has mostly relied on pure data-driven approaches and has high estimation errors given the limited training data. 
To address these limitations, we propose a novel Eco-toll estimation Physics-informed Neural Network framework (Eco-PiNN) using three novel ideas, namely, (1) a physics-informed decoder that integrates the physical laws of the vehicle engine into the network, (2) an attention-based contextual information encoder, and (3) a physics-informed regularization to reduce overfitting. Experiments on real-world
heavy-duty truck data show that the proposed method can greatly improve the accuracy of eco-toll estimation compared with state-of-the-art methods

## [Requirements](https://github.com/yang-mingzhou/Eco-PiNN/blob/main/requirements.txt):
```
bintrees==2.2.0
geopandas==0.10.2
networkx==2.6.3
numpy==1.21.3
osmnx==0.16.1
pandas==1.2.4
plotly==5.3.1
psycopg2==2.9.1
Shapely==1.8.0
torch==1.11.0
torch_geometric==2.0.4
torchvision==0.12.0
tqdm==4.59.0
```
## Files:
1. [config.py](https://github.com/yang-mingzhou/Eco-PiNN/blob/main/config.py): the setting of hyper-parameters in Eco-PiNN.
2. [training_testing_EcoPiNN.py](https://github.com/yang-mingzhou/Eco-PiNN/blob/main/training_testing_EcoPiNN.py): training and testing of EcoPiNN.


## File Folders:

1. Folder [utils](https://github.com/yang-mingzhou/Eco-PiNN/tree/main/utils):
   1. Definition of [contextual information encoder](https://github.com/yang-mingzhou/Eco-PiNN/blob/main/utils/ciEncoder.py);
   2. Definition of [physics informed decoder](https://github.com/yang-mingzhou/Eco-PiNN/blob/main/utils/piDecoder.py);
   3. The [aggregation functions](https://github.com/yang-mingzhou/Eco-PiNN/blob/main/utils/funcs.py) that aggregates the encoder&decoder together to generate the eco-toll estimation.
   4. A [dataloader](https://github.com/yang-mingzhou/Eco-PiNN/blob/main/utils/obdDataLoader.py) that generates subpath representations of eco-toll queries (Sec 3.1 of the paper). 
2. Folder [pretrained model](https://github.com/yang-mingzhou/Eco-PiNN/tree/main/pretained%20model) contains the pretrained NODE2VEC model(detailed in Sec 3.1 of the paper).
   
   
## Datasets
We are not allowed to publish the on-board diagnostics (OBD) dataset described in the paper. 
In future works, we plan to generate some synthetic vehicle travel datasets using vehicle powertrain simulators 
(e.g., [FastSim](https://www.nrel.gov/transportation/fastsim.html#:~:text=The%20Future%20Automotive%20Systems%20Technology,%2C%20cost%2C%20and%20battery%20life.)).
We will publish the code for generating the synthetic datasets and the corresponding datasets in the future.

Cite
-----
```
@inproceedings{li2023eco,
  title={Eco-pinn: A physics-informed neural network for eco-toll estimation},
  author={Li, Yan and Yang, Mingzhou and Eagon, Matthew and Farhadloo, Majid and Xie, Yiqun and Northrop, William F and Shekhar, Shashi},
  booktitle={Proceedings of the 2023 SIAM International Conference on Data Mining (SDM)},
  pages={838--846},
  year={2023},
  organization={SIAM}
}
```
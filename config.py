# storing network parameters.
import os

class Params:
    def __init__(self):
        self.batchsz = 512  # Batch size.
        self.max_epochs = 1500 # Number of epochs to train for.
        self.lr = 1e-4 # Learning rate.
        self.lengthOfVelocityProfile = 60  # divide a segment equally into n parts according to the length
        self.meanOfSegmentLength= 608.2156661
        self.stdOfSegmentLength= 900.4150229
        self.meanOfSegmentHeightChange= -0.063065664
        self.stdOfSegmentHeightChange= 8.62278608
        self.meanOfSpeedLimit= 80.73990991
        self.stdOfSpeedLimit= 21.5979505
        self.meanOfMass= 23204.9788
        self.stdOfMass= 8224.139199
        self.omega_jerk= 1e-6  # weight of jerk loss (MSE)
        self.omega_fuel= 0.6
        self.omega_time= 0.4  # omega_time + omega_fuel = 1
        self.feature_dimension= 6 # dimension of the input numerical features:
        # [speed limit, mass, elevation change, previous orientation, length, direction angle]
        # there are also 7 categorical features:
        # "road_type", "time_stage", "week_day", "lanes", "bridge", "endpoint_u", "endpoint_v"
        self.head_number= 1
        self.train_path_length= 20 #length of path used for training/validation
        self.window_sz= 3
        self.pace_train= 5 if 5 < self.train_path_length else self.train_path_length  # pace between paths
        self.pace_test= 5
        self.n2v_dim= 32
        self.beta_1= 0.9
        self.beta_2= 0.98
        self.eps= 1e-9
        self.patienceOfTrainingEpochs= 10  # Number of epochs with no improvement after which training stage will be ended
        self.ckpt_path = os.path.join(os.getcwd(), r"multitaskModels/pinnMultihead.mdl")
        self.data_root = "ExpDataset/recursion11"
        self.output_root = "prediction_result.csv"
        self.pathLossWeight = 1


params = Params()
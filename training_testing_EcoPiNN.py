import torch
from torch import nn, optim
import os
from torch.utils.data import DataLoader
from obddata import ObdData
from utils.ciEncoder import CiEncoder
import numpy as np
import csv
from tqdm import tqdm
import torch.profiler
import torch.utils.data
import config
from utils.funcs import calLossOfPath


use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using GPU..')
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def eval(model, loader_time, loader_fuel, output = False):
    """
    Evaluate the model accuracy by MAPE of the energy consumption & average MSE of energy and time
    :param model: trained model
    :param loader: data loader
    :param output: output the estimation result or not
    :return: MAPE of energy, average MSE of energy and time
    """

    if output:
        csvFile = open(config.params.output_root, "w")
        writer = csv.writer(csvFile)
        writer.writerow(["id", "ground truth fuel(l)", "ground truth time (s)", "estimated fuel (l)", "estimated time (s)"])
    loss_mape_fuel_total = 0
    loss_mape_time_total = 0
    loss_total = 0
    loss_fuel_total = 0
    loss_time_total = 0
    mse_time_total = 0
    mae_time_total = 0
    mse_fuel_total = 0
    mae_fuel_total = 0
    cnt = 0
    model.eval()
    for (xt, yt, ct, idt),(xf,yf,cf,idf) in zip(loader_time,loader_fuel):
        #x, y, c = x.to(device), y.to(device), c.to(device)
        with torch.no_grad():
            # loss_time is not the true mse
            loss_time, loss_mape_time, mse_time, mae_time,  cnt_add = calLossOfPath(model, xt, yt, ct, idt, mode ='time', output=output)
            loss_fuel, loss_mape_fuel, mse_fuel, mae_fuel,  cnt_add = calLossOfPath(model, xf, yf, cf, idf, mode='fuel', output=output)
            loss_fuel_total += loss_fuel * cnt_add
            loss_time_total += loss_time * cnt_add
            loss_total += config.params.omega_fuel * loss_fuel * cnt_add + config.params.omega_time * loss_time * cnt_add
            loss_mape_time_total += loss_mape_time * cnt_add
            loss_mape_fuel_total += loss_mape_fuel * cnt_add
            mse_time_total += mse_time * cnt_add
            mae_time_total += mae_time * cnt_add
            mse_fuel_total += mse_fuel * cnt_add
            mae_fuel_total += mae_fuel * cnt_add
            cnt += cnt_add
    return loss_total/cnt, loss_fuel_total/cnt, loss_time_total/cnt, loss_mape_fuel_total/cnt, loss_mape_time_total/cnt , mse_fuel_total/cnt, mse_time_total/cnt, mae_fuel_total/cnt, mae_time_total/cnt


def train():
    # random seed
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    train_db_fuel = ObdData(root=config.params.data_root, mode="train", fuel=True, percentage=20, window_size=config.params.window_sz,\
                       path_length=config.params.train_path_length, label_dimension=1, pace=config.params.pace_train,
                       withoutElevation=False)
    train_db_time = ObdData(root=config.params.data_root, mode="train", fuel=False, percentage=20, window_size=config.params.window_sz,\
                       path_length=config.params.train_path_length, label_dimension=1, pace=config.params.pace_train,
                       withoutElevation=False)
    train_sampler_fuel = torch.utils.data.RandomSampler(train_db_fuel, replacement=True, num_samples=len(train_db_time),
                                                   generator=None)
    train_loader_fuel = DataLoader(train_db_fuel, sampler = train_sampler_fuel, batch_size=config.params.batchsz, num_workers=0)
    train_loader_time = DataLoader(train_db_time, batch_size=config.params.batchsz, num_workers=0)


    val_db_fuel = ObdData(root=config.params.data_root, mode="val", fuel=True, percentage=20, window_size=config.params.window_sz,\
                     path_length=config.params.train_path_length, label_dimension=1, pace=config.params.pace_test,
                     withoutElevation=False)
    val_db_time = ObdData(root=config.params.data_root, mode="val", fuel=False, percentage=20, window_size=config.params.window_sz, \
                          path_length=config.params.train_path_length, label_dimension=1, pace=config.params.pace_test,
                          withoutElevation=False)
    val_sampler_fuel = torch.utils.data.RandomSampler(val_db_fuel, replacement=True, num_samples=len(val_db_time),
                                             generator=None)

    val_loader_fuel = DataLoader(val_db_fuel, sampler = val_sampler_fuel, batch_size=config.params.batchsz, num_workers=0)
    val_loader_time = DataLoader(val_db_time, batch_size=config.params.batchsz, num_workers=0)

    # Create a new model or load an existing one.
    model = CiEncoder(feature_dim=config.params.feature_dimension,embedding_dim=[4,2,2,2,2,4,4],num_heads=config.params.head_number,
                  output_dimension=config.params.lengthOfVelocityProfile, n2v_dim=config.params.n2v_dim,window_size=config.params.window_sz)
    print('Creating new model parameters..')
    # It initialises the parameters with a
    # range of values that stops the signal fading or getting too big.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model.to(device)
    # load the pre-trained model
    model.n2v.load_state_dict(torch.load('node2vec.mdl'))
    model.n2v.embedding.weight.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.params.lr,
                           betas=(config.params.beta_1, config.params.beta_2), eps=config.params.eps)
    train_loss = []
    global_step = 0
    best_mape, best_mse, best_epoch = torch.tensor(float("inf")), torch.tensor(float("inf")), 0
    for epoch in range(config.params.max_epochs):

        model.train()
        #prof.start()
        for step, ((xt, yt, ct, idt),(xf,yf,cf,idf)) in tqdm(enumerate(zip(train_loader_time,train_loader_fuel))):
            # x: numerical features [batch, path length, window size, feature dimension]
            # y: label [batch, path length, window size, (label dimension)]
            # c: categorical features [batch, number of categorical features, path length, window size]
            # For each batch, predict fuel consumption/time for each segment in a path and sum them
            # [batch size, output dimension]
            mse_time, loss_mape_time, _,_,_ = calLossOfPath(model, xt, yt, ct, idt, mode='time')
            mse_fuel, loss_mape_fuel, _,_,_ = calLossOfPath(model, xf, yf, cf, idf, mode='fuel')
            loss = config.params.omega_fuel*mse_fuel + config.params.omega_time*mse_time
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())

        if epoch % 1 == 0:
            val_mse, val_loss_fuel,val_loss_time, val_fuel_mape, val_time_mape, mse_fuel, mse_time, mae_fuel, mae_time = eval(model,val_loader_time, val_loader_fuel)
            # schedule.step(val_mse)
            if val_mse < best_mse:
                best_epoch = epoch
                best_mse = val_mse
                torch.save(model.state_dict(), config.params.ckpt_path)
            if best_epoch < epoch - config.params.patienceOfTrainingEpochs:
                break

        global_step += 1
    return len(train_loss)


def test(model, test_path_length, test_pace, output = False):
    """

    :param output: "True" -> output the estimation results to output_root
    :return:
    """

    # load an existing model.
    test_db_time = ObdData(root=config.params.data_root, mode="test",fuel=False, percentage=20, window_size=config.params.window_sz,\
                      path_length=test_path_length, label_dimension=1, pace=test_pace,
                      withoutElevation=False)
    test_db_fuel = ObdData(root=config.params.data_root, mode="test",fuel=True, percentage=20, window_size=config.params.window_sz, \
                        path_length=test_path_length, label_dimension=1, pace=test_pace,
                      withoutElevation=False)

    test_loader_time = DataLoader(test_db_time, batch_size=config.params.batchsz, num_workers=0)
    test_loader_fuel = DataLoader(test_db_fuel, batch_size=config.params.batchsz, num_workers=0)
    print(model)
    test_loss,_,_, test_fuel_mape, test_time_mape, mse_fuel, mse_time, mae_fuel, mae_time = eval(model, test_loader_time, test_loader_fuel, output=output)
    return np.array(test_fuel_mape.cpu()) * 100


def trainTest(mode, output = False):
    """
    :param mode: "train" for training, "test" for testing the existing model or predicting
    :return:
    """

    if mode == "train":
        train()
        return
    elif mode == "test":
        model = CiEncoder(feature_dim=config.params.feature_dimension, embedding_dim=[4, 2, 2, 2, 2, 4, 4], num_heads=config.params.head_number,
                             output_dimension=config.params.lengthOfVelocityProfile, n2v_dim=32, window_size=config.params.window_sz)
        if os.path.exists(config.params.ckpt_path):
            print('Reloading model parameters..')
            model.load_state_dict(torch.load(config.params.ckpt_path, map_location=device))
        else:
            print('Error: no existing model')
        model.to(device)
        fuelMapeList = []
        # length of path used for test
        test_path_length_list = [1,2,5,10,20,50,100,200]
        for length in test_path_length_list:
            pace_test = config.params.pace_test
            if  pace_test > length:
                pace_test = length
                print("pace test has been changed to:", pace_test)
            #print("test path length:",length)
            test_fuel_mape = test(model, length,pace_test,output = output)
            fuelMapeList.append(test_fuel_mape)
        return fuelMapeList
    else:
        return



if __name__ == '__main__':
    # mode= "train" / "test"
    trainTest(input("mode="), output = False)




import torch
from torch import nn, optim
import numpy as np
import csv
import torch.profiler
import torch.utils.data
import config
from utils.piDecoder import timeEstimation, fuelEstimation, vt2t, vt2a




def denormalize(normalized, mean, std):
    return normalized*std + mean

def mape_loss(label, pred):
    """
    Calculate Mean Absolute Percentage Error
    labels with 0 value are masked
    :param pred: [batchsz]
    :param label: [batchsz]
    :return: MAPE
    """
    mask = label >= 1e-4

    mape = torch.mean(torch.abs((pred[mask] - label[mask]) / label[mask]))
    return mape

def calLossOfPath(model, x, y, c, id, mode='time', output=False):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Using GPU..')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    segCriterion = nn.HuberLoss()
    pathCriterion = nn.MSELoss()
    maeCriterion = nn.L1Loss()
    jerkCriterion = nn.MSELoss()
    label_segment = torch.zeros(y.shape[0]).to(device)
    pred_segment = torch.zeros(y.shape[0]).to(device)
    if output:
        csvFile = open(config.params.output_root, "a")
        writer = csv.writer(csvFile)
        writer.writerow(
            ["id", "ground truth fuel(l)", "estimated fuel (l)", "ground truth time (s)", "estimated time (s)"])
    for i in range(x.shape[1]):
        # [batch size, window length, feature dimension]
        x_segment = x[:, i, :, :]
        # [batch, categorical_dim, window size]
        c_segment = c[:, :, i, :]
        # [batch, window size]
        id_segment = id[:, i, :]

        # [batch size, output dimension]
        label = y[:, i, config.params.window_sz // 2]

        # [batch size, output dimension]
        label_segment += label

        # [batch size, lengthOfVelocityProfile]
        # offset to make sure the average velocity is higher than 0
        velocityProfile = model(x_segment, c_segment, id_segment)

        # extract the length of this segment
        # [batch size]
        length = denormalize(x_segment[:, config.params.window_sz // 2, 4], config.params.meanOfSegmentLength,
                             config.params.stdOfSegmentLength)
        height = denormalize(x_segment[:, config.params.window_sz // 2, 2], config.params.meanOfSegmentHeightChange,
                             config.params.stdOfSegmentHeightChange)
        speedLimit = denormalize(x_segment[:, config.params.window_sz // 2, 0], config.params.meanOfSpeedLimit,
                                 config.params.stdOfSpeedLimit) / 3.6

        m = denormalize(x_segment[:, config.params.window_sz // 2, 1], config.params.meanOfMass,
                        config.params.stdOfMass).unsqueeze(-1)
        v, t = vt2t(velocityProfile, length)
        acc = vt2a(v, t)
        jerk = vt2a(acc, t)
        zeros = torch.zeros(acc.shape, requires_grad=False).to(device)

        if mode == 'time':
            pred = timeEstimation(t)
            if output:
                for j in range(y.shape[0]):
                    writer.writerow(
                        [id_segment[j, config.params.window_sz // 2].item(), "-", "-", np.array(label[j].cpu()),
                         np.array(pred[j].cpu())])
        else:
            pred = fuelEstimation(v, t, acc, m, height, length)
            if output:
                for j in range(y.shape[0]):
                    writer.writerow(
                        [id_segment[j, config.params.window_sz // 2].item(), np.array(label[j].cpu()),
                         np.array(pred[j].cpu()), "-", "-"])
        # [batch size, output dimension]

        if i == 0:
            seg_loss = segCriterion(label, pred)
            jerk_loss = jerkCriterion(jerk, zeros).to(device)
        else:
            seg_loss += segCriterion(label, pred)
            jerk_loss += jerkCriterion(jerk, zeros).to(device)
        pred_segment += pred

    if output:
        csvFile.close()

    # path level
    mape = mape_loss(label_segment, pred_segment)
    mae = maeCriterion(label_segment, pred_segment)
    mse = pathCriterion(label_segment, pred_segment)

    coefficient = config.params.omega_time if mode == 'time' else config.params.omega_fuel
    if coefficient != 0:
        totalLoss = config.params.pathLossWeight * mape + (
                    seg_loss + config.params.omega_jerk * config.params.lengthOfVelocityProfile * jerk_loss / coefficient) / \
                    x.shape[1]
    else:
        totalLoss = config.params.pathLossWeight * mape + (
                    seg_loss + config.params.omega_jerk * config.params.lengthOfVelocityProfile * jerk_loss) / x.shape[
                        1]
    return totalLoss, mape, mse, mae, y.shape[0]
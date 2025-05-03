import glob
import json
import os
import pickle
import random
import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib as mpl

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from data import mstar_data


def train_one_epoch(model, optimizer, data_loader, device, epoch, **kwargs):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)       # 累计损失
    accu_num = torch.zeros(1).to(device)        # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    feat_loader = []
    label_loader = []
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data  # (b, c, h, w) (16, 2, 128, 128)
        sample_num += images.shape[0]
        feature, pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device).argmax(1)).sum()
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.5f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()

        feat_loader.append(feature)
        label_loader.append(labels)
    # feat = torch.cat(feat_loader, 0)
    # labs = torch.cat(label_loader, 0)
    # if epoch % 10 == 0:
    #     visualize(feat.data.cpu().numpy(), labs.data.cpu().numpy(), epoch, kwargs['feature_dst_path'])
    #
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, input_view, feature_dst_path):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    feat_loader = []
    label_loader = []
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        feature, pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device).argmax(1)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[test epoch {}] loss: {:.3f}, acc: {:.5f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    #     feat_loader.append(feature)
    #     label_loader.append(labels)
    # feat = torch.cat(feat_loader, 0)
    # labs = torch.cat(label_loader, 0)
    # if epoch % 10 == 0:
    #     visualize(feat.data.cpu().numpy(), labs.data.cpu().numpy(), epoch, feature_dst_path)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def visualize(feat, labels, epoch, dst_path):
    # # plt.ion()
    # color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
    #          '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    # # plt.clf()
    # for i in range(10):
    #     plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=color[i])
    #     # 将特征点分到10个类里面，并画在坐标轴上
    #
    # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    # # plt.xlim(xmin=-5, xmax=5)
    # # plt.ylim(ymin=-5, ymax=5)
    # plt.title("epoch=%d" % epoch)
    # sor_path = 'G:/Xing_zhang/Transformer/Multiview_Transformer_Mstar/result/pic'
    # save_path = sor_path + dst_path
    # if os.path.exists(save_path) is False:
    #     os.makedirs(save_path)
    # save_path = os.path.join(save_path, f'epoch={epoch}.jpg')
    # plt.savefig(save_path)
    # # plt.draw()
    # # plt.pause(0.001)

    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(feat)
    tsne.fit_transform(feat)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    fig = plt.figure(figsize=(5, 4))
    cmp = mpl.colors.ListedColormap([plt.cm.tab10(i) for i in range(0, 10)])
    # cmp = copy.copy(plt.cm.tab10)'
    norm = mpl.colors.Normalize(vmin=0, vmax=10)
    # norm = mpl.colors.BoundaryNorm([0,1,2,3,4,5,6],cmp.N)
    # im1 = mpl.cm.ScalarMappable(norm=norm, cmap=cmp)
    im1 = mpl.cm.ScalarMappable(norm=norm, cmap=cmp)
    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], c=labels[i], cmap=cmp, norm=norm, s=0.6)
    cb = fig.colorbar(
        im1, orientation='vertical',
        ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
        label=' '
    )
    namelist = ['2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU234']
    cb.ax.set_yticklabels(namelist)
    sor_path = 'G:/Xing_zhang/Transformer/Multiview_Transformer_Mstar/result/pic'
    save_path = sor_path + dst_path
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    save_path = os.path.join(save_path, f'epoch={epoch}.jpg')
    plt.savefig(save_path)
    # plt.draw()
    # plt.pause(0.001)
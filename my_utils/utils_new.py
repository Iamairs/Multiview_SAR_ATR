import glob
import json
import os
import pickle
import random
import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib as mpl
import os.path as osp
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from data import mstar_data


class AverageMeter(object):
    """Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_features(features, labels, num_classes, epoch, save_dir):
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances).
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    dirname = osp.join(save_dir)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


def train_one_epoch(model, optimizer, data_loader, device, epoch, **kwargs):
    model.train()
    criterion_xent = torch.nn.CrossEntropyLoss()
    xent_losses = AverageMeter()
    if kwargs["use_cent"]:
        criterion_cent = kwargs["cent_loss"]
        optimizer_centloss = kwargs["optimizer_centloss"]
        cent_losses = AverageMeter()
    losses = AverageMeter()
    if kwargs["is_plot"]:
        all_features, all_labels = [], []
    accu_num = torch.zeros(1).to(device)        # 累计预测正确的样本数

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)  # (b, c, h, w) (16, 2, 128, 128)
        sample_num += images.shape[0]
        feature, pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.argmax(1)).sum()
        loss_xent = criterion_xent(pred, labels)
        loss = loss_xent
        if kwargs["use_cent"]:
            loss_cent = criterion_cent(feature, labels.argmax(1))
            if xent_losses.avg < 0.01:
                loss = loss_xent + loss_cent * kwargs["weight_cent"]
            else:
                loss = loss_xent + loss_cent * kwargs["weight_cent"]
            optimizer_centloss.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # by doing so, weight_cent would not impact on the learning of centers
        if kwargs["use_cent"]:
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / kwargs["weight_cent"])
            optimizer_centloss.step()
            cent_losses.update(loss_cent.item(), labels.size(0))
        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        if kwargs["is_plot"]:
            all_features.append(feature.data.cpu().numpy())
            all_labels.append(labels.argmax(1).data.cpu().numpy())
        if kwargs["use_cent"]:
            data_loader.desc = "[train epoch {}] Loss {:.3f} ({:.3f}) XentLoss {:.3f} ({:.3f}) CenterLoss {:.3f} " \
                               "({:.3f}), acc: {:.5f}".format(epoch, losses.val, losses.avg, xent_losses.val,
                                                              xent_losses.avg, cent_losses.val, cent_losses.avg,
                                                              accu_num.item() / sample_num)
        else:
            data_loader.desc = "[train epoch {}] Loss {:.3f} ({:.3f}) acc: {:.5f}"\
                .format(epoch, losses.val, losses.avg, xent_losses.val,xent_losses.avg, cent_losses.val,
                        cent_losses.avg, accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.zero_grad()

    if kwargs["is_plot"]:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, kwargs["num_classes"], epoch, kwargs["plot_path"])
    # return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, is_plot, plot_path, num_classes):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    if is_plot:
        all_features, all_labels = [], []
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        feature, pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device).argmax(1)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        if is_plot:
            all_features.append(feature.data.cpu().numpy())
            all_labels.append(labels.argmax(1).data.cpu().numpy())
        data_loader.desc = "[test epoch {}] loss: {:.3f}, acc: {:.5f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    if is_plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, plot_path)
    # if epoch % 10 == 0:
    #     visualize(feat.data.cpu().numpy(), labs.data.cpu().numpy(), epoch, test_plot_path)

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
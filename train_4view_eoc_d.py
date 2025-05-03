# -*- coding: utf-8 -*-
# coder：Xing_z
# e-mail: zhangxing3212@163.com
import os
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.mstar_data import read_dataset
from model.swinT_cosatt_2view import SwinTransformer

from my_utils.utils_new import train_one_epoch, evaluate
from my_utils.center_loss import CenterLoss

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# project_root = D:/A-资料/pycharm_project/Mutilview_Transformer_Newdataset
device_num = 'cuda:1'  # device id (i.e. 0 or 0,1 or cpu)
if torch.cuda.is_available():
    device = torch.device(device_num if torch.cuda.is_available() else "cpu")
else:
    print("Not find available gpu!")
name = 'eoc_d'
use_cent = True
num_classes = 4
input_view = 4
loss_dim = 2
epochs = 500
batch_size = 64
lr = 0.0001
test_sample = 1
weight_cent = 10
data_types = 'float32'
depth = '222222'
dropout = '222'
dataset_path = project_root + "/dataset" + "/eoc_d/"
save_acc_path = project_root + "/result/historys/"
save_acc_name = f'{name}-{input_view}view_{depth}depth-{batch_size}bz_{dropout}dropout_{use_cent}.npy'
save_weights_path = project_root + f'/result/{name}_{input_view}view_{depth}depth_{batch_size}bz_{dropout}dropout_{use_cent}/'
plot_path = project_root + f'/result/{name}_{input_view}view_{depth}depth_{batch_size}bz_{dropout}dropout_{use_cent}'
load_weights_path = save_weights_path + 'model_weights.pth'
if os.path.exists(save_weights_path) is False:
    print("生成文件夹： " + str(save_weights_path))
    os.makedirs(save_weights_path)
if os.path.exists(save_acc_path) is False:
    os.makedirs(save_acc_path)
    print("生成文件夹： " + str(save_acc_path))


def main():
    # data
    train_dataset = read_dataset(dataset_path, name, 'train', input_view, num_classes, data_types)
    test_dataset = read_dataset(dataset_path, name, 'test', input_view, num_classes, data_types)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # model
    model = SwinTransformer(in_chans=input_view, num_classes=num_classes, embed_dim=16*input_view,
                            batch_size=batch_size, num_heads=(2, 2, 4, 4, 8, 8), window_size=4,
                            depths=(2, 2, 2, 2, 2, 2),
                            drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.2)
    if os.path.exists(load_weights_path):
        model.load_state_dict(torch.load(load_weights_path, map_location=device), strict=False)
    if os.path.exists(save_acc_path + save_acc_name):
        acc_last = np.load(save_acc_path + save_acc_name)
        print("best_acc:" + str(acc_last))
    else:
        acc_last = 0.0
    if data_types == 'float64':
        model.double().to(device)
    elif data_types == 'float32':
        model.to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=lr, weight_decay=5E-2)
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-3)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer=optimizer,
    #     milestones=[100],
    #     gamma=0.1)
    if use_cent:
        cent_loss = CenterLoss(num_classes=num_classes, feat_dim=2, device=device_num)
        optimizer_centloss = torch.optim.SGD(cent_loss.parameters(), lr=0.5)
    for epoch in range(epochs):
        # train
        train_plot_path = os.path.join(plot_path, 'train')
        if use_cent:
            train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch,
                            use_cent=use_cent, cent_loss=cent_loss, optimizer_centloss=optimizer_centloss,
                            weight_cent=weight_cent, is_plot=True, plot_path=train_plot_path, num_classes=num_classes)
        else:
            train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch)
        # lr = lr_scheduler.get_last_lr()[0]
        # lr_scheduler.step()
        # test
        if epoch % test_sample == 0:
            test_plot_path = os.path.join(plot_path, 'test')
            test_loss, test_acc = evaluate(model=model, data_loader=test_loader, device=device, epoch=epoch,
                                           is_plot=True, plot_path=test_plot_path, num_classes=num_classes)
            if test_acc > acc_last:
                torch.save(model.state_dict(), load_weights_path)
                np.save(save_acc_path + save_acc_name, test_acc)
                print('test_acc = ' + str(test_acc) + '\nsave model')
                acc_last = test_acc


if __name__ == '__main__':
    main()

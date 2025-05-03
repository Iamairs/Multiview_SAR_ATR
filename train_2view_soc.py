# -*- coding: utf-8 -*-
# coder：Xing_z
# e-mail: zhangxing3212@163.com
import os
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.mstar_data import read_dataset
from model.swinT import SwinTransformer

from my_utils.utils_new import train_one_epoch, evaluate

# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = r"D:\数据集\MSTAR\multiviews"
model_weight_path = r"D:\modelWeight\Multiview_SAR_ATR"
device_num = 'cuda:0'  # device id (i.e. 0 or 0,1 or cpu)
if torch.cuda.is_available():
    device = torch.device(device_num if torch.cuda.is_available() else "cpu")
else:
    print("Not find available gpu!")
name = 'soc'
num_classes = 10
input_view = 2
dloss_dim = 128
epochs = 200
batch_size = 32
lr = 0.0001
test_sample = 1
data_types = 'float32'
depth = '111111'
dropout = '000'
dataset_path = dataset_path + "/soc/"
save_acc_path = model_weight_path + "/result/historys/"
save_acc_name = f'{name}-{input_view}view_{depth}depth-{batch_size}bz_{dropout}dropout.npy'
save_weights_path = model_weight_path + f'/result/{name}_{input_view}view_{depth}depth_{batch_size}bz_{dropout}dropout/'
feature_dst_path = f'/result/pic/{name}_{input_view}view_{depth}depth_{batch_size}bz_{dropout}dropout'
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
    model = SwinTransformer(in_chans=input_view, num_classes=num_classes, embed_dim=32,
                            batch_size=batch_size,  num_heads=(2, 2, 4, 8, 16, 32), window_size=4,
                            depths=(1, 1, 1, 1, 1, 1))
    if os.path.exists(load_weights_path):
        model.load_state_dict(torch.load(load_weights_path, map_location=device), strict=False)
    if os.path.exists(save_acc_path + save_acc_name):
        acc_last = np.load(save_acc_path + save_acc_name)
        print("best_acc=" + str(acc_last))
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
    for epoch in range(epochs):
        # train
        train_feature_dst_path = os.path.join(feature_dst_path, 'train')
        train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch,
                        input_view=input_view, feature_dst_path=train_feature_dst_path, use_cent=False, is_plot=False)
        # lr = lr_scheduler.get_last_lr()[0]
        # lr_scheduler.step()
        # test
        if epoch % test_sample == 0:
            test_feature_dst_path = os.path.join(feature_dst_path, 'test')
            test_loss, test_acc = evaluate(model=model, data_loader=test_loader, device=device, epoch=epoch,
                                           input_view=input_view, feature_dst_path=test_feature_dst_path)
            if test_acc > acc_last:
                torch.save(model.state_dict(), load_weights_path)
                np.save(save_acc_path + save_acc_name, test_acc)
                print('test_acc = ' + str(test_acc) + '\nsave model')
                acc_last = test_acc


if __name__ == '__main__':
    main()

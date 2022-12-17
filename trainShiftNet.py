# -- coding:utf-8

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from config_shift import *
from dataset import BuildingDataset
import pandas as pd
from ShiftNet import ShiftNet
import os
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.891972, 0.93623203, 0.9399001], std = [0.16588122, 0.090553254, 0.12211764])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # test_rect_train
    # transforms.Normalize(mean = [0.70381516, 0.8888911, 0.92238843], std =  [0.40284097, 0.11763619, 0.15052465])
    # test_area_train
    # transforms.Normalize(mean=[0.84431416, 0.9468392, 0.96895444], std = [0.2916492, 0.08882934, 0.098634705])
    # valid_test2
    transforms.Normalize(mean = [0.8904361, 0.9363585, 0.94014686], std = [0.165799, 0.09002642, 0.12338792])
    # transforms.Normalize(mean = [0.8904361, 0.9363585, 0.94014686], std = [0.165799, 0.09002642, 0.12338792])
    # compareValid
    # transforms.Normalize(mean = [0.84808147, 0.9096524, 0.91053045], std = [0.20253241, 0.10373465, 0.14877456])
])

df = pd.DataFrame(columns=['loss', 'accuracy'])
# 定义训练函数
def train(dataloader, model, loss_fn, optimizer, epoch):
    loss, current, n = 0.0, 0.0, 0
    model.train()
    # enumerate返回为数据和标签还有批次
    for batch, (X, y) in enumerate(dataloader):
        # 前向传播
        X, y = X.to(device), y.to(device)
        output = model(X)
        cur_loss = loss_fn(output, y)
        # output1 = output.squeeze(-1)
        # cur_loss = loss_fn(output1, y.float())
        # torch.max返回每行最大的概率和最大概率的索引,由于批次是16，所以返回16个概率和索引
        _, pred = torch.max(output, axis=1)

        # 计算每批次的准确率， output.shape[0]为该批次的多少
        cur_acc = torch.sum(y == pred) / output.shape[0]
        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        # 取出loss值和精度值
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
        rate = (batch + 1) / train_num
        writer.add_scalar('Train/Loss_batch', cur_loss, epoch * len(dataloader) + n)
        # writer.add_scalar('Train/acc', cur_loss, optimizer.param_groups[0]['lr'])
        # print(f"train loss: {rate * 100:.1f}%,{cur_loss:.3f}")
    print(f"train_loss' : {(loss / n):.3f}  train_acc : {(current / n):.3f}")
    # print(f"train_acc' : {(current / n):.3f}")
    writer.add_scalar('Train/Loss', loss / n, epoch)
    writer.add_scalar('Train/Acc', current / n, epoch)

# 定义验证函数
def val(dataloader, model, loss_fn, epoch):
    # 将模型转为验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    # 非训练，推理期用到（测试时模型参数不用更新， 所以no_grad）
    # print(torch.no_grad)
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            # output1 = output.squeeze(-1)
            # cur_loss = loss_fn(output1, y.float())
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
            # writer.add_scalar('Valid/valid_batch', cur_loss, epoch*len(dataloader)+n)
        print(f"valid_loss' : {(loss / n):.3f}  valid_acc : {(current / n):.3f}")
        # print(f"valid_acc' : {(current / n):.3f}")
    writer.add_scalar('Valid/Loss', loss / n, epoch)
    writer.add_scalar('Valid/Acc', current / n, epoch)
    df.loc[epoch] = {'loss': loss / n, 'accuracy': current / n}
    return current / n

if __name__ == '__main__':
    s = f"Diffnet,{train_dir},{valid_dir},batch{BATCH_SIZE},lr{LR},wd{weight_decay}"
    writer = SummaryWriter(comment=s)
    # build MyDataset
    # class_sample_counts = [33288,4128] #compareTrain
    class_sample_counts = [38220, 5328]  # fixtrain_test2
    # class_sample_counts = [32412,3984] # test_rect_train
    # class_sample_counts = [38220, 5328]
    # class_sample_counts = [15028,1832]
    # class_sample_counts = [15394,1832]
    # class_sample_counts = [45084,5496]
    # class_sample_counts = [46876,6264] # 数据集不同类别的比例
    # class_sample_counts = [46008,5794,1042]
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    # 这个 get_classes_for_all_imgs是关键
    train_data = BuildingDataset(data_dir=train_dir, transform=transform)
    train_targets = train_data.get_classes_for_all_imgs()
    samples_weights = weights[train_targets]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    valid_data = BuildingDataset(data_dir=valid_dir, transform=transform_val)

    # build DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                              pin_memory=True, sampler=sampler)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True,
                              shuffle=True)
    # AlexNet model and training
    net = ShiftNet(num_classes=N_FEATURES, init_weights=True)
    # net = GoogleNet(num_class=N_FEATURES)
    # 模拟输入数据，进行网络可视化
    # input_data = Variable(torch.rand(16, 3, 224, 224))
    # with writer:
    #     writer.add_graph(net, (input_data,))pythp
    # 模型进入GPU
    # if torch.cuda.device_count() > 1:
    #     print("Use", torch.cuda.device_count(), 'gpus')
    #     net = nn.DataParallel(net)
    net.to(device)

    # 定义损失函数（交叉熵损失）
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = focal_loss(alpha= [1.12196,9.20306],gamma=2,num_classes=2)
    # loss_fn = nn.BCEWithLogitsLoss()

    # 定义优化器,SGD,
    # optimizer = optim.Adam(net.parameters(), lr=LR,weight_decay=weight_decay)
    optimizer = optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    # 学习率按数组自定义变化
    lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # 开始训练
    epoch = MAX_EPOCH
    min_acc = 0
    train_num = len(train_loader)
    for t in range(epoch):
        start = time.time()
        print(f"epoch{t + 1}\n-------------------")
        train(train_loader, net, loss_fn, optimizer, t)
        a = val(valid_loader, net, loss_fn, t)
        lr_scheduler.step()
        print("目前学习率:", optimizer.param_groups[0]['lr'])
        # 保存最好的模型权重文件
        if a > min_acc:
            folder = 'save_model'
            if not os.path.exists(folder):
                os.mkdir('save_model')
            min_acc = a
            print('save best model', )
            torch.save(net.state_dict(), "save_model/shifted/best_model.pth")
        # 保存最后的权重文件
        torch.save(net.state_dict(), "save_model/shifted/every_model.pth")
        if t == epoch - 1:
            torch.save(net.state_dict(), "save_model/shifted/last_model.pth")
        finish = time.time()
        time_elapsed = finish - start
        print('本次训练耗时 {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    print(f'** Finished Training **')
    df.to_csv('runs/train.txt', index=True, sep=';')


import numpy as np
import time

import torchvision
from matplotlib import pyplot as plt
from sklearn.compose import TransformedTargetRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from GoogleNet import GoogLeNet
# from config_google import *
from dataset import BuildingDataset
import pandas as pd

from DifferenceNet import DifferenceNet
from config_diff import *
# from STNet import STNet
# from config_stn import *
import os
# from focal_loss import focal_loss
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
import torchmetrics

from trainSTNet import convert_image_np

plt.ion()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # function_test_20
    # transforms.Normalize(mean=[0.9318218, 0.9599232, 0.96266234], std=[0.13674922, 0.07559318, 0.09820192])

])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # test_rect_train
    # transforms.Normalize(mean = [0.70381516, 0.8888911, 0.92238843], std =  [0.40284097, 0.11763619, 0.15052465])

])

df = pd.DataFrame(columns=['loss', 'accuracy'])


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer, epoch):
    loss, current, n = 0.0, 0.0, 0
    # test_acc = torchmetrics.Accuracy().to(device)
    test_recall = torchmetrics.Recall(average='none', num_classes=N_FEATURES).to(device)
    test_precision = torchmetrics.Precision(average='none', num_classes=N_FEATURES).to(device)
    test_F1 = torchmetrics.F1(num_classes=N_FEATURES, average='none').to(device)
    model.train()
    # enumerate返回为数据和标签还有批次
    for batch, (X, y) in enumerate(dataloader):
        # 前向传播
        X, y = X.to(device), y.to(device)
        # print(y)
        # print(type(y))
        output, output2, output1 = model(X)
        # output = model(X)
        cur_loss0 = loss_fn(output, y)
        cur_loss1 = loss_fn(output1, y)
        cur_loss2 = loss_fn(output2, y)
        # output1 = output.squeeze(-1)
        # cur_loss = loss_fn(output1, y.float())
        # torch.max返回每行最大的概率和最大概率的索引,由于批次是16，所以返回16个概率和索引
        _, pred = torch.max(output, axis=1)
        # 计算每批次的准确率， output.shape[0]为该批次的多少
        cur_acc = torch.sum(y == pred) / output.shape[0]
        cur_loss = cur_loss0 + cur_loss1 * 0.3 + cur_loss2 * 0.3

        # test_acc(pred.argmax(1), y)
        # test_recall = test_recall.to(device)
        # test_precision = test_precision.to(device)
        # test_acc = test_acc.to(device)

        # test_acc(output.argmax(1), y)
        test_F1(output.argmax(1), y)
        test_recall(output.argmax(1), y)
        test_precision(output.argmax(1), y)

        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        # 取出loss值和精度值
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
        rate = (batch + 1) / train_num
        # print(f"train loss: {rate * 100:.1f}%,{cur_loss:.3f}")
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_F1 = test_F1.compute()
    # total_acc = test_acc.compute()
    print(f"train_loss' : {(loss / n):.3f}  train_acc : {(current / n):.3f}")
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)
    print("F1 of every test dataset class: ", total_F1)
    writer.add_scalar('Train/Loss', loss / n, epoch)
    writer.add_scalar('Train/Acc', current / n, epoch)
    writer.add_scalar('Train/Recall', total_recall[1].item(), epoch)
    writer.add_scalar('Train/Precision', total_precision[1].item(), epoch)
    writer.add_scalar('Train/F1', total_F1[1].item(), epoch)
    test_precision.reset()
    test_recall.reset()
    test_F1.reset()


# 定义验证函数
def val(dataloader, model, loss_fn, epoch):
    # 将模型转为验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    test_recall = torchmetrics.Recall(average='none', num_classes=N_FEATURES).to(device)
    test_precision = torchmetrics.Precision(average='none', num_classes=N_FEATURES).to(device)
    test_F1 = torchmetrics.F1(average='none', num_classes=N_FEATURES).to(device)
    # 非训练，推理期用到（测试时模型参数不用更新， 所以no_grad）
    # print(torch.no_grad)
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            # output1 = output.squeeze(-1)
            # cur_loss = loss_fn(output1, y.float())
            test_F1(output.argmax(1), y)
            test_recall(output.argmax(1), y)
            test_precision(output.argmax(1), y)

            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_F1 = test_F1.compute()
    print(f"valid_loss' : {(loss / n):.3f}  valid_acc : {(current / n):.3f}")
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)
    print("F1 of every test dataset class: ", total_F1)
    writer.add_scalar('Valid/Loss', loss / n, epoch)
    writer.add_scalar('Valid/Acc', current / n, epoch)
    writer.add_scalar('Valid/Recall', total_recall[1].item(), epoch)
    writer.add_scalar('Valid/Precision', total_precision[1].item(), epoch)
    writer.add_scalar('Valid/F1', total_F1[1].item(), epoch)
    test_precision.reset()
    test_recall.reset()
    test_F1.reset()
    df.loc[epoch] = {'loss': loss / n, 'accuracy': current / n}
    return current / n

# stn可视化函数
def visualize_stn(model):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(train_loader))[0].to(device)

        input_tensor = data.cpu()
        #tensor1 = model.stn(data)
        transformed_input_tensor = model.stn(input_tensor).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

if __name__ == '__main__':
    s = f"googlenet,{train_dir},{valid_dir},batch{BATCH_SIZE},lr{LR},wd{weight_decay_f}"
    writer = SummaryWriter(comment=s)
    # build MyDataset
    # class_sample_counts = [32412,3984] # test_rect_train
    # class_sample_counts = [38220, 5328] # fixtrain_test2
    # class_sample_counts = [33444,4128] #function_test_5
    # class_sample_counts = [33456,4128] #function_test_10
    # class_sample_counts = [36432, 4344] # function_test_20_old
    class_sample_counts = [33444, 4128]  # function_test_20
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    # 这个 get_classes_for_all_imgs是关键
    train_data = BuildingDataset(data_dir=train_dir, transform=transform)
    train_targets = train_data.get_classes_for_all_imgs()
    samples_weights = weights[train_targets]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    valid_data = BuildingDataset(data_dir=valid_dir, transform=transform_val)

    # build DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
                              sampler=sampler)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=True)
    # AlexNet model and training
    # net = AlexNet(num_classes=N_FEATURES, init_weights=True)
    net = DifferenceNet(num_classes=N_FEATURES, init_weights=True, aux_logits=True)
    # net = STNet(num_classes=N_FEATURES, init_weights=True, aux_logits=True)
    # 模拟输入数据，进行网络可视化
    # input_data = Variable(torch.rand(16, 3, 224, 224))
    # with writer:
    #     writer.add_graph(net, (input_data,))
    # 模型进入GPU
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        net = nn.DataParallel(net)
    model = net.to(device)

    # 定义损失函数（交叉熵损失）
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = focal_loss(alpha= [1.12196,9.20306],gamma=2,num_classes=2)
    # loss_fn = nn.BCEWithLogitsLoss()

    # 定义优化器,SGD,
    # optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay_f)
    optimizer = optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=weight_decay_f)
    # 学习率每隔10epoch变为原来的0.1
    # lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

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
                os.mkdir('../../本地深度学习项目/Feature-Fusion-CNN-master/save_model')
            min_acc = a
            print('save best model', )
            torch.save(net.state_dict(), "save_model/googleNet/best_model.pth")
        torch.save(net.state_dict(), "save_model/googleNet/every_model.pth")
        if float(a) < float(85):
            torch.save(net.state_dict(), "save_model/googleNet/lunwen_model.pth")
        # 保存最后的权重文件
        if t == epoch - 1:
            torch.save(net.state_dict(), "save_model/googleNet/last_model.pth")
        finish = time.time()
        time_elapsed = finish - start
        print('本次训练耗时 {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        visualize_stn(model)
        plt.ioff()
        plt.show()
    print(f'** Finished Training **')
    df.to_csv('runs/train_focal.txt', index=True, sep=';')
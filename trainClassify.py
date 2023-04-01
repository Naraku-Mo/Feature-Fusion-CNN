# -- coding:utf-8
# Feature Fusion
import os
import random
import string
from collections import defaultdict

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
from ShiftNet import ShiftNet
from DifferenceNet import DifferenceNet
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


N_FEATURES = 2
device = 'cuda:0'
# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # train_shift
    # transforms.Normalize(mean=[0.89731014, 0.9391688, 0.9396487], std=[0.18013711, 0.09479259, 0.12887895])
    # origin_diff
    # transforms.Normalize(mean=[0.9251727, 0.95890087, 0.9619809], std=[0.14847293, 0.07731944, 0.101800375])
])


class DiffImgDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)
        self.imgs_dict = {}
        for img in self.imgs:
            id_num = int(img.split('groupid')[1].split('diff')[0])
            label_num = int(img.split('diff')[1].split('x')[0])
            if id_num not in self.imgs_dict:
                self.imgs_dict[id_num] = []
            self.imgs_dict[id_num].append((img, label_num))

    def __len__(self):
        return len(self.imgs_dict)

    def __getitem__(self, idx):
        id_num = list(self.imgs_dict.keys())[idx]
        imgs_list = self.imgs_dict[id_num]
        if len(imgs_list) > 4:
            imgs_list = sorted(imgs_list, key=lambda x: x[1], reverse=True)[:4]
        elif len(imgs_list) < 4:
            while len(imgs_list) < 4:
                imgs_list.append(random.choice(imgs_list))
        imgs = []
        for img_name, label in imgs_list:
            img_path = os.path.join(self.img_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        return torch.stack(imgs), id_num


shiftimg_dir = 'train_shift'


class ShiftImgDataset(Dataset):
    def __init__(self, data_path=shiftimg_dir, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.imgs = os.listdir(data_path)
        self.imgs_dict = {}
        for img in self.imgs:
            id_num = int(img.split('groupid')[1].split('ex')[0])
            label_num = int(img.split('ex')[1].split('.')[0])
            if id_num not in self.imgs_dict:
                self.imgs_dict[id_num] = []
            self.imgs_dict[id_num].append((img, label_num))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        id = list(self.imgs_dict.keys())[index]
        imgs_list = self.imgs_dict[id]
        for img_name, label in imgs_list:
            img_path = os.path.join(self.data_path, img_name)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, id, label


def extract_features(model, dataloader):
    features_dict = {}
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            imgs, id_nums, labels = data
            batch_size = imgs.size(0)
            imgs = imgs.view(-1, 3, 224, 224)
            features = model(imgs)
            features = features.view(batch_size, -1)
            for i in range(batch_size):
                features_dict[id_nums[i].item()] = (features[i].cpu().numpy(), labels[i].item())
    return features_dict


# shiftimg_dir = 'train_shift'
diffimg_dir = 'train_diff'
difffeatures_dir = 'diff_features'
shiftfeatures_dir = 'shift_features'
if not os.path.exists(difffeatures_dir):
    os.makedirs(difffeatures_dir)

diffdataset = DiffImgDataset(img_dir=diffimg_dir, transform=transform)
diffimgloader = DataLoader(dataset=diffdataset, batch_size=64)
shiftdataset = ShiftImgDataset(transform=transform)
shiftimgloader = DataLoader(dataset=shiftdataset, batch_size=64)

# 加载预训练模型
# 载入模型参数
DiffNet = DifferenceNet(num_classes=2, init_weights=True)
DiffNet.load_state_dict(torch.load(r'save_model\different\last_model.pth'))
# DiffNet.cuda()
ShiftNet = ShiftNet(num_classes=2, init_weights=True)
ShiftNet.load_state_dict(torch.load(r'save_model\shift\last_model.pth'))

# features_diff = extract_features(model=DiffNet, dataloader=diffimgloader)
# for id_num in tqdm(features_diff):
#     feature_path = os.path.join(features_dir, f'id{id_num}.npy')
#     np.save(feature_path, features_diff[id_num])

# features_shift = extract_features(model=ShiftNet, dataloader=shiftimgloader)
# for id_num in tqdm(features_shift):
#     feature_path = os.path.join(shiftfeatures_dir, f'id{id_num}ex{features_shift[id_num][1]}.npy')
#     np.save(feature_path, features_shift[id_num][0])

# fusion_features_dict = {}
# for imgs, ids, labels in tqdm(shiftimgloader):
#     with torch.no_grad():
#         features = ShiftNet(imgs)
#     for i in range(len(ids)):
#         id = ids[i]
#         label = labels[i]
#         feature_path = os.path.join('shift_features', f'id{id}ex{label}.npy')
#         np.save(feature_path, features[i])
# print(f"特征融合，id' : {id}  类别为 : {label}")
# fusion_features_dict[id].append((torch.cat((features_diff[id], features[i]), dim=0), label))

# 遍历文件夹中的所有文件
fusion_features_dict = {}
# 准备标签值
labels_dict = {}
for file_name in tqdm(os.listdir(shiftfeatures_dir)):
    # 检查文件是否为 .npy 文件
    if file_name.endswith('.npy'):
        # 提取 id
        id = file_name.split('id')[1].split('ex')[0]
        label = int(file_name.split('ex')[1].split('.')[0])
        if label==0:
            label=0
        elif label!=0:
            label =1
        # 读取特征
        diff_filename=f'id{id}.npy'
        if diff_filename in os.listdir(difffeatures_dir):
            diff_feature = np.load(os.path.join(difffeatures_dir, diff_filename))
        shift_feature = np.load(os.path.join(shiftfeatures_dir, file_name))
        # 融合特征
        fusion_features_dict[id]=np.concatenate((shift_feature, diff_feature), axis=0)
        labels_dict[id] = label
        # if id in features_shift:
        #     fusion_features_dict[id] = (
        #     np.concatenate((features_shift[id][0], difffeature), axis=0), features_shift[id][1])
        # else:
        #     fusion_features_dict[id] = (difffeature, features_shift[id][1])



# 准备 SVM 输入数据
X = []
y = []
for id in tqdm(fusion_features_dict):
    # print("X", fusion_features_dict[id])
    # print("y", labels_dict[id])
    X.append(fusion_features_dict[id])
    y.append(labels_dict[id])  # 根据您的数据设置标签

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# 计算类别权重
class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw = dict(enumerate(class_weight))
writer = SummaryWriter()
# 训练 SVM 模型并输出结果
clf = svm.SVC(class_weight=cw)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# 计算分类评价指标
target_names = ['class 0', 'class 1']
print(classification_report(y_train, y_train_pred, target_names=target_names))
print(classification_report(y_test, y_test_pred, target_names=target_names))
# t_accuracy = accuracy_score(y_train, y_train_pred)
# t_precision = precision_score(y_train, y_train_pred)
# t_recall = recall_score(y_train, y_train_pred)
# t_f1 = f1_score(y_train, y_train_pred)
# v_accuracy = accuracy_score(y_test, y_test_pred)
# v_precision = precision_score(y_test, y_test_pred)
# v_recall = recall_score(y_test, y_test_pred)
# v_f1 = f1_score(y_test, y_test_pred)
# print("accuracy:",accuracy)

# 可视化结果
#
# writer.add_scalar('Valid/Loss', loss / n, epoch)
# writer.add_scalar('Valid/Acc', current / n, epoch)
# writer.add_scalar('Valid/Recall', total_recall[1].item(), epoch)
# writer.add_scalar('Valid/Precision', total_precision[1].item(), epoch)
# writer.add_scalar('Valid/F1', total_F1[1].item(), epoch)
# writer.close()
#


# # 读取图像并按 id 分组
# images_by_id = defaultdict(list)
# for filename in os.listdir(image_folder):
#     if filename.endswith('.jpg'):
#         image_id = filename.split('groupid')[1].split('diff')[0]
#         label = filename.split('diff')[1].split('x')[0]
#         # id_, label = filename.split('.')[0].split('_')
#         images_by_id[image_id].append((filename, label))
#
# # 处理每个 id 对应的图像
# for image_id, images in images_by_id.items():
#     # 如果图像多于四张，则随机选择四张
#     if len(images) > 4:
#         images = random.sample(images, 4)
#     # 如果图像少于四张，则进行补全
#     elif len(images) < 4:
#         # 方法一：补零
#         # features = [extract_feature(image) for image, _ in images]
#         # while len(features) < 4:
#         #     features.append(torch.zeros_like(features[0]))
#         # 方法二：随机复制
#         features = [extract_feature(image) for image, _ in images]
#         while len(features) < 4:
#             features.append(random.choice(features))
#     else:
#         features = [extract_feature(image) for image, _ in images]
#
#     # 拼接特征并存储
#     feature = torch.cat(features)
#     label = images[0][1]
#     torch.save(feature, os.path.join(features_dir, f'{image_id}_diff.pt'))
#
# def extract_feature(image):
#     """提取图像特征"""
#     image = Image.open(os.path.join(image_dir, image))
#     input_tensor = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         output = DiffNet(input_tensor)
#     return output.squeeze()

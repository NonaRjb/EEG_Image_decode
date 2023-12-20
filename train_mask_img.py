import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from itertools import combinations

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from cls_eegdatasets import EEGDataset
from eegencoder import eeg_encoder
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from utils import wandb_logger
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from lavis.models.clip_models.loss import ClipLoss

# 定义自我注意模块
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.Wq = nn.Linear(input_dim, input_dim)
        self.Wk = nn.Linear(input_dim, input_dim)
        self.Wv = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        
        d = q.size(-1)  # 获取输入x的维度

        # 计算Scaled Dot-Product Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (d**0.5)
        attn = self.softmax(attn)

        # 使用Attention权重对V进行加权求和
        out = torch.matmul(attn, v)

        return out
    
class GraphAttention(nn.Module):  
    def __init__(self, ch, T, hidden_size):  
        super(GraphAttention, self).__init__()  
        self.ch = ch  
        self.T = T  
        self.hidden_size = hidden_size  
        self.W = nn.Linear(T, hidden_size)  
        self.a = nn.Linear(hidden_size * 2, 1)  
        self.leakyrelu = nn.LeakyReLU(0.2)  
  
    def forward(self, x):  
        batch_size = x.size(0)  
        x = x.view(-1, self.ch * self.T)  
        x = self.W(x)  
        x = x.view(batch_size, self.ch, self.hidden_size)  
        a_input = torch.cat([x.repeat(1, self.ch).view(-1, self.hidden_size * 2), x.repeat(self.ch, 1)], dim=1)  
        a = self.leakyrelu(self.a(a_input)).squeeze(2)  
        alpha = torch.exp(a) / torch.exp(a).sum(dim=1, keepdim=True)  
        n_prime = torch.matmul(alpha, x)  
        return n_prime.view(batch_size, self.ch, self.hidden_size)

def make_block(h_c, h_l, m1, m2, s2, ch):
    block = nn.Sequential(
        # SelfAttention(h_l),
        # GraphAttention(ch, 100, 512),
        nn.Conv1d(1, 40, kernel_size=(1,m1), stride=1),  # 一维卷积捕捉时间特征
        nn.AvgPool1d(kernel_size=(1,m2), stride=(1,s2)),  # 平均池化
        nn.Conv1d(40, 40, kernel_size=(ch,1), stride=1),  # 处理空间特征的一维卷积
        nn.BatchNorm1d(h_c),
        nn.GELU(),
        nn.LayerNorm(h_l),
        nn.Linear(h_l, h_l), 
        nn.GELU(),
        Rearrange('B C L->B L C'),
        nn.LayerNorm(h_c),
        nn.Linear(h_c, h_c), 
        nn.GELU(),
        Rearrange('B L C->B C L'),
    )
    return block

class Projector(nn.Module):

    def __init__(self, x, h_dim=(64, 512), n_hidden_layer=4, dropout_rate=0.5):
        # in_features: (c, l)
        super().__init__()
        c, l = x.shape[1], x.shape[2]
        print(f'Channels: {c}, Length: {l}')
        h_c, h_l = h_dim
        c_o, l_o = 1, 512

        self.input_layer = nn.Sequential(
            nn.LayerNorm(l),
            nn.Linear(l, h_l), 
            nn.GELU(),
            # nn.Dropout(dropout_rate),
            Rearrange('B C L->B L C'),
            nn.LayerNorm(c),
            nn.Linear(c, h_c), 
            nn.GELU(),
            # nn.Dropout(dropout_rate),
            Rearrange('B L C->B C L'),
            # print(in_features.shape)
            # print(c.shape)
            print(f'Channels: {c}, Length: {l}')
        )
        
        self.output_layer = nn.Sequential(
            nn.LayerNorm(h_l),
            nn.Linear(h_l, l_o), 
            nn.GELU(),
            # nn.Dropout(dropout_rate),
            Rearrange('B C L->B L C'),
            nn.LayerNorm(h_c),
            nn.Linear(h_c, c_o), 
            nn.GELU(),
            # nn.Dropout(dropout_rate),
            Rearrange('B L C->B (C L)'),
        )
        
        self.blocks = nn.Sequential(*[
            make_block(h_c, h_l, m1=25, m2=51, s2=5, ch=17) for _ in range(n_hidden_layer)
            ])

        
        self.projector = nn.Sequential(*[
            self.input_layer,
            self.blocks,
            self.output_layer,
        ])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
    
    def forward(self, x):
        eeg_embeds = self.projector(x)
        eeg_features = F.normalize(eeg_embeds, dim=-1)
        return eeg_features

class eegEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.mae_encoder = eeg_encoder() 
        self.freeze_mae()
      
    def freeze_mae(self):
        for param in self.mae_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        eeg_embeds = self.mae_encoder(x)
        print("eeg_embeds", eeg_embeds.shape)
        return eeg_embeds


def train_model(model, dataloader, optimizer, device, text_features_all, img_features_all):
    model.train()
    text_features_all = text_features_all.to(device).float() # (n_cls, d)
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        print("Device of tensor:", eeg_data.device)
        print("eeg_data", eeg_data.shape)        
        optimizer.zero_grad()
        
        eeg_features = model(eeg_data).float() #eeg_data torch.Size([512, 17, 51])
        # projector = Projector(eeg_features) #eeg_embeds torch.Size([512, 17, 512])
        logit_scale = model.logit_scale
        # eeg_features = projector.forward(eeg_embeds)
        # logit_scale = projector.logit_scale        
        loss = model.loss_func(eeg_features, img_features, logit_scale)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        
        logits = logit_scale * eeg_features @ text_features_all.T # (n_batch, n_cls)
        predicted = torch.argmax(logits, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()

    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy

def evaluate_model(model, dataloader, device, text_features_all, img_features_all, k):
    model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0

    # 获取所有独特的类别
    all_labels = set(range(text_features_all.size(0)))

    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            
            eeg_features = model(eeg_data).float()
            logit_scale = model.logit_scale 

            loss = model.loss_func(eeg_features, text_features, logit_scale)
            total_loss += loss.item()

            for idx, label in enumerate(labels):
                # 先从除了正确类别之外的类别中选择 k-1 个
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_text_features = text_features_all[selected_classes]
                
                # 计算对应的 logits
                logits_single = logit_scale * eeg_features[idx] @ selected_text_features.T

                # 获取预测的类别
                predicted_label = selected_classes[torch.argmax(logits_single).item()]

                if predicted_label == label.item():
                    correct += 1

                total += 1

    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy

def main_train_loop(model, train_dataloader, test_dataloader, optimizer, device, 
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None):
    logger = wandb_logger(config) if logger else None
    logger.watch(model,logger) 
    
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}

    for epoch in range(config['epochs']):
        # 训练模型
        train_loss, train_accuracy = train_model(model, train_dataloader, optimizer, device, text_features_train_all, img_features_train_all)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 评估模型
        test_loss, test_accuracy = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all,k=200)
        _, v2_acc = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 2)
        _, v4_acc = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 4)
        _, v10_acc = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 10)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)

        logger.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "v2 Accuracy": v2_acc,
            "v4 Accuracy": v4_acc,
            "v10 Accuracy": v10_acc,
            "Epoch": epoch
        })

        print(f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Epoch {epoch + 1}/{config['epochs']} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc}")
    
# 创建5个子图
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # 损失图
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # 总体正确率图
    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    # 以下是你新添加的三个图，这里假设你已经计算了对应的正确率
    # 2分类正确率图
    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    # 4分类正确率图
    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    # 10分类正确率图
    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # 构造你要注释的字符串信息
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
                f"v2_acc:{best_epoch_info['v2_acc']:.4f}\n"
                f"v4_acc:{best_epoch_info['v4_acc']:.4f}\n"
                f"v10_acc:{best_epoch_info['v10_acc']:.4f}")

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)


    # 添加大标题
    plt.suptitle('best_model_lr=3e-4_img_maskeeg', fontsize=16, y=1.05)
    plt.savefig('best_model_lr=3e-4_img_maskeeg.png')
    logger.log({"Plots": logger.Image(plt)})
    logger.finish()

def main():
    config = {
        # 'project': 'eeg_clf',
        # 'entity': 'vis-obj',
        # 'name': 'lr=3e-4_img_maskeeg_注意机制2', 
        'project': 'BCI',
        'entity': 'sustech_rethinkingbci',
        'name': '', 
        
        'lr': 3e-4, 
        'epochs': 50, 
        'batch_size': 512, 
        'logger': True, 
    }

   # Instantiate the dataset and dataloader
    data_path = '/home/geek/Workspace/BCI/Data/THINGS/EEG/osfstorage-archive'  # Replace with the path to your data
    train_dataset = EEGDataset(data_path, train=True)
    test_dataset = EEGDataset(data_path, train=False)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=0, drop_last=True)
    
    text_features_train_all = train_dataset.text_features
    text_features_test_all = test_dataset.text_features
    img_features_train_all = train_dataset.img_features
    img_features_test_all = test_dataset.img_features
    
    
    # 模型和优化器定义
    in_features = (17, 512)
    model = eegEncoder(in_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    print('number of parameters:', sum([p.numel() for p in model.parameters()]))

    # 确定使用的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    main_train_loop(model, train_loader, test_loader, optimizer, device, 
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=config['logger'])
    
if __name__ == '__main__':
    main()
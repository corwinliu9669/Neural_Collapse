import torch
from model import ResNet18
from torch.optim import SGD
import matplotlib.pyplot as plt 
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
from utils import *
from copy import deepcopy


## 参数设置
setting = {}
setting['epochs'] = 350
setting['lr'] = 0.1
setting['use_gpu'] = True
setting['lr_shedule'] =  [350 //3, 700//3 ]
setting['lr_decay'] = 0.05
setting['batch_size'] = 128
setting['dataset'] = 'CIAFR10'
#### 导入数据
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/root/data/cifar', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=setting['batch_size'], shuffle=True,
        num_workers=8, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/root/data/cifar', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=setting['batch_size'], shuffle=False,
        num_workers=8, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/root/data/cifar', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=setting['batch_size'], shuffle=False,
        num_workers=8, pin_memory=True)


#### 模型
model = ResNet18()
if setting['use_gpu'] == True:
    model = model.cuda()

### 优化器
optimizer = SGD(model.parameters(), lr=setting['lr'], momentum=0.9, weight_decay=5e-4)


#### 训练过程
if setting['use_gpu']:
    device = 'cuda'
else:
    device = 'cpu'

train_acc = []
test_acc = []
eq_norm = []
eq_norm_w = []
eq_ang = []
eq_ang_w = []
eq_ang_max = []
eq_ang_w_max = []
classifier_collapse = []
variation_collapse = []
criterion  = nn.CrossEntropyLoss()
for ep in range(1, setting['epochs']+1):
    ### train
    model.train()
    descent_lr(ep, optimizer, setting['lr'], setting['lr_decay'], setting['lr_shedule'])
    loss_val = 0
    correct = num = 0
    for iter, pack in enumerate(train_loader):
        data, target = pack['image'].to(device), pack['label'].to(device)
        logits = model(data)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred = logits.max(1)
        loss_val += loss.item()
        correct += pred.eq(target).sum().item()
        num += data.shape[0]
        if (iter + 1) % 50 == 0:
            print('*******************************')
            print('epoch : ', ep )
            print('iteration : ', iter + 1)
            print('loss : ', loss_val/100)
            print('Correct : ', correct)
            print('Num : ', num)
            print('Train ACC : ', correct/num)
            correct = num = 0
            loss_val = 0
    ### test
    with torch.no_grad():
        print('Val ACC')
        train_acc_e = evaluate_batch(model, val_loader, device)
        print('Test ACC')
        test_acc_e = evaluate_batch(model, test_loader, device)
        train_acc.append(train_acc_e)
        test_acc.append(test_acc_e)
        equal_norm, clas_weight, equal_ang, equal_ang_2, intra_var = neural_collapse_embedding(model, val_loader, device)
  #      print(model.state_dict()['linear.weight'].size())
        tmp_weight =  deepcopy(model.state_dict()['linear.weight'].cpu().detach().numpy())
 #       print(tmp_weight)
        equinorm_weight, n_weight, equal_ang_w, equal_ang_w_2 = weight_feature(tmp_weight)
        del tmp_weight
        eq_norm.append(equal_norm)
        eq_norm_w.append(equinorm_weight)
        eq_ang.append(equal_ang)
        eq_ang_w.append(equal_ang_w)
        eq_ang_max.append(equal_ang_2)
        eq_ang_w_max.append(equal_ang_w_2)
        variation_collapse.append(intra_var)
        classifier_collapse.append(np.linalg.norm(clas_weight-n_weight, ord='fro'))
    ### record information

#zero_train_error_epoch = [i for i in range(len(train_acc)) if train_acc[i] > 0.999][0]
#print('Zero Training Error')
#print(zero_train_error_epoch)
zero_train_error_epoch = 30
### plot
### training curve
plt.figure()
plt.title('Train ACC')
plt.plot(range(1, len(train_acc)+1), train_acc)
plt.vlines(zero_train_error_epoch, 0, 1, colors = "r", linestyles = "dashed")
plt.savefig('train_acc.png')
plt.close()
### testing curve
plt.figure()
plt.title('Test ACC')
plt.plot(range(1, len(test_acc)+1), test_acc)
plt.vlines(zero_train_error_epoch, 0, 1, colors = "r", linestyles = "dashed")
plt.savefig('test_acc.png')
plt.close()
### equal norm
plt.figure()
plt.title('Equal Norm')
plt.plot(range(1, len(eq_norm)+1), eq_norm,label='Activation')
plt.plot(range(1, len(eq_norm)+1), eq_norm_w,label='Weight')
plt.legend()
plt.vlines(zero_train_error_epoch, 0, 1, colors = "r", linestyles = "dashed", )
plt.savefig('equalnorm.png')
plt.close()

### equal ang
plt.figure()
plt.title('Equal Ang')
plt.plot(range(1, len(eq_norm)+1), eq_ang,label='Activation')
plt.plot(range(1, len(eq_norm)+1), eq_ang_w,label='Weight')
plt.legend()
plt.vlines(zero_train_error_epoch, 0, 1, colors = "r", linestyles = "dashed", )
plt.savefig('equalang.png')
plt.close()


### equal ang
plt.figure()
plt.title('Equal Ang MAx')
plt.plot(range(1, len(eq_norm)+1), eq_ang_max,label='Activation')
plt.plot(range(1, len(eq_norm)+1), eq_ang_w_max,label='Weight')
plt.legend()
plt.vlines(zero_train_error_epoch, 0, 1, colors = "r", linestyles = "dashed", )
plt.savefig('equalang_max.png')
###  collapse

plt.figure()
plt.title('Weight Collapse')
plt.plot(range(1, len(eq_norm)+1), classifier_collapse)

plt.vlines(zero_train_error_epoch, 0, 1, colors = "r", linestyles = "dashed", )
plt.savefig('weightcollapse.png')
plt.close()

###  intra variation collpase


plt.figure()
plt.title('Intra Class  Collapse')
plt.plot(range(1, len(eq_norm)+1), variation_collapse)

plt.vlines(zero_train_error_epoch, 0, 1, colors = "r", linestyles = "dashed", )
plt.savefig('variationcollapse.png')
plt.close()


import torch
import torch.nn as nn
import numpy as np
from model import MyNet
from dataset import MyDataset
import copy
import random
import torch.nn.functional as F
import copy

class MixMatch(nn.Module):
    def __init__(self, alpha):
        super(MixMatch, self).__init__()
        self.sampler = Beta(torch.FloatTensor([alpha]), torch.FloatTensor([alpha]))

    def forward(self, x1, y1, x2, y2):
        lamb = self.sampler.sample()
        lamb = torch.max(lamb, 1-lamb).cuda()
        x = x1 * lamb + x2 * (1-lamb)
        y = y1 * lamb+ y2 * (1-lamb)
        return x, y

def skoptimize(P):
    # the minimization objective is min <Q, -logP> + H(Q) 
    lamb = 10
    optimize_iters = 200
    P = F.softmax(P, dim=1)
    N, C = P.size()
    K = torch.pow(P, lamb) 
    u = (torch.ones(N, 1) / N).type_as(P)
    v = (torch.ones(C, 1) / C).type_as(P)
    r = (torch.ones(N, 1) / N).type_as(P)
    c = (torch.ones(C, 1) / C).type_as(P)
    for _ in range(optimize_iters):
        u = r / (K @ v)  
        v = c / (K.transpose(0, 1) @ u)
    u = u.squeeze()
    v = v.squeeze()

    #labels =  torch.diag(u)@K@torch.diag(v)
    labels = u.unsqueeze(1) * K * v.unsqueeze(0)
    #np.save('./after_ot.npy',N * labels.data.cpu().numpy())
    if torch.isnan(labels).sum()!=0:
        raise('Nan value encountered in pseudo labels')
    max_inds = torch.max(labels, dim=1, keepdim=True)[1]
    new_labels = torch.zeros([N, C]).type_as(labels)
    new_labels.scatter_(1, max_inds, 1.0)
    return new_labels, N * labels

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, pred, label):
        pred_log_softmax = self.log_softmax(pred)
        label = F.softmax(label, dim=-1)
        loss = -pred_log_softmax * label
        loss = loss.sum(-1).mean()
        return loss

class HardCrossEntropy(nn.Module):
    def __init__(self):
        super(HardCrossEntropy, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, pred, label):
        label = torch.max(label, dim=-1)[1]
        loss = self.cross_entropy(pred, label)
        return loss

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, pred):
        pred_log_softmax = self.log_softmax(pred)
        label = F.softmax(pred, dim=-1)
        loss = -pred_log_softmax * label
        loss = loss.sum(-1).mean()
        return loss

def get_consistency_weight(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def update_teacher(model, ema_model, update_step):
    # Use the true average until the exponential average is more correct
    alpha = 0.995
    alpha = min(1 - 1 / (update_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(other=param.data,alpha=1 - alpha)

def train(model, teacher_model, labeled_dataloader, unlabeled_dataloader, optimizer, scheduler, epoches):
    criterion = nn.CrossEntropyLoss()
    best_accu = 0
    iters = 0
    total_iter = epoches * len(labeled_dataloader.dataset) // labeled_dataloader.batch_size
    ramp_iter = total_iter 
    print('ramp_iter', ramp_iter)
    for epoch in range(epoches):
        print('training for epoch:{}'.format(epoch))
        model.train()
        teacher_model.train()
        for labeled_data in labeled_dataloader:
            labeled_imgs, labels = labeled_data['img'].cuda(), labeled_data['label'].cuda()
            labeled_imgs, labels = labeled_imgs.squeeze(0), labels.view(-1)
            out = model(labeled_imgs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_teacher(model, teacher_model, iters)
            iters += 1
            if iters % 10 == 0:
                print('epoches:{}, iter:{}, loss:{:.4f}'.format(epoch, iters, loss.item()))
        scheduler.step()
        # torch.save(model.state_dict(), os.path.join('./models', str(epoch+1) + '.pth'))
        accuracy = validate(model, teacher_model, unlabeled_dataloader)
        accuracy, base_accu, accu_batch = validate(model, teacher_model, unlabeled_dataloader)
        print('accuracy for test set in epoch {} is {:.4f}/{:.4f}'.format(epoch, accuracy, base_accu))
        if accuracy > best_accu:
            best_accu = accuracy
            best_accuracy_batch = accu_batch
            np.save('./best_accu_batch.npy', np.array(best_accuracy_batch))
            np.save('./best_accu.npy', np.array([best_accu]))
        epoch += 1


def validate(model, tea_model, dataloader):
    tea_model.eval()
    model.eval()
    images = []
    gt_labels = []
    pred_labels = []
    accuracy = []
    base_correct = 0.0
    sample_num = 0.0
    for _, data in enumerate(dataloader):
        image, label = data['img'].cuda(), data['label'].cuda()
        image, label = image.squeeze(0), label.view(-1)
        out = tea_model(image)
        images.append(image)
        pred_labels.append(out)
        gt_labels.append(label)

        pred = torch.cat(pred_labels, dim=0)
        pred, _ = skoptimize(pred)
        pred = torch.max(pred, dim=1)[1]
        current_model = copy.deepcopy(model).eval()
        # current_accuracy = test_model_accuracy(current_model, images, pred, gt_labels)
        # accuracy.append(current_accuracy)
        
        out = model(image)
        base_pred = torch.max(out, dim=1)[1]
 
        base_correct += torch.sum(base_pred==label).item()
        sample_num += image.size(0)
    current_accuracy = test_model_accuracy(current_model, images, pred, gt_labels)
    accuracy.append(current_accuracy)
    return accuracy[-1], base_correct / sample_num, accuracy


def test_model_accuracy(model, images, pse_labels, gts):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4,  weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    count = 0
    for i in range(10):
        count = 0
        for image in images:
            label = pse_labels[count:count+image.size(0)]
            out = model(image)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += image.size(0)
    model.eval()
    correct = 0.0
    pred = []
    for image in images:
        out = model(image)
        pred.append(out)
    pred = torch.cat(pred, dim=0)    
    pred, _ = skoptimize(pred)
    pred = torch.max(pred, dim=1)[1]
    gts = torch.cat(gts, dim=0)
    correct = torch.sum(pred==gts).item()
    return correct / count

if __name__ == '__main__':
    setup_seed(0)
    class_num = 7
    lr = 1e-3
    epoches = 20
    batch_size = 10
    model = MyNet(class_num)
    teacher_model = copy.deepcopy(model)
    model.cuda()
    teacher_model.cuda()
    labeled_dataset = MyDataset('./data/source',subset='train')
    unlabeled_dataset = MyDataset('./data/target',subset='test')

    labeled_dataloader = torch.utils.data.DataLoader(dataset=labeled_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_dataloader = torch.utils.data.DataLoader(dataset=unlabeled_dataset, batch_size=1, shuffle=True)
    weight = []
    bias = []
    fc_weight = []
    fc_bias = []
    for name, param in model.named_parameters():
        print(name)
        if 'cls' in name:
            if 'weight' in name:
                fc_weight.append(param)
            else:
                fc_bias.append(param)
            continue
        if 'bias' in name or 'bn' in name:
            bias.append(param)
        else:
            weight.append(param)
    params = [
        {'params': weight, 'lr_mult': 1, 'decay_mult': 1},
        {'params': bias, 'lr_mult': 2, 'decay_mult': 0},
        {'params': fc_weight, 'lr_mult': 5, 'decay_mult': 1},
        {'params': fc_bias, 'lr_mult': 10, 'decay_mult': 0},
    ]

    optimizer = torch.optim.SGD(model.parameters(), lr=lr,  weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 20])
    train(model, teacher_model, labeled_dataloader, unlabeled_dataloader, optimizer, scheduler, epoches)



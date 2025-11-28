import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Sequential):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers,
                 norm_layer = lambda dim: nn.LayerNorm(dim, eps=1e-6),
                 act_layer = lambda: nn.LeakyReLU(0.1)):

        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(norm_layer(hidden_dim))
            layers.append(act_layer())
        layers.append(nn.Linear(in_dim if num_layers == 1 else hidden_dim, out_dim))
        super().__init__(*layers)

class MLPFFNNeck(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, hidden_channels=1024):
        super().__init__()
        self.in_dim = in_channels
        self.hidden_dim = hidden_channels
        self.out_dim = out_channels
        #print("in_dim", self.in_dim, "hidden_dim", self.hidden_dim, "out_dim", self.out_dim)

        self.shortcut = nn.Linear(in_channels, out_channels, bias=False)
        self.residual = MLP(in_channels, hidden_channels, out_channels, num_layers=3)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        x = self.shortcut(x) + self.residual(x)
        return x

def reg_ETF(output, label, classifier, mse_loss):
#    cur_M = classifier.cur_M
    target = classifier.cur_M[:, label].T  ## B, d
    loss = mse_loss(output, target)
    return loss

def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate"""
    # if config.cos:
    #     lr_min = 0
    #     lr_max = config
    #     lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / config.num_epochs * 3.1415926535))
    #
    # else:
    epoch = epoch + 1
    if epoch <= 100:
        lr = config
    elif epoch > 180:
        lr = config * 0.01
    elif epoch > 160:
        lr = config * 0.1
    # elif epoch > 220:
    #     lr = config * 0.001
    else:
        lr = config

    for param_group in optimizer.param_groups:
        print('lr now:', lr)
        print("epoch:", epoch)
        param_group['lr'] = lr

def dot_loss(output, label, cur_M, classifier, criterion, H_length, reg_lam=0):
    target = cur_M[:, label].T ## B, d  output: B, d
    if criterion == 'dot_loss':
        loss = - torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1).mean()
    elif criterion == 'reg_dot_loss':
        dot = torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1) #+ classifier.module.bias[label].view(-1)

        with torch.no_grad():
            M_length = torch.sqrt(torch.sum(target ** 2, dim=1, keepdims=False))
        loss = (1/2) * torch.mean(((dot-(M_length * H_length)) ** 2) / H_length)

        if reg_lam > 0:
            reg_Eh_l2 = torch.mean(torch.sqrt(torch.sum(output ** 2, dim=1, keepdims=True)))
            loss = loss + reg_Eh_l2*reg_lam
    else:
        # Default case for other criteria like cross_entropy
        logits = torch.matmul(output, cur_M)
        loss = F.cross_entropy(logits, label)

    return loss

def produce_Ew(label, num_classes):
    uni_label, count = torch.unique(label, return_counts=True)
    batch_size = label.size(0)
    uni_label_num = uni_label.size(0)
    assert batch_size == torch.sum(count)
    gamma = batch_size / uni_label_num
    Ew = torch.ones(1, num_classes).cuda(label.device)
    for i in range(uni_label_num):
        label_id = uni_label[i]
        label_count = count[i]
        length = torch.sqrt(gamma / label_count)
#        length = (gamma / label_count)
        #length = torch.sqrt(label_count / gamma)
        Ew[0, label_id] = length
    return Ew

def produce_global_Ew(cls_num_list):
    num_classes = len(cls_num_list)
    cls_num_list = torch.tensor(cls_num_list).cuda()
    total_num = torch.sum(cls_num_list)
    gamma = total_num / num_classes
    Ew = torch.sqrt(gamma / cls_num_list)
    Ew = Ew.unsqueeze(0)
    return Ew

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam          #mixed:打乱后的数据   y_a:打乱前数据的标签   y_b:打乱后数据的标签

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def validate(self, val_loader, model, classifier, criterion, logger, feat_num_features, neck):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Eval: ')
    # switch to evaluate mode
    model.eval()
    neck.eval()

    classifier.eval()
    class_num = torch.zeros(self.args["num_classes"]).cuda()
    correct = torch.zeros(self.args["num_classes"]).cuda()

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])


    with torch.no_grad():

        for i, (_,images, target) in enumerate(val_loader):

            target = target.to(torch.long)
            images, target = images.to(self._device), target.to(self._device)

            if self.args["ETF_classifier"]:  # and config.reg_dot_loss and config.GivenEw:
                if self.args["dataset"] == 'imagenet':
                    cur_M = classifier.module.ori_M
                else:
                    cur_M = classifier.ori_M

            # compute output
            feat ,__, features= model(images)
            if self.args["ETF_classifier"]:
                neck = neck.to(device=0)
                feat = classifier(neck(feat))
                output = torch.matmul(feat, cur_M)
                if self.args["reg_dot_loss"]:
                    with torch.no_grad():
                        feat_nograd = feat.detach()
                        H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)), 1e-8)
                    loss = dot_loss(feat, target, cur_M, classifier, criterion, H_length)
                else:
                    loss = criterion(output, target)
            else:
                output = classifier(feat)
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, predicted = output.max(1)
            target_one_hot = F.one_hot(target, self.args["num_classes"])
            predict_one_hot = F.one_hot(predicted, self.args["num_classes"])
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            true_class = np.append(true_class, target.cpu().numpy())

            if i % 10 == 0:
                progress.display(i, logger)
        acc_classes = correct / class_num
        head_acc = acc_classes[self.args["head_class_idx"][0]:self.args["head_class_idx"][1]].mean() * 100

        med_acc = acc_classes[self.args["med_class_idx"][0]:self.args["med_class_idx"][1]].mean() * 100
        tail_acc = acc_classes[self.args["tail_class_idx"][0]:self.args["tail_class_idx"][1]].mean() * 100
        logger.info(
            '* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'.format(
                top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))

        # cal = calibration(true_class, pred_class, confidence, num_bins=15)
        # logger.info('* ECE   {ece:.3f}%.'.format(ece=cal['expected_calibration_error'] * 100))

    return top1.avg
    # return top1.avg, cal['expected_calibration_error'] * 100


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'






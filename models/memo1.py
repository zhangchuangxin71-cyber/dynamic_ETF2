import logging
import math

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import copy
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import AdaptiveNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

from models import base

from methods import dot_loss, reg_ETF, produce_Ew, produce_global_Ew, mixup_data, mixup_criterion, accuracy, \
    AverageMeter, ProgressMeter, validate, adjust_learning_rate, MLPFFNNeck

num_workers = 8


class MEMO(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._old_base = None
        self._network = AdaptiveNet(args['convnet_type'], False)
        logging.info(
            f'>>> train generalized blocks:{self.args["train_base"]} train_adaptive:{self.args["train_adaptive"]}')

    def after_task(self):
        self._known_classes = self._total_classes
        if self._cur_task == 0:
            if self.args['train_base']:
                logging.info("Train Generalized Blocks...")
                self._network.TaskAgnosticExtractor.train()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = True
            else:
                logging.info("Fix Generalized Blocks...")
                self._network.TaskAgnosticExtractor.eval()  # 这里的TaskAgnosticExtractor(任务无关提取器)指的是basenet
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = False  # 参数更新设置为False   即冻结参数。

        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.AdaptiveExtractors[i].parameters():
                    if self.args['train_adaptive']:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False  # 将上一个任务的模型骨干的高层参数冻结
            self.args["num_classes"] += self.args["increment"]

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source='train',
            mode='train',
            appendent=self._get_memory()
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=num_workers
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source='test',
            mode='test'
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)  # 这里可qu
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def set_network(self):
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.train()  # All status from eval to train
        if self.args['train_base']:
            self._network.TaskAgnosticExtractor.train()
        else:
            self._network.TaskAgnosticExtractor.eval()

        # set adaptive extractor's status
        self._network.AdaptiveExtractors[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):  # 旧模型的高层设置为.eval()
                if self.args['train_adaptive']:
                    self._network.AdaptiveExtractors[i].train()
                else:
                    self._network.AdaptiveExtractors[i].eval()
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        # model_params = [p for p in self._network.parameters() if p.requires_grad]

        # criterion = nn.CrossEntropyLoss().to(self._device)
        # if self.args["reg_dot_loss"]:
        #     criterion = self.args["criterion"]
        #     print('----  Dot-Regression Loss is adopted ----')
        if self.args["dataset"] == 'cifar100':
            feat_num_features = 128
            feat_len = feat_num_features * 2
            if self.args["ETF_classifier"]:
                print('########   Using a ETF as the last linear classifier    ##########')
                classifier = getattr(base, 'ETF_Classifier')(feat_in=feat_len, num_classes=self.args["num_classes"])
            else:
                classifier = getattr(base, 'Classifier')(feat_in=feat_len, num_classes=self.args["num_classes"])
            neck = MLPFFNNeck(in_channels=feat_num_features,
                              out_channels=feat_len,
                              hidden_channels=feat_num_features * 4)  # 128 x 2048 x 1024
            neck.to(device=0)
            classifier = classifier.to(self._device)
        elif self.args["dataset"] == 'imagenet100':
            feat_num_features = 512
            feat_len = feat_num_features * 16
            if self.args["ETF_classifier"]:
                print('########   Using a ETF as the last linear classifier    ##########')
                classifier = getattr(base, 'ETF_Classifier')(feat_in=feat_len, num_classes=self.args["num_classes"])
            else:
                classifier = getattr(base, 'Classifier')(feat_in=feat_len, num_classes=self.args["num_classes"])
            neck = MLPFFNNeck(in_channels=feat_num_features,
                              out_channels=feat_len,
                              hidden_channels=feat_num_features * 16)  # 128 x 2048 x 1024
            neck.to(device=0)
            classifier = classifier.to(self._device)
        elif self.args["dataset"] == 'cub200':
            if self.args["ETF_classifier"]:
                print('########   Using a ETF as the last linear classifier    ##########')
                classifier = getattr(base, 'ETF_Classifier')(feat_in=4096, num_classes=self.args["num_classes"])
            else:
                classifier = getattr(base, 'Classifier')(feat_in=4096, num_classes=self.args["num_classes"])

        # neck = MLPFFNNeck(in_channels=feat_num_features,
        #                   out_channels=feat_num_features * 2,
        #                   hidden_channels=feat_num_features * 4)  # 128 x 2048 x 1024
        # neck.to(device=0)
        # classifier = classifier.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                [{"params": filter(lambda p: p.requires_grad, self._network.parameters())},
                 {"params": classifier.parameters()}, {"params": neck.parameters()}],
                # [{"params": filter(lambda p: p.requires_grad, self._network.parameters())},{"params": classifier.parameters()}]
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"]
            )
            # optimizer = optim.Adam(
            #     [{"params": filter(lambda p: p.requires_grad, self._network.parameters())},
            #      {"params": classifier.parameters()}, {"params": neck.parameters()}],
            #     # [{"params": filter(lambda p: p.requires_grad, self._network.parameters())},{"params": classifier.parameters()}]
            #     #momentum=0.9,
            #     lr=self.args["init_lr"],
            #     #weight_decay=self.args["init_weight_decay"]
            # )
            if self.args['scheduler'] == 'steplr':
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer,
                    milestones=self.args['init_milestones'],
                    gamma=self.args['init_lr_decay']
                )
            elif self.args['scheduler'] == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args['init_epoch']
                )
            else:
                raise NotImplementedError

            if not self.args['skip']:
                self._init_train(train_loader, test_loader, optimizer, scheduler, classifier, logger, feat_num_features,
                                 neck)
            else:
                if isinstance(self._network, nn.DataParallel):
                    self._network = self._network.module
                load_acc = self._network.load_checkpoint(self.args)
                self._network.to(self._device)

                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)

                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
        else:
            optimizer = optim.SGD(
                [{"params": filter(lambda p: p.requires_grad, self._network.parameters())},
                 {"params": classifier.parameters()},
                 {"params": neck.parameters()}],
                lr=self.args['lrate'],
                momentum=0.9,
                weight_decay=self.args['weight_decay']
            )
            if self.args['scheduler'] == 'steplr':
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer,
                    milestones=self.args['milestones'],
                    gamma=self.args['lrate_decay']
                )
            elif self.args['scheduler'] == 'cosine':
                assert self.args['t_max'] is not None
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args['t_max']
                )
            else:
                raise NotImplementedError
            self._update_representation(train_loader, test_loader, optimizer, scheduler, classifier, logger,
                                        feat_num_features, neck)  # 更新新concat到模型的representation
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes - self._known_classes)
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, classifier, logger, feat_num_features, neck):
        # global best_acc1, its_ece
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.3f')
        top1 = AverageMeter('Acc@1', ':6.3f')
        top5 = AverageMeter('Acc@5', ':6.3f')

        confidence = np.array([])
        pred_class = np.array([])
        true_class = np.array([])
        init_train = []

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().to(self._device)

        if self.args["reg_dot_loss"]:
            criterion = self.args["criterion"]
            print('----  Dot-Regression Loss is adopted ----')
        # if self.args["dataset"] == 'cifar100':
        #     feat_len = 128
        #     if self.args["ETF_classifier"]:
        #         print('########   Using a ETF as the last linear classifier    ##########')
        #         classifier = getattr(base, 'ETF_Classifier')(feat_in=feat_len, num_classes=self.args["num_classes"])
        #     else:
        #         classifier = getattr(base, 'Classifier')(feat_in=feat_len, num_classes=self.args["num_classes"])
        # elif self.args["dataset"] == 'imagenet':
        #
        #     if self.args["ETF_classifier"]:
        #         print('########   Using a ETF as the last linear classifier    ##########')
        #         classifier = getattr(base, 'ETF_Classifier')(feat_in=4096, num_classes=self.args["num_classes"])
        #     else:
        #         classifier = getattr(base, 'Classifier')(feat_in=4096, num_classes=self.args["num_classes"])
        # elif self.args["dataset"] == 'cub200':
        #     if self.args["ETF_classifier"]:
        #         print('########   Using a ETF as the last linear classifier    ##########')
        #         classifier = getattr(base, 'ETF_Classifier')(feat_in=4096, num_classes=self.args["num_classes"])
        #     else:
        #         classifier = getattr(base, 'Classifier')(feat_in=4096, num_classes=self.args["num_classes"])

        # classifier = classifier.to(self._device)
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):

            # losses = 0.
            correct, total = 0, 0
            progress = ProgressMeter(
                len(train_loader),
                [batch_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))
            adjust_learning_rate(optimizer, epoch, self.args["init_lr"])
            for i, (_, inputs, targets) in enumerate(train_loader):
                self._network.train()
                classifier.train()
                neck.train()

                targets = targets.to(torch.long)
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits, base_feature_map, __ = self._network(inputs)  # 自己加的
                # logits = self._network(inputs)['logits']
                # 对于修改self._network的AdaptiveExtractors模块（高层块）的特征，要注意除了满足ETF分类器的需要外，还要满足后续的高层块堆叠。
                feat = logits  # 这里的对于要使用ETF分类器来说，feat的形状应该是128*128     第一个128为一个batch_size的样本数，第二个128为每个样本的维度。
                if self.args["ETF_classifier"]:  # 是否使用ETF分类器
                    if self.args["reg_dot_loss"] and self.args["GivenEw"]:
                        learned_norm = produce_Ew(targets, self.args["num_classes"])  # Ew为分类器向量的2范数约束
                        if self.args["dataset"] == 'imagenet':
                            cur_M = learned_norm * classifier.module.ori_M
                        else:
                            cur_M = learned_norm * classifier.ori_M
                    else:
                        if self.args["dataset"] == 'imagenet':
                            cur_M = classifier.module.ori_M
                        else:
                            cur_M = classifier.ori_M
                else:
                    out = self.fc(feat)  # {logits: self.fc(features)}

                    aux_logits = self.aux_fc(feat[:, -self.out_dim:])["logits"]

                    out.update({"aux_logits": aux_logits, "features": feat})
                    out.update({"base_features": base_feature_map})
                if self.args["mixup"] is True:  # 是否使用mixup数据增强方法。
                    images, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=self.args["alpha"])
                    ###      images:打乱后的图像数据      targets_a:打乱前数据的标签      targets_b:打乱后数据的标签

                    if self.args["ETF_classifier"]:
                        feat = classifier(feat)  # 这里mode的输出特征维度和classefier的接收特征维度要注意保持一致。
                        output = torch.matmul(feat, cur_M)  # + classifier.module.bias
                        if self.args["reg_dot_loss"]:  ## ETF classifier + DR loss
                            with torch.no_grad():
                                feat_nograd = feat.detach()
                                H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)),
                                                       1e-8)
                            loss_a = dot_loss(feat, targets_a, cur_M, classifier, criterion, H_length,
                                              reg_lam=self.args["reg_lam"])
                            loss_b = dot_loss(feat, targets_b, cur_M, classifier, criterion, H_length,
                                              reg_lam=self.args["reg_lam"])
                            loss = lam * loss_a + (1 - lam) * loss_b
                        else:  ## ETF classifier + CE loss
                            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)

                    else:  ## learnable classifier + CE loss
                        output = classifier(feat)
                        loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
                else:
                    if self.args["ETF_classifier"]:
                        feat = classifier(neck(feat))
                        output = torch.matmul(feat, cur_M)
                        if self.args["reg_dot_loss"]:  ## ETF classifier + DR loss
                            with torch.no_grad():
                                feat_nograd = feat.detach()
                                H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)),
                                                       1e-8)
                            loss = dot_loss(feat, targets, cur_M, classifier, criterion, H_length,
                                            reg_lam=self.args["reg_lam"])
                        else:  ## ETF classifier + CE loss
                            loss = criterion(output, targets)
                    else:  ## learnable classifier + CE loss
                        output = classifier(feat)
                        loss = criterion(output, targets)
                        # 当使用ETF时，下一行的loss要注释掉
                # loss=F.cross_entropy(logits,targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # losses += loss.item()

                acc1, acc5 = accuracy(output, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))
                if i % 10 == 0:
                    progress.display(i, logger)
                    # progress.display(i, logging)
                    # train_acc = top1.avg
                    # print('train_acc:{}'.format(train_acc))
                    # top1.reset()
                    # top5.reset()

            # 找到模型对样本的预测值，以计算训练精确度。

            # _, preds = output.max(1)
            # target_one_hot = F.one_hot(targets, self.args["num_classes"])
            # predict_one_hot = F.one_hot(preds, self.args["num_classes"])
            # class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            # correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)
            #
            # prob = torch.softmax(output, dim=1)
            # confidence_part, pred_class_part = torch.max(prob, dim=1)
            # confidence = np.append(confidence, confidence_part.cpu().numpy())
            # pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            # true_class = np.append(true_class, targets.cpu().numpy())

            # # measure elapsed time
            # # batch_time.update(time.time() - end)
            # # end = time.time()
            # progress.display(i, logging)
            #
            # #_, preds = torch.max(logits, dim=1)
            # correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            # total += len(targets)

            if epoch % 5 == 0 or epoch == 49:
                test_acc = validate(self, test_loader, self._network, classifier, criterion, logging, feat_num_features,
                                    neck)
                print('################################################test_acc:{}'.format(test_acc))
                init_train.append(test_acc.item())
                print("init_train:", init_train)
                print("max_acc:", max(init_train))
                logging.info("test_acc:{}".format(test_acc))
                logging.info("init_train:{}".format(init_train))
                logging.info("max_acc:{}".format(max((init_train))))

            scheduler.step()
            # train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            # print('train_acc:{}'.format(train_acc))
            # if epoch%5==0 or epoch ==49:
            #     test_acc = self._compute_accuracy(self._network, test_loader)
            #     info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
            #     self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc, test_acc)
            # else:
            #     info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
            #     self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc)
            # # prog_bar.set_description(info)
            # logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler, classifier, logger,
                               feat_num_features, neck):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.3f')
        top1 = AverageMeter('Acc@1', ':6.3f')
        top5 = AverageMeter('Acc@5', ':6.3f')

        confidence = np.array([])
        pred_class = np.array([])
        true_class = np.array([])
        incremental_train = []

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().to(self._device)

        if self.args["reg_dot_loss"]:
            criterion = self.args["criterion"]
            print('----  Dot-Regression Loss is adopted ----')
        # if self.args["dataset"] == 'cifar100':
        #     feat_len = 128
        #     if self.args["ETF_classifier"]:
        #         print('########   Using a ETF as the last linear classifier    ##########')
        #         classifier = getattr(base, 'ETF_Classifier')(feat_in=feat_len, num_classes=self.args["num_classes"])
        #     else:
        #         classifier = getattr(base, 'Classifier')(feat_in=feat_len, num_classes=self.args["num_classes"])
        # elif self.args["dataset"] == 'imagenet':
        #
        #     if self.args["ETF_classifier"]:
        #         print('########   Using a ETF as the last linear classifier    ##########')
        #         classifier = getattr(base, 'ETF_Classifier')(feat_in=4096, num_classes=self.args["num_classes"])
        #     else:
        #         classifier = getattr(base, 'Classifier')(feat_in=4096, num_classes=self.args["num_classes"])
        # elif self.args["dataset"] == 'cub200':
        #     if self.args["ETF_classifier"]:
        #         print('########   Using a ETF as the last linear classifier    ##########')
        #         classifier = getattr(base, 'ETF_Classifier')(feat_in=4096, num_classes=self.args["num_classes"])
        #     else:
        #         classifier = getattr(base, 'Classifier')(feat_in=4096, num_classes=self.args["num_classes"])
        #
        # classifier = classifier.to(self._device)

        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            progress = ProgressMeter(
                len(train_loader),
                [batch_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))
            # adjust_learning_rate(optimizer, epoch, self.args["init_lr"])
            self.set_network()  # 这里设置模型的哪些块可用于参数更新，哪些块只用于模型验证。
            # losses = 0.
            losses_clf = 0.
            losses_aux = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                self._network.train()
                neck.train()
                classifier.train()
                targets = targets.to(torch.long)
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits, base_feature_map, features = self._network(inputs)  # 自己加的
                # logits = self._network(inputs)['logits']
                # 对于修改self._network的AdaptiveExtractors模块（高层块）的特征，要注意除了满足ETF分类器的需要外，还要满足后续的高层块堆叠。
                feat = logits  # 这里的对于要使用ETF分类器来说，feat的形状应该是128*128     第一个128为一个batch_size的样本数，第二个128为每个样本的维度。
                if self.args["ETF_classifier"]:  # 是否使用ETF分类器
                    if self.args["reg_dot_loss"] and self.args["GivenEw"]:
                        learned_norm = produce_Ew(targets, self.args["num_classes"])  # Ew为分类器向量的2范数约束
                        if self.args["dataset"] == 'imagenet':
                            cur_M = learned_norm * classifier.module.ori_M
                        else:
                            cur_M = learned_norm * classifier.ori_M
                    else:
                        if self.args["dataset"] == 'imagenet':
                            cur_M = classifier.module.ori_M
                        else:
                            cur_M = classifier.ori_M
                else:
                    out = self.fc(feat)  # {logits: self.fc(features)}

                    aux_logits = self.aux_fc(feat[:, -self.out_dim:])["logits"]

                    out.update({"aux_logits": aux_logits, "features": feat})
                    out.update({"base_features": base_feature_map})
                if self.args["mixup"] is True:  # 是否使用mixup数据增强方法。
                    images, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=self.args["alpha"])
                    ###      images:打乱后的图像数据      targets_a:打乱前数据的标签      targets_b:打乱后数据的标签

                    if self.args["ETF_classifier"]:

                        feat = classifier(neck(feat))
                        # feat = classifier(feat)  # 这里mode的输出特征维度和classefier的接收特征维度要注意保持一致。
                        output = torch.matmul(feat, cur_M)  # + classifier.module.bias
                        if self.args["reg_dot_loss"]:  ## ETF classifier + DR loss
                            with torch.no_grad():
                                feat_nograd = feat.detach()
                                H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)),
                                                       1e-8)
                            loss_a = dot_loss(feat, targets_a, cur_M, classifier, criterion, H_length,
                                              reg_lam=self.args["reg_lam"])
                            loss_b = dot_loss(feat, targets_b, cur_M, classifier, criterion, H_length,
                                              reg_lam=self.args["reg_lam"])
                            loss = lam * loss_a + (1 - lam) * loss_b
                        else:  ## ETF classifier + CE loss
                            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)

                    else:  ## learnable classifier + CE loss
                        output = classifier(feat)
                        loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
                else:
                    if self.args["ETF_classifier"]:

                        feat = classifier(neck(feat))
                        # feat = classifier(feat)
                        output = torch.matmul(feat, cur_M)
                        if self.args["reg_dot_loss"]:  ## ETF classifier + DR loss
                            with torch.no_grad():
                                feat_nograd = feat.detach()
                                H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)),
                                                       1e-8)
                            loss = dot_loss(feat, targets, cur_M, classifier, criterion, H_length,
                                            reg_lam=self.args["reg_lam"])
                        else:  ## ETF classifier + CE loss
                            loss = criterion(output, targets)
                    else:  ## learnable classifier + CE loss
                        output = classifier(feat)
                        loss = criterion(output, targets)
                        # 当使用ETF时，下一行的loss要注释掉
                # loss=F.cross_entropy(logits,targets)

                loss_d = 0.5 * torch.pow((F.cosine_similarity(features[-2], features[-1], dim=1) - 1.0), 2)
                if epoch > 100:
                    loss_all = loss
                else:
                    loss_all = loss + loss_d.sum() * self.args["distill_weight"]  # + loss_d.sum() * 0.25

                optimizer.zero_grad()
                loss_all.backward()
                optimizer.step()
                # losses += loss.item()

                acc1, acc5 = accuracy(output, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))
                if i % 40 == 0:
                    progress.display(i, logger)
                    # progress.display(i, logging)
                    # train_acc = top1.avg
                    # print('train_acc:{}'.format(train_acc))
                    # top1.reset()
                    # top5.reset()

                # 找到模型对样本的预测值，以计算训练精确度。

                # _, preds = output.max(1)
                # target_one_hot = F.one_hot(targets, self.args["num_classes"])
                # predict_one_hot = F.one_hot(preds, self.args["num_classes"])
                # class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
                # correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)
                #
                # prob = torch.softmax(output, dim=1)
                # confidence_part, pred_class_part = torch.max(prob, dim=1)
                # confidence = np.append(confidence, confidence_part.cpu().numpy())
                # pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
                # true_class = np.append(true_class, targets.cpu().numpy())

                # # measure elapsed time
                # # batch_time.update(time.time() - end)
                # # end = time.time()
                # progress.display(i, logging)
                #
                # #_, preds = torch.max(logits, dim=1)
                # correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                # total += len(targets)

            if epoch % 5 == 0 or epoch == 49:
                test_acc = validate(self, test_loader, self._network, classifier, criterion, logging, feat_num_features,
                                    neck)
                print('##########################################test_acc:{}'.format(test_acc))
                incremental_train.append(test_acc.item())
                print("incremental_train:", incremental_train)
                print("max_acc:", max(incremental_train))
                logging.info("test_acc:{}".format(test_acc))
                logging.info("incremental_train:{}".format(incremental_train))
                logging.info("max_acc:{}".format(max((incremental_train))))
            scheduler.step()

    # def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
    #     prog_bar = tqdm(range(self.args["epochs"]))
    #     for _, epoch in enumerate(prog_bar):
    #         self.set_network()                          #这里设置模型的哪些块可用于参数更新，哪些块只用于模型验证。
    #         losses = 0.
    #         losses_clf=0.
    #         losses_aux=0.
    #         correct, total = 0, 0
    #         for i, (_, inputs, targets) in enumerate(train_loader):
    #             targets = targets.to(torch.long)
    #             inputs, targets = inputs.to(self._device), targets.to(self._device)
    #
    #             outputs,__= self._network(inputs)
    #             logits,aux_logits=outputs["logits"],outputs["aux_logits"]
    #             loss_clf=F.cross_entropy(logits,targets)
    #             aux_targets = targets.clone()
    #             aux_targets=torch.where(aux_targets-self._known_classes+1>0,  aux_targets-self._known_classes+1,0)
    #             loss_aux=F.cross_entropy(aux_logits,aux_targets)
    #             loss=loss_clf+self.args['alpha_aux']*loss_aux
    #
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             losses += loss.item()
    #             losses_aux+=loss_aux.item()
    #             losses_clf+=loss_clf.item()
    #
    #             _, preds = torch.max(logits, dim=1)
    #             correct += preds.eq(targets.expand_as(preds)).cpu().sum()
    #             total += len(targets)
    #
    #         scheduler.step()
    #         train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
    #         if epoch%5==0:
    #             test_acc = self._compute_accuracy(self._network, test_loader)
    #             info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux  {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
    #             self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader),losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc, test_acc)
    #         else:
    #             info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}'.format(
    #             self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader), losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc)
    #         prog_bar.set_description(info)
    #     logging.info(info)


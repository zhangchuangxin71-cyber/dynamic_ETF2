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
                self._network.TaskAgnosticExtractor.eval()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = False

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
                        p.requires_grad = False
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
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def set_network(self):
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.train()
        if self.args['train_base']:
            self._network.TaskAgnosticExtractor.train()
        else:
            self._network.TaskAgnosticExtractor.eval()

        self._network.AdaptiveExtractors[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
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
                              hidden_channels=feat_num_features * 4)
            neck.to(self._device)
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
                              hidden_channels=feat_num_features * 16)
            neck.to(self._device)
            classifier = classifier.to(self._device)
        elif self.args["dataset"] == 'cub200':
            feat_num_features = 512
            feat_len = 4096
            if self.args["ETF_classifier"]:
                print('########   Using a ETF as the last linear classifier    ##########')
                classifier = getattr(base, 'ETF_Classifier')(feat_in=feat_len, num_classes=self.args["num_classes"])
            else:
                classifier = getattr(base, 'Classifier')(feat_in=feat_len, num_classes=self.args["num_classes"])
            neck = MLPFFNNeck(in_channels=feat_num_features,
                              out_channels=feat_len,
                              hidden_channels=feat_num_features * 8)
            neck.to(self._device)
            classifier = classifier.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                [{"params": filter(lambda p: p.requires_grad, self._network.parameters())},
                 {"params": classifier.parameters()}, {"params": neck.parameters()}],
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"]
            )
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
                                        feat_num_features, neck)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes - self._known_classes)
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _validate_custom(self, test_loader, network, classifier, criterion, neck):
        """自定义验证函数，确保使用neck"""
        network.eval()
        classifier.eval()
        neck.eval()

        top1 = AverageMeter('Acc@1', ':6.3f')

        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(test_loader):
                targets = targets.to(torch.long)
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                logits, _, __ = network(inputs)
                feat = neck(logits)

                if self.args["ETF_classifier"]:
                    feat = classifier(feat)
                    if self.args["dataset"] == 'imagenet':
                        cur_M = classifier.module.ori_M
                    else:
                        cur_M = classifier.ori_M
                    output = torch.matmul(feat, cur_M)
                else:
                    output = classifier(feat)

                acc1, _ = accuracy(output, targets, topk=(1, 5))
                top1.update(acc1[0], inputs.size(0))

        return top1.avg

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, classifier, logger, feat_num_features, neck):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.3f')
        top1 = AverageMeter('Acc@1', ':6.3f')
        top5 = AverageMeter('Acc@5', ':6.3f')

        init_train = []

        criterion = nn.CrossEntropyLoss().to(self._device)

        if self.args["reg_dot_loss"]:
            criterion = self.args["criterion"]
            print('----  Dot-Regression Loss is adopted ----')

        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):

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

                logits, base_feature_map, __ = self._network(inputs)

                # 添加梯度裁剪和NaN检查
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logging.warning(f"NaN or Inf detected in logits at epoch {epoch}, batch {i}")
                    continue

                feat = neck(logits)

                # 检查neck输出
                if torch.isnan(feat).any() or torch.isinf(feat).any():
                    logging.warning(f"NaN or Inf detected in neck output at epoch {epoch}, batch {i}")
                    continue

                if self.args["ETF_classifier"]:
                    if self.args["reg_dot_loss"] and self.args["GivenEw"]:
                        learned_norm = produce_Ew(targets, self.args["num_classes"])
                        if self.args["dataset"] == 'imagenet':
                            cur_M = learned_norm * classifier.module.ori_M
                        else:
                            cur_M = learned_norm * classifier.ori_M
                    else:
                        if self.args["dataset"] == 'imagenet':
                            cur_M = classifier.module.ori_M
                        else:
                            cur_M = classifier.ori_M

                if self.args["mixup"] is True:
                    images, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=self.args["alpha"])

                    if self.args["ETF_classifier"]:
                        feat = classifier(feat)
                        output = torch.matmul(feat, cur_M)
                        if self.args["reg_dot_loss"]:
                            with torch.no_grad():
                                feat_nograd = feat.detach()
                                H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)),
                                                       1e-8)
                            loss_a = dot_loss(feat, targets_a, cur_M, classifier, criterion, H_length,
                                              reg_lam=self.args["reg_lam"])
                            loss_b = dot_loss(feat, targets_b, cur_M, classifier, criterion, H_length,
                                              reg_lam=self.args["reg_lam"])
                            loss = lam * loss_a + (1 - lam) * loss_b
                        else:
                            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
                    else:
                        output = classifier(feat)
                        loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
                else:
                    if self.args["ETF_classifier"]:
                        feat = classifier(feat)
                        output = torch.matmul(feat, cur_M)
                        if self.args["reg_dot_loss"]:
                            with torch.no_grad():
                                feat_nograd = feat.detach()
                                H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)),
                                                       1e-8)
                            loss = dot_loss(feat, targets, cur_M, classifier, criterion, H_length,
                                            reg_lam=self.args["reg_lam"])
                        else:
                            loss = criterion(output, targets)
                    else:
                        output = classifier(feat)
                        loss = criterion(output, targets)

                # 检查loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"NaN or Inf loss detected at epoch {epoch}, batch {i}, skipping batch")
                    continue

                optimizer.zero_grad()
                loss.backward()

                # 添加梯度裁剪
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(neck.parameters(), max_norm=1.0)

                optimizer.step()

                acc1, acc5 = accuracy(output, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))
                if i % 10 == 0:
                    progress.display(i, logger)

            if epoch % 5 == 0 or epoch == (self.args["init_epoch"] - 1):
                test_acc = self._validate_custom(test_loader, self._network, classifier, criterion, neck)
                print('################################################test_acc:{}'.format(test_acc))
                init_train.append(test_acc.item())
                print("init_train:", init_train)
                print("max_acc:", max(init_train))
                logging.info("test_acc:{}".format(test_acc))
                logging.info("init_train:{}".format(init_train))
                logging.info("max_acc:{}".format(max((init_train))))

            scheduler.step()

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler, classifier, logger,
                               feat_num_features, neck):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.3f')
        top1 = AverageMeter('Acc@1', ':6.3f')
        top5 = AverageMeter('Acc@5', ':6.3f')

        incremental_train = []

        criterion = nn.CrossEntropyLoss().to(self._device)

        if self.args["reg_dot_loss"]:
            criterion = self.args["criterion"]
            print('----  Dot-Regression Loss is adopted ----')

        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            progress = ProgressMeter(
                len(train_loader),
                [batch_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))
            self.set_network()

            for i, (_, inputs, targets) in enumerate(train_loader):
                self._network.train()
                neck.train()
                classifier.train()
                targets = targets.to(torch.long)
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits, base_feature_map, features = self._network(inputs)

                # 添加NaN检查
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logging.warning(f"NaN or Inf detected in logits at epoch {epoch}, batch {i}")
                    continue

                feat = neck(logits)

                if torch.isnan(feat).any() or torch.isinf(feat).any():
                    logging.warning(f"NaN or Inf detected in neck output at epoch {epoch}, batch {i}")
                    continue

                if self.args["ETF_classifier"]:
                    if self.args["reg_dot_loss"] and self.args["GivenEw"]:
                        learned_norm = produce_Ew(targets, self.args["num_classes"])
                        if self.args["dataset"] == 'imagenet':
                            cur_M = learned_norm * classifier.module.ori_M
                        else:
                            cur_M = learned_norm * classifier.ori_M
                    else:
                        if self.args["dataset"] == 'imagenet':
                            cur_M = classifier.module.ori_M
                        else:
                            cur_M = classifier.ori_M

                if self.args["mixup"] is True:
                    images, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=self.args["alpha"])

                    if self.args["ETF_classifier"]:
                        feat = classifier(feat)
                        output = torch.matmul(feat, cur_M)
                        if self.args["reg_dot_loss"]:
                            with torch.no_grad():
                                feat_nograd = feat.detach()
                                H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)),
                                                       1e-8)
                            loss_a = dot_loss(feat, targets_a, cur_M, classifier, criterion, H_length,
                                              reg_lam=self.args["reg_lam"])
                            loss_b = dot_loss(feat, targets_b, cur_M, classifier, criterion, H_length,
                                              reg_lam=self.args["reg_lam"])
                            loss = lam * loss_a + (1 - lam) * loss_b
                        else:
                            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
                    else:
                        output = classifier(feat)
                        loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
                else:
                    if self.args["ETF_classifier"]:
                        feat = classifier(feat)
                        output = torch.matmul(feat, cur_M)
                        if self.args["reg_dot_loss"]:
                            with torch.no_grad():
                                feat_nograd = feat.detach()
                                H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)),
                                                       1e-8)
                            loss = dot_loss(feat, targets, cur_M, classifier, criterion, H_length,
                                            reg_lam=self.args["reg_lam"])
                        else:
                            loss = criterion(output, targets)
                    else:
                        output = classifier(feat)
                        loss = criterion(output, targets)

                loss_d = 0.5 * torch.pow((F.cosine_similarity(features[-2], features[-1], dim=1) - 1.0), 2)
                if epoch > 100:
                    loss_all = loss
                else:
                    loss_all = loss + loss_d.sum() * self.args["distill_weight"]

                # 检查loss
                if torch.isnan(loss_all) or torch.isinf(loss_all):
                    logging.warning(f"NaN or Inf loss detected at epoch {epoch}, batch {i}, skipping batch")
                    continue

                optimizer.zero_grad()
                loss_all.backward()

                # 添加梯度裁剪
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(neck.parameters(), max_norm=1.0)

                optimizer.step()

                acc1, acc5 = accuracy(output, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))
                if i % 40 == 0:
                    progress.display(i, logger)

            if epoch % 5 == 0 or epoch == (self.args["epochs"] - 1):
                test_acc = self._validate_custom(test_loader, self._network, classifier, criterion, neck)
                print('##########################################test_acc:{}'.format(test_acc))
                incremental_train.append(test_acc.item())
                print("incremental_train:", incremental_train)
                print("max_acc:", max(incremental_train))
                logging.info("test_acc:{}".format(test_acc))
                logging.info("incremental_train:{}".format(incremental_train))
                logging.info("max_acc:{}".format(max((incremental_train))))
            scheduler.step()
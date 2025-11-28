import logging
from methods import dot_loss,reg_ETF,produce_Ew,produce_global_Ew,mixup_data,mixup_criterion,accuracy,AverageMeter,ProgressMeter

batch_time = AverageMeter('Time', ':6.3f')
data_time = AverageMeter('Data', ':6.3f')
losses = AverageMeter('Loss', ':.3f')
top1 = AverageMeter('Acc@1', ':6.3f')
top5 = AverageMeter('Acc@5', ':6.3f')
progress = ProgressMeter(
    55,
    [batch_time, losses, top1, top5],
    prefix="Epoch: [{}]".format(50))



acc1=3.3
acc5=3.4
top1.update(acc1[0], images.size(0))
top5.update(acc5[0], images.size(0))

i=3
progress.display(i,logging)
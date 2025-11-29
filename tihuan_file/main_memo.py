import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    args.config = f"./exps/{args.model_name}.json"
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    # args.update(param)  # Add parameters from json
    param.update(args) 
    train(param)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')

    parser.add_argument('--dataset', type=str, default="cifar100")
    parser.add_argument('--memory_size','-ms',type=int, default=3312)       #可去
    parser.add_argument('--init_cls', '-init', type=int, default=50)
    parser.add_argument('--increment', '-incre', type=int, default=10)
    parser.add_argument('--model_name','-model', type=str, default='memo', required=False)
    parser.add_argument('--convnet_type','-net', type=str, default='memo_resnet32')
    parser.add_argument('--prefix','-p',type=str, help='exp type', default='fair', choices=['benchmark', 'fair', 'auc'])
    parser.add_argument('--device','-d', nargs='+', type=int, default=[0])
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--skip', action="store_true",)
    
    # Added in MEMO
    parser.add_argument('--train_base',action='store_true')
    parser.add_argument('--train_adaptive',action='store_true')

    #ETF
    parser.add_argument('--ETF_classifier', type=bool, default=True)
    parser.add_argument('--mixup', type=bool, default=True)
    parser.add_argument('--reg_dot_loss', type=bool, default=True)
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=100)                  #数据集的总类别数，由数据集决定。
    parser.add_argument('--GivenEw', type=bool, default=True)
    parser.add_argument('--reg_lam', type=float, default=0.0)
    parser.add_argument('--criterion', type=str, default="reg_dot_loss")

    #configer
    parser.add_argument('--head_class_idx', nargs='+', type=int, default=[0, 36])
    parser.add_argument('--med_class_idx', nargs='+', type=int, default=[36, 71])
    parser.add_argument('--tail_class_idx', nargs='+', type=int, default=[71, 100])


    # init
    parser.add_argument('--scheduler', type=str, default='steplr', choices=['steplr','cosine'])
    parser.add_argument('--init_epoch', type=int, default=100)
    parser.add_argument('--t_max', type=int, default=None)
    parser.add_argument('--init_lr', type=float, default=0.1)
    parser.add_argument('--init_milestones', type=list, default=[60,120,170])
    parser.add_argument('--init_lr_decay', type=float, default=0.1)
    parser.add_argument('--init_weight_decay', type=float, default=0.0005)
    
    # update
    parser.add_argument('--epochs', type=int, default=170)
    parser.add_argument('--lrate', type=float, default=0.1)
    parser.add_argument('--milestones', type=list, default=[80,120,150])
    parser.add_argument('--lrate_decay', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=2e-4)

    parser.add_argument('--alpha_aux', type=float, default=1.0)
    return parser


if __name__ == '__main__':
    main()

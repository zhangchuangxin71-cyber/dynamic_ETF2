import json
import argparse
from trainer import train


def main():
    parser = setup_parser()
    args = parser.parse_args()

    # --- Start of Validation and Auto-configuration Logic ---
    if args.reg_dot_loss and not args.ETF_classifier:
        parser.error("--reg_dot_loss requires --ETF_classifier to be enabled.")

    # Automatically set criterion based on reg_dot_loss for convenience
    if args.reg_dot_loss:
        args.criterion = "reg_dot_loss"
    # --- End of Validation Logic ---

    # Load config file if it exists
    args.config = f"./exps/{args.model_name}.json"
    try:
        param = load_json(args.config)
    except FileNotFoundError:
        print(f"Warning: Config file {args.config} not found. Using command-line arguments only.")
        param = {}

    # Update params with command-line arguments
    args_dict = vars(args)
    param.update(args_dict)

    # Print key configurations for verification
    print("\n" + "=" * 50)
    print("Running with the following key configurations:")
    print(f"  - Dataset: {param['dataset']}")
    print(f"  - Backbone: {param['convnet_type']}")
    print(f"  - Use Adapt Layer (AL): {param['use_adapt_layer']}")
    print(f"  - Use ETF Classifier: {param['ETF_classifier']}")
    print(f"  - Use Dot Loss: {param['reg_dot_loss']}")
    print(f"  - Criterion: {param['criterion']}")
    print("=" * 50 + "\n")

    train(param)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Reproduce of multiple continual learning algorithms.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- General ---
    parser.add_argument('--dataset', type=str, default="imagenet100", help='Dataset to use.')
    parser.add_argument('--memory_size', '-ms', type=int, default=3312, help='Total memory size for exemplars.')
    parser.add_argument('--init_cls', '-init', type=int, default=50, help='Number of initial classes.')
    parser.add_argument('--increment', '-incre', type=int, default=10, help='Number of classes per increment.')
    parser.add_argument('--model_name', '-model', type=str, default='memo', help='Model name for config file.')
    parser.add_argument('--convnet_type', '-net', type=str, default='memo_resnet34_imagenet',
                        choices=['memo_resnet34_imagenet', 'memo_resnet32'], help='Backbone network type.')
    parser.add_argument('--prefix', '-p', type=str, default='fair', help='Experiment prefix.')
    parser.add_argument('--device', '-d', nargs='+', type=int, default=[0], help='GPU devices to use.')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode.')
    parser.add_argument('--skip', action='store_true', default=False, help='Skip existing experiments.')

    # --- SCL-PNC Components (Fixed Boolean Logic) ---
    parser.add_argument('--use_adapt_layer', action='store_true', help='Enable the Adapt-Layer (AL). Default is ON.')
    parser.add_argument('--no-use_adapt_layer', dest='use_adapt_layer', action='store_false',
                        help='Disable the Adapt-Layer (AL).')
    parser.set_defaults(use_adapt_layer=True)

    parser.add_argument('--ETF_classifier', action='store_true', default=False,
                        help='Enable the Parametric ETF Classifier. Default is OFF.')
    parser.add_argument('--reg_dot_loss', action='store_true', default=False,
                        help='Enable the Dot Regression Loss (requires --ETF_classifier). Default is OFF.')

    parser.add_argument('--GivenEw', action='store_true', help='Enable sample re-weighting (Ew). Default is ON.')
    parser.add_argument('--no-GivenEw', dest='GivenEw', action='store_false', help='Disable sample re-weighting (Ew).')
    parser.set_defaults(GivenEw=True)

    # --- Training Strategy ---
    parser.add_argument('--train_base', action='store_true', default=False, help='Flag to train the base model.')
    parser.add_argument('--train_adaptive', action='store_true',
                        help='Enable training of the adaptive part. Default is ON.')
    parser.add_argument('--no-train_adaptive', dest='train_adaptive', action='store_false',
                        help='Disable training of the adaptive part.')
    #parser.set_defaults(train_adaptive=True)

    parser.add_argument('--mixup', action='store_true', default=False, help='Enable mixup augmentation.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Mixup alpha parameter.')
    parser.add_argument('--reg_lam', type=float, default=0.0, help='Lambda for regularization.')
    parser.add_argument('--criterion', type=str, default="cross_entropy", choices=["reg_dot_loss", "cross_entropy"],
                        help='Loss function. Automatically set to reg_dot_loss if --reg_dot_loss is used.')

    # --- Dataset & Classes ---
    parser.add_argument('--num_classes', type=int, default=100, help='Total number of classes in the dataset.')
    parser.add_argument('--head_class_idx', nargs=2, type=int, default=[0, 50],
                        help='Start and end index for head classes.')
    parser.add_argument('--med_class_idx', nargs=2, type=int, default=[50, 80],
                        help='Start and end index for medium classes.')
    parser.add_argument('--tail_class_idx', nargs=2, type=int, default=[80, 100],
                        help='Start and end index for tail classes.')

    # --- Initial Phase Training Hyperparameters ---
    parser.add_argument('--scheduler', type=str, default='steplr', choices=['steplr', 'cosine'],
                        help='Learning rate scheduler.')
    parser.add_argument('--init_epoch', type=int, default=200, help='Epochs for initial phase training.')
    parser.add_argument('--t_max', type=int, default=None, help='T_max for cosine scheduler.')
    parser.add_argument('--init_lr', type=float, default=0.1, help='Initial learning rate for first phase.')
    parser.add_argument('--init_milestones', nargs='+', type=int, default=[60, 120, 170],
                        help='Milestones for initial phase StepLR.')
    parser.add_argument('--init_lr_decay', type=float, default=0.1, help='LR decay factor for initial phase.')
    parser.add_argument('--init_weight_decay', type=float, default=0.0005, help='Weight decay for initial phase.')

    # --- Incremental Phase Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=200, help='Epochs for incremental training.')
    parser.add_argument('--lrate', type=float, default=0.1, help='Learning rate for incremental training.')
    parser.add_argument('--milestones', nargs='+', type=int, default=[80, 120, 150],
                        help='Milestones for incremental phase StepLR.')
    parser.add_argument('--lrate_decay', type=float, default=0.1, help='LR decay factor for incremental phase.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='Weight decay for incremental phase.')

    # --- Loss Weights ---
    parser.add_argument('--alpha_aux', type=float, default=1.0, help='Auxiliary loss weight.')
    parser.add_argument('--distill_weight', type=float, default=0.25, help='Distillation loss weight.')

    return parser


if __name__ == '__main__':
    main()

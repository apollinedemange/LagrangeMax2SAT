""" The handler for inference."""

""" The values n_layers and hidden_dim in this file must be identical to those used for training."""

import torch
import argparse
import time

from lagrangemax2sat.build_weights import build_weights_matrix
from lagrangemax2sat.io.ReadWcnf import ReadWcnf
from lagrangemax2sat.pl_maxsat_model import MAX2SATModel

from argparse import ArgumentParser


def arg_parser():
    parser = ArgumentParser(description='Train a Pytorch-Lightning model on a dataset.')
    parser.add_argument('--task', type=str, default='maxsat')
    parser.add_argument('--storage_path', type=str, default='')
    
    parser.add_argument("--filepath", type=str, default='data/wcnf/instance_50c_10v_1.wcnf')

    parser.add_argument('--loss_function', type=str, default='TSL') #TSL = TwoStepsLoss; SML = SumMinLoss; L2L= L2Loss
    
    parser.add_argument('--training_split', type=str, default='data/train/')
    parser.add_argument('--training_split_label_dir', type=str, default='data/train_label/',
                        help="Directory containing labels for training split (used for MIS).")
    parser.add_argument('--validation_split', type=str, default='data/valid/')
    parser.add_argument('--validation_split_label_dir', type=str, default='data/valid_label/',
                        help="Directory containing labels for validation split (used for MIS).")
    parser.add_argument('--test_split', type=str, default='data/valid/')    
    parser.add_argument('--test_split_label_dir', type=str, default='data/valid_label/',
                        help="Directory containing labels for test split (used for MIS).")
    
    parser.add_argument('--validation_examples', type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_scheduler', type=str, default='constant') # or 'one-cycle' or 'cosine-decay'

    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--fp16', action='store_true')

    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--two_opt_iterations', type=int, default=1000)
    parser.add_argument('--save_numpy_heatmap', action='store_true')

    parser.add_argument('--project_name', type=str, default='lagrangemax2sat')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_logger_name', type=str, default='modif_transpose')
    parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
    parser.add_argument('--ckpt_path', type=str, default="models/400c_50v_6l_128hd.ckpt")
    parser.add_argument('--resume_weight_only', action='store_true')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_valid_only', action='store_true')

    args = parser.parse_args()
    return args


def inference(args, D=2):
    # Build weight matrix and find inital minimal weight
    wcnf_file = ReadWcnf(args.filepath)
    wcnf_filepath = wcnf_file.wcnf_filepath
    N = wcnf_file.N
    weights, weight_min = build_weights_matrix(N, D, wcnf_filepath)

    # Unary weights
    index = torch.arange(N)
    unary_weights = weights[index, index].diagonal(dim1=1, dim2=2)

    # Binary weights
    binary_weights = weights.clone()
    binary_weights[index, index] = 0.0
    binary_weights_t = torch.transpose(binary_weights, 0, 1)
    binary_weights_t = torch.transpose(binary_weights_t, 2, 3)
    binary_weights = binary_weights + binary_weights_t

    if torch.cuda.is_available():
        # Get the CUDA device
        device = torch.device("cuda")

    # Load the checkpoint
    model = MAX2SATModel.load_from_checkpoint(args.ckpt_path, param_args=args)

    
    # Infere on the model to find the predicted solution
    start = time.time()
    sol_pred = model.infer_solution(unary_weights.unsqueeze(0).to(device), binary_weights.unsqueeze(0).to(device), weights.to(device), weight_min)
    end = time.time()

    # Be sure the solution is never negative
    if sol_pred<0:
        sol_pred = 0

    print(f"TIME={end-start}")
    print(f"SOLUTION={sol_pred}")


if __name__ == "__main__":
    args = arg_parser()
    solution = inference(args)

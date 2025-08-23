"""The handler for training and evaluation."""

import os
import torch
import wandb

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from pl_maxsat_model import MAX2SATModel


def arg_parser():
    parser = ArgumentParser(description='Train a Pytorch-Lightning model on a dataset.')
    parser.add_argument('--task', type=str, default='maxsat')
    parser.add_argument('--storage_path', type=str, default='')

    parser.add_argument('--loss_function', type=str, default='TSL') #TSL = TwoStepsLoss; SML = SumMinLoss; L2L= L2Loss
    
    parser.add_argument('--training_split', type=str, default='data/train/')
    parser.add_argument('--training_split_label_dir', type=str, default='data/train_label/',
                        help="Directory containing labels for training split (used for MIS).")
    parser.add_argument('--validation_split', type=str, default='data/valid/')
    parser.add_argument('--validation_split_label_dir', type=str, default='data/valid_label/',
                        help="Directory containing labels for validation split (used for MIS).")
    parser.add_argument('--test_split', type=str, default='data/wncf/instance_50c_10v_2.wcnf')    
    parser.add_argument('--test_split_label_dir', type=str, default='data/osac_solution/',
                        help="Directory containing labels for test split (used for MIS).")
    
    parser.add_argument('--validation_examples', type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_scheduler', type=str, default='constant') # or 'one-cycle' or 'cosine-decay'

    parser.add_argument('--num_workers', type=int, default=1)
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
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--resume_weight_only', action='store_true')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_valid_only', action='store_true')

    args = parser.parse_args()
    return args


def main(args):
    # Parameters of the model
    epochs = args.num_epochs
    project_name = args.project_name
    if args.task != 'maxsat':
        assert False, "Error: wrong task"
    model_class = MAX2SATModel
    saving_mode = 'min'
        
    # Create a model
    model = model_class(param_args=args)

    # Logger configuration
    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name=args.wandb_logger_name,
        project=project_name,
        entity=args.wandb_entity,
        save_dir=os.path.join(args.storage_path, f'models'),
        id=args.resume_id or wandb_id,
    )
    rank_zero_info(f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}")

    # Callbacks creation
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss', mode=saving_mode,
        save_top_k=3, save_last=True,
        dirpath=os.path.join(wandb_logger.save_dir,
                            args.wandb_logger_name,
                            wandb_logger._id,
                            'checkpoints'),
    )
    lr_callback = LearningRateMonitor(logging_interval='step')

    # Trainer initialization
    trainer = Trainer(
        accelerator="auto",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        max_epochs=epochs,
        #profiler="simple",
        callbacks=[TQDMProgressBar(refresh_rate=20, leave=True), checkpoint_callback, lr_callback], #Timer(duration="00:00:02:00")],
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        strategy=DDPStrategy(static_graph=True),
        precision=16 if args.fp16 else 32,
    )

    # Print model architecture
    rank_zero_info(
        f"{'-' * 100}\n"
        f"{str(model.model)}\n"
        f"{'-' * 100}\n"
    )

    # Load checkpoint if it is specified
    ckpt_path = args.ckpt_path

    # Training, testing and validating
    if args.do_train:
        if args.resume_weight_only:
            model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=ckpt_path)

        if args.do_test:
            trainer.test(ckpt_path=checkpoint_callback.best_model_path)

    elif args.do_test:
        trainer.validate(model, ckpt_path=ckpt_path)
        if not args.do_valid_only:
            trainer.test(model, ckpt_path=ckpt_path)
    
    trainer.logger.finalize("success")


    last_ckpt_path = checkpoint_callback.last_model_path
    if os.path.exists(last_ckpt_path):
        last_model = model_class.load_from_checkpoint(last_ckpt_path, param_args=args)
        val_results = trainer.validate(last_model, verbose=False)
        final_val_loss = val_results[0]['val/loss']
        final_val_gap = val_results[0]['val/gap']
        osac = val_results[0]['val/mean_osac']
        return final_val_loss, final_val_gap, osac
    return None


if __name__ == '__main__':
    args = arg_parser()
    loss = main(args)

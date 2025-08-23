"""A meta PyTorch Lightning model for training and evaluating models."""

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_info
import torch.optim.lbfgs

from models.gnn_encoder import GNNEncoder
from lagrangemax2sat.utils.lr_schedulers import get_schedule_fn
from lagrangemax2sat.co_datasets.maxsat_dataset import custom_collate

class COMetaModel(pl.LightningModule):
  def __init__(self,
              param_args):
    super(COMetaModel, self).__init__()
    self.args = param_args
    self.custom = custom_collate

    self.model = GNNEncoder(
        n_layers=self.args.n_layers,
        hidden_dim=self.args.hidden_dim,
        aggregation=self.args.aggregation
    )
    self.num_training_steps_cached = None

  def get_total_num_training_steps(self) -> int:
    """Total training steps inferred from datamodule and devices."""
    if self.num_training_steps_cached is not None:
      return self.num_training_steps_cached
    dataset = self.train_dataloader()
    if self.trainer.max_steps and self.trainer.max_steps > 0:
      return self.trainer.max_steps

    dataset_size = (
        self.trainer.limit_train_batches * len(dataset)
        if self.trainer.limit_train_batches != 0
        else len(dataset)
    )

    num_devices = max(1, self.trainer.num_devices)
    effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
    self.num_training_steps_cached = (dataset_size // effective_batch_size) * self.trainer.max_epochs
    return self.num_training_steps_cached

  def configure_optimizers(self):
    rank_zero_info('Parameters: %d' % sum([p.numel() for p in self.model.parameters()]))
    rank_zero_info('Training steps: %d' % self.get_total_num_training_steps())

    optimizer = torch.optim.AdamW(
          self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
    
    if self.args.lr_scheduler == "constant":
      return optimizer
    else:
      scheduler = get_schedule_fn(self.args.lr_scheduler, self.get_total_num_training_steps())(optimizer)
      return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler": scheduler,
              "interval": "step",
          },
      }
    

  def train_dataloader(self):
    batch_size = self.args.batch_size
    dataloader = torch.utils.data.DataLoader(
        self.train_dataset,
        batch_size=batch_size,
        collate_fn=self.custom,
        shuffle=True,
        num_workers=self.args.num_workers,
        pin_memory=True
    )
    return dataloader

  def test_dataloader(self):
    batch_size = len(self.test_dataset)
    print("Test dataset size:", len(self.test_dataset))
    test_dataloader = torch.utils.data.DataLoader(
        self.test_dataset,
        batch_size=batch_size,
        collate_fn=self.custom,
        shuffle=False,
        pin_memory=True
    )
    return test_dataloader

  def val_dataloader(self):
    batch_size = self.args.validation_examples
    val_dataset = torch.utils.data.Subset(self.validation_dataset, range(self.args.validation_examples))
    print("Validation dataset size:", len(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=self.custom,
        shuffle=False,
        num_workers=self.args.num_workers,
        pin_memory=True
    )
    return val_dataloader

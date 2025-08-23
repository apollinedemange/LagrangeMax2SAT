"""Lightning module for training the MAX2SAT model"""

import os
import torch.utils.data

from lagrangemax2sat.co_datasets.maxsat_dataset import MAX2SATDataset
from pl_meta_model import COMetaModel
from lagrangemax2sat.LossMethods.TwoStepsLoss import TwoStepsLoss
from lagrangemax2sat.LossMethods.SumMinLoss import SumMinLoss
from lagrangemax2sat.LossMethods.L2Loss import L2Loss

class MAX2SATModel(COMetaModel):
    def __init__(self,
                param_args=None):
        super(MAX2SATModel, self).__init__(param_args=param_args)
        
        # Training set
        data_label_dir = None
        if self.args.training_split_label_dir is not None:
            data_label_dir = os.path.join(self.args.storage_path, self.args.training_split_label_dir)

        self.train_dataset = MAX2SATDataset(
            data_file=os.path.join(self.args.storage_path, self.args.training_split),
            data_label_dir=data_label_dir,
        )

        # Testing set
        data_label_dir = None
        if self.args.test_split_label_dir is not None:
            data_label_dir = os.path.join(self.args.storage_path, self.args.test_split_label_dir)

        self.test_dataset = MAX2SATDataset(
            data_file=os.path.join(self.args.storage_path, self.args.test_split),
            data_label_dir=data_label_dir,
        )

        # Validation set
        data_label_dir = None
        if self.args.validation_split_label_dir is not None:
            data_label_dir = os.path.join(self.args.storage_path, self.args.validation_split_label_dir)

        self.validation_dataset = MAX2SATDataset(
            data_file=os.path.join(self.args.storage_path, self.args.validation_split),
            data_label_dir=data_label_dir,
        )

    def forward(self, x, graph):
        return self.model(x, graph=graph)


    def training_step(self, batch):
        """Function for categorical training step
        Args:
            batch (_type_): _description_

        Returns:
            loss: _description_
        """
        # Inputs
        x = batch["x"]                    # (B*N*2)
        graph = batch["graph"]            # (B*N*N*2*2)
        weight_min = batch["weight_min"]  # (B)
        weights = batch["weights"]        # (B*N*N*2*2)
        osac_solution = batch["osac_solution"]
        unary_lagrangian_ref = batch["unary_lagrangian_ref"]
        binary_lagrangian_ref = batch["binary_lagrangian_ref"]

        batch_size = len(weight_min)

        # Predictions
        x_pred, graph_pred = self.forward(x, graph)

        # Loss mean calculation on the batch
        loss_sum = 0
        for i in range (batch_size):
            osac_i = osac_solution[i]

            # Loss for element i
            if self.args.loss_function == "TSL":
                loss_func = TwoStepsLoss(weights[i],          # (N*N*2*2)
                                        weight_min[i],        # scalar
                                        x_pred[i],            # unary_lagrangian predicted (N)
                                        graph_pred[i],        # binary_lagrangian predicted (N*N*2)
                                        osac_i)               # scalar
            elif self.args.loss_function == "SML":
                loss_func = SumMinLoss(weights[i],            # (N*N*2*2)
                                        weight_min[i],        # scalar
                                        x_pred[i],            # unary_lagrangian predicted (N)
                                        graph_pred[i],        # binary_lagrangian predicted (N*N*2)
                                        osac_i)               # scalar
            elif self.args.loss_function == "L2L":
                loss_func = L2Loss( x_pred[i],                # unary_lagrangian predicted (N)
                                    graph_pred[i],            # binary_lagrangian predicted (N*N*2)
                                    unary_lagrangian_ref[i],  # unary_lagrangian of reference (N)
                                    binary_lagrangian_ref[i]) # binary_lagrangian of reference (N*N*2)
            else :
                raise ValueError("Unvalid loss function.")

            loss_i = loss_func.loss()
            loss_sum = loss_sum + loss_i

        # Mean of losses over all the batch
        loss = -loss_sum/batch_size

        # Log the mean of the losses on all the batches at each epoch (sync_dist usefull if several gpu)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, batch_size=batch_size, reduce_fx="mean", sync_dist=True)
        return loss

    def test_step(self, batch, split='test'):
        # Inputs
        x = batch["x"]                    # (B*N*2)
        graph = batch["graph"]            # (B*N*N*2*2)
        weight_min = batch["weight_min"]  # (B)
        weights = batch["weights"]        # (B*N*N*2*2)
        osac_solution = batch["osac_solution"]
        unary_lagrangian_ref = batch["unary_lagrangian_ref"]
        binary_lagrangian_ref = batch["binary_lagrangian_ref"]

        batch_size = len(weight_min)

        # Predictions
        x_pred, graph_pred = self.forward(x, graph)
        
        # Loss mean calculation on the batch
        loss_sum, gap_sum, osac_sum = 0, 0, 0
        for i in range (batch_size):
            osac_i = osac_solution[i]
            osac_sum += osac_i
            
            # Loss for element i
            if self.args.loss_function == "TSL":
                loss_func = TwoStepsLoss(weights[i],          # (N*N*2*2)
                                        weight_min[i],        # scalar
                                        x_pred[i],            # unary_lagrangian predicted (N)
                                        graph_pred[i],        # binary_lagrangian predicted (N*N*2)
                                        osac_i)               # scalar
            elif self.args.loss_function == "SML":
                loss_func = SumMinLoss(weights[i],            # (N*N*2*2)
                                        weight_min[i],        # scalar
                                        x_pred[i],            # unary_lagrangian predicted (N)
                                        graph_pred[i],        # binary_lagrangian predicted (N*N*2)
                                        osac_i)               # scalar
            elif self.args.loss_function == "L2L":
                loss_func = L2Loss( x_pred[i],                # unary_lagrangian predicted (N)
                                    graph_pred[i],            # binary_lagrangian predicted (N*N*2)
                                    unary_lagrangian_ref[i],  # unary_lagrangian of reference (N)
                                    binary_lagrangian_ref[i]) # binary_lagrangian of reference (N*N*2)
            else :
                raise ValueError("Unvalid loss function.")
            
            loss_i = loss_func.loss()
            loss_sum = loss_sum + loss_i

            # GAP calculation
            if len(osac_solution)>0:
                if osac_i == 0:
                    gap_i = 0
                else:
                    gap_i = abs(loss_i - osac_i) / abs(osac_i) * 100
                gap_sum += gap_i
            
        loss = loss_sum/batch_size
        gap = gap_sum/batch_size
        osac = osac_sum/batch_size

        metrics = {
            f"{split}/loss": loss,
            f"{split}/gap": torch.tensor(gap),
            f"{split}/mean_osac": osac
        }

        self.log(f"{split}/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size, reduce_fx="mean")
        self.log(f"{split}/gap", gap, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size, reduce_fx="mean")
        self.log(f"{split}/mean_osac", osac, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size, reduce_fx="mean")
        
        return metrics
    
    def validation_step(self, batch):
        return self.test_step(batch, split='val')
    
    def infer_solution(self, x, graph, weights, weight_min):
        # Lagrangians prediction
        x_pred, graph_pred = self.forward(x, graph)
        # Solution calculation
        if self.args.loss_function == "TSL":
            loss_func = TwoStepsLoss(weights,          
                                    weight_min,     
                                    x_pred.squeeze(0),        
                                    graph_pred.squeeze(0))
        elif self.args.loss_function == "SML":
            loss_func = SumMinLoss(weights,            
                                   weight_min,     
                                   x_pred.squeeze(0),     
                                   graph_pred.squeeze(0))

        sol_pred = loss_func.loss()
        if sol_pred < 0 :
            sol_pred = 0
        return sol_pred
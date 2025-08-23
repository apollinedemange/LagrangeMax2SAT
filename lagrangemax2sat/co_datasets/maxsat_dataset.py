""" MAX2SATDataset class allows to load all files from a folder and store informations about 
    them or it could only load one specific file for inference for instance."""

import glob
import os
import torch
import json
from pathlib import Path

from lagrangemax2sat.build_weights import build_weights_matrix
from lagrangemax2sat.io.ReadWcnf import ReadWcnf
from lagrangemax2sat.LossMethods.TwoStepsLoss import TwoStepsLoss
from lagrangemax2sat.LossMethods.SumMinLoss import SumMinLoss


class MAX2SATDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, data_label_dir = None, D=2):
        self.data_file = data_file
        self.data_label_dir = data_label_dir  # only the direction, no filename of extension
        self.D = D
        self.epsilon = 1e-3                   # desired precision of TwoStepLoss
        if data_file.endswith(".wcnf"):       # only one .wcnf file is loaded
            self.file_lines = [data_file]
        elif data_file.endswith("/"):         # a whole folder containing .wcnf is loaded
            self.file_lines = glob.glob(os.path.join(data_file, "*.wcnf"))
        else:
            raise ValueError("Must be a .wcnf' or path to a folder containing .wcnf.")
        print(f'Loaded "{data_file}" with {len(self.file_lines)} examples')

    def __len__(self):
        return len(self.file_lines)

    def __getitem__(self, idx):
        graph = ReadWcnf(self.file_lines[idx])
        num_nodes = graph.N
        weights, weight_min = build_weights_matrix(num_nodes, self.D, self.file_lines[idx])
        graph = weights.clone()

        # Extract unary weights from diagonal of weights
        index = torch.arange(num_nodes)
        unary_weights = weights[index, index].diagonal(dim1=1, dim2=2)
        
        # Create symetrical binary weights tensor
        graph[index, index] = 0.0  # elements on diagonal equal to zero
        graph_t = torch.transpose(graph, 0, 1)
        graph_t = torch.transpose(graph_t, 2, 3)
        graph = graph + graph_t

        # Extract optimal solution, unary and binary lagrangians of reference from .cfn file
        osac_solution = None
        if self.data_label_dir is not None:
            path = Path(self.file_lines[idx])
            filename = str(path.stem)
            if self.data_label_dir.endswith(".cfn"):
                solution_file = self.data_label_dir
            else :
                solution_file = self.data_label_dir + filename + '.cfn'

            # Open cfn file containing optimal solution
            with open(solution_file, 'r') as file:
                cfn_model = json.load(file)
            osac_solution = cfn_model.get("optimal_solution", {}).get("c0")
                        
            # Extract optimal unary lagrangians
            unary_lagrangian_ref = torch.zeros(num_nodes)
            for key, val in cfn_model["unary_lagrangians"].items():
                index = int(key.split("_")[-1]) - 1
                unary_lagrangian_ref[index] = float(val)

            # Extract optimal binary lagrangians
            binary_lagrangian_ref = torch.zeros((num_nodes, num_nodes, self.D))
            for key, val in cfn_model["binary_lagrangians"].items():
                _, _,  x_str, y_str = key.split("_")
                x, y = int(x_str) - 1, int(y_str) - 1
                list = [float(v) for v in val]
                assert len(list) == self.D * self.D, "Unexpected list in .cfn"
                binary_lagrangian_ref[x, y] = torch.tensor(list[:self.D])
                binary_lagrangian_ref[y, x] = torch.tensor(list[self.D:])

            # Check if both losses produce the optimal solution with optimal lagrangians
            loss_func = SumMinLoss(weights,
                                   weight_min,   
                                   unary_lagrangian_ref,
                                   binary_lagrangian_ref,
                                   osac_solution)
            loss_1 = loss_func.loss()

            loss_func = TwoStepsLoss(weights,
                                     weight_min,   
                                     unary_lagrangian_ref,
                                     binary_lagrangian_ref,
                                     osac_solution)
            loss_2 = loss_func.loss()

            if loss_1 != osac_solution:
                print("The SumMinLoss is:", loss_1, "The optimal solution:", osac_solution)
                assert False, "The loss SumMinLoss isn't working."

            if torch.abs(loss_2-osac_solution) > self.epsilon:
                print("The TwoStepsLoss is:", loss_2, "The optimal solution:", osac_solution)
                assert False, "The loss TwoStepsLoss isn't working."

        return {
            'x' : unary_weights,                            # nodes features (N*2)
            'graph' : graph,                                # edges features (N*N*2*2)
            'weight_min' : weight_min,                      # usefull for loss (scalar)
            'weights' : weights,                            # usefull for loss (N*N*2*2)
            'num_nodes' : num_nodes,
            'osac_solution' : osac_solution,                # usefull for validation and testing (scalar)
            'filename' : self.file_lines[idx],  
            'unary_lagrangian_ref' : unary_lagrangian_ref,  # (N)
            'binary_lagrangian_ref' : binary_lagrangian_ref # (N*N*2)
        }


def custom_collate(batch):
    """ Custom collate function to batch graphs. """
    # Find maximum number of nodes and edges in the batch
    max_nodes = max([item['num_nodes'] for item in batch])

    # Initialize lists for batched tensors
    batched_x = []
    batched_graph = []
    batched_num_nodes = []
    batched_point_indicator = []
    batched_weights = []
    batched_weight_min = []
    batched_osac_solution = []
    batched_filename = []
    batched_unary_lagrangian = []
    batched_binary_lagrangian = []

    for item in batch:
        pad_N = max_nodes - item['num_nodes']

        # Pad node features [num_nodes, 2] -> [max_nodes, 2]
        x_pad = torch.nn.functional.pad(
            item['x'],
            (0, 0, 0, pad_N),
            value=0
        )
        batched_x.append(x_pad)

        # Pad edge features [num_nodes, num_nodes, 2, 2] -> [max_nodes, max_nodes, 2, 2]
        graph_pad = torch.nn.functional.pad(
            item['graph'], 
            (0, 0, 0, 0, 0, pad_N, 0, pad_N),
            value=0)
        batched_graph.append(graph_pad)

        # Pad weights
        weights_pad = torch.nn.functional.pad(
            item['weights'], 
            (0, 0, 0, 0, 0, pad_N, 0, pad_N),
            value=0
        )
        batched_weights.append(weights_pad)

        # Store the number of nodes for each graph
        batched_num_nodes.append(item['num_nodes'])
        
        # Store point indicator
        batched_point_indicator.append([max_nodes])

        # Store minimal weight
        batched_weight_min.append(item['weight_min'])

        # Store optimal solution
        if item['osac_solution'] != None:
            batched_osac_solution.append(item['osac_solution'])

        # Store filename (for debug purpose)
        batched_filename.append(item['filename'])

        # Pad unary_lagrangian_ref [num_nodes] -> [max_nodes]
        unary_lagrangian_pad = torch.nn.functional.pad(
            item['unary_lagrangian_ref'],
            (0, pad_N),
            value=0
        )
        batched_unary_lagrangian.append(unary_lagrangian_pad)

        # Pad binary_lagrangian_ref [num_nodes, num_nodes, 2] -> [max_nodes, max_nodes, 2]
        binary_lagrangian_pad = torch.nn.functional.pad(
            item['binary_lagrangian_ref'],
            (0, 0, 0, pad_N, 0, pad_N),
            value=0
        )
        batched_binary_lagrangian.append(binary_lagrangian_pad)

    # Stack tensors to create the batch
    return {
        'x': torch.stack(batched_x, dim=0),                                          # [batch_size, max_nodes, 2]
        'graph': torch.stack(batched_graph, dim=0),                                  # [batch_size, max_nodes, max_nodes, 2, 2]
        'num_nodes': torch.tensor(batched_num_nodes, dtype=torch.long),              # (batch_size)
        'point_indicator': torch.tensor(batched_point_indicator, dtype=torch.long),
        'weights' : torch.stack(batched_weights, dim=0),                             # usefull for loss (batch_size*N*N*2*2)
        'weight_min' : torch.tensor(batched_weight_min, dtype=torch.float32),        # usefull for loss (batch_size)
        'osac_solution' : torch.tensor(batched_osac_solution, dtype=torch.float32),  # usefull for validation and testing (batch_size)
        'filename' : batched_filename,                                               # usefull for DEBUG
        'unary_lagrangian_ref' : torch.stack(batched_unary_lagrangian, dim=0),       # usefull for L2 loss
        'binary_lagrangian_ref' : torch.stack(batched_binary_lagrangian, dim=0)      # usefull for L2 loss
    }

import sys
sys.path.append('/home/apdemange/lagrangemax2sat/')
import torch

from lagrangemax2sat.LossMethods.LossMethods import LossMethods
from lagrangemax2sat.update_weights import update_weights_torch, update_weights


class SumMinLoss(LossMethods):
    #TODO: modify the weight storage format if storage space problems occur later
    ''' This child class contains a method which calculates the sum of the minimums of all the functions to 
        update weights. The goal is to find the new minimal weight. '''
    def loss(self):        
        ''' 
        Parameters:
            - w : tensor containing the original weights of the problem
            - w_min : original minimal weight of the problem
            - unary_lagrangian : unary lagrangian multipliers that could be noisy or predicted by the GNN
            - binary_lagrangian : binary lagrangian multipliers that could be noisy or predicted by the GNN
        Outputs:
            - correct_w : tensor containing the reformulate problem obtained feasability conditions respected
            - correct_w_min : minimal weight of the reformulate problem obtained with feasability conditions respected
        '''
        # Update the weights with lagrangian multipliers even if it can possibily lead to unfeasible solution
        self.correct_w, self.correct_w_min = update_weights_torch(self.w, 
                                                                  self.w_min,
                                                                  self.unary_lagrangian, 
                                                                  self.binary_lagrangian)
        
        # Find the sum of the minima of all weight function on the LOWER HALF of the weights tensor correct_w.
        # The weight function are the matrix DxD contained inside correct_w.
        func_min = self.correct_w.amin(dim=(2, 3))  # (N,N)
        sum_lower = torch.tril(func_min, diagonal=-1).sum()

        # Find the sum of the minima of all weight function on the DIAGONAL of the weights tensor correct_w.
        # The goal is to find the minimum between elements (0,0) and (1,1) for each matrix DxD on the diagonal 
        # of correct_w.
        diag_idx = torch.arange(self.N, device=self.correct_w.device)
        diag_min = torch.minimum(self.correct_w[diag_idx, diag_idx, 0, 0], self.correct_w[diag_idx, diag_idx, 1, 1]) 
        sum_diag = diag_min.sum()

        # Calculate the corrected minimal weight
        self.correct_w_min = self.correct_w_min + sum_lower + sum_diag

        # Check feasability of new the solution
        if not self.test_feasibility():
            assert False, "The SumMinLoss doesn't provide a feasible solution."

        return self.correct_w_min
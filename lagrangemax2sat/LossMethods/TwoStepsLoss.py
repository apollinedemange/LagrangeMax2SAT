import torch

from lagrangemax2sat.LossMethods.LossMethods import LossMethods


class TwoStepsLoss(LossMethods):
    #TODO: modify the weight storage format if storage space problems occur later
    ''' This child class contains a method which function in 2 steps: first adjusting binary lagrangians, then 
        correcting unary lagrangians. '''
    def loss(self):
        ''' This function calculates the new weights of the problem after adjusting the lagrangian mulitpliers in 
            order to respect feasibility conditions. The goal is to find the new minimal weight.
        
        Parameters:
            - w : tensor containing the original weights of the problem
            - w_min : original minimal weight of the problem
            - unary_lagrangian : unary lagrangian multipliers that could be noisy or predicted by the GNN
            - binary_lagrangian : binary lagrangian multipliers that could be noisy or predicted by the GNN
        Outputs:
            - correct_w : tensor containing the reformulate problem obtained with CORRECTED lagrangian multipliers
            - correct_w_min : minimal weight of the reformulate problem obtained with CORRECTED lagrangian multipliers
            - correct_unary_lagrangian : unary lagrangian multipliers corrected to respect feasibility conditions
            - correct_binary_lagrangian : binary lagrangian multipliers corrected to respect feasibility conditions
        '''
        new_w = self.w.clone()
        self.correct_w = self.w.clone()

        epsilon = 1e-5  # precision

        ##########################
        ### New binary weights ###
        ##########################
        
        # Calculation of the new binary weights with uncorrect binary lagrangians
        x_idx, y_idx = torch.tril_indices(self.N, self.N, offset=-1, device=self.w.device) # (x,y) indexes of each pair of variables (lower half of the w matrix)
        x = x_idx.view(-1, 1).repeat(1, self.D * self.D).flatten()                         # index x duplicated D*D times
        y = y_idx.view(-1, 1).repeat(1, self.D * self.D).flatten()                         # index y duplicated D*D times
        a_idx = torch.arange(self.D, device=self.w.device).view(-1, 1).repeat(1, self.D).flatten()  # index a corresponding to the value of the variable x
        b_idx = torch.arange(self.D, device=self.w.device).repeat(self.D)                           # index b corresponding to the value of the variable y
        a = a_idx.repeat(len(x_idx))                                                       # index a duplicated in order to have the same length as x and y
        b = b_idx.repeat(len(x_idx))                                                       # index b duplicated in ordre to have the same length as x and y
        new_w[x,y,a,b] += - self.binary_lagrangian[x,y,a] - self.binary_lagrangian[y,x,b]  # calculating new binary weights

        # Correction of binary lagrangians
        self.correct_binary_lagrangian = self.binary_lagrangian.clone()
        mask = new_w[x,y,a,b] < 0
        correction = (new_w[x,y,a,b]) * 0.5 - epsilon  # adjustment of the 2 binary lagrangian multipliers equaly
        self.correct_binary_lagrangian.index_put_((x[mask], y[mask], a[mask]),correction[mask],accumulate=True)
        self.correct_binary_lagrangian.index_put_((y[mask], x[mask], b[mask]),correction[mask],accumulate=True)

        # Update binary weights after correction of binary lagrangians
        self.correct_w[x,y,a,b] += - self.correct_binary_lagrangian[x,y,a] - self.correct_binary_lagrangian[y,x,b]
        
        #########################
        ### New unary weights ###
        #########################

        # Calculation of the new unary weights with correct binary lagrangians and uncorrect unary lagrangians
        binary_sum = torch.sum(self.correct_binary_lagrangian, dim=1)               # (N,D), in (x,a) sum of weights projected on value a of variable x
        diag_vals = torch.diagonal(self.correct_binary_lagrangian, dim1=0, dim2=1)  # (D,N), data on the diagonal of binary_lagrangian (supposed to be zero)
        binary_sum = binary_sum - diag_vals.t() 
        unary_lagrangian_reshape = self.unary_lagrangian.view(self.N, 1)            # reshaping the unary lagrangians into a column
        x = torch.arange(self.N, device=self.w.device).view(-1, 1).repeat(1, self.D).flatten()  # tensor of indexes corresponding to variables
        a = torch.arange(self.D, device=self.w.device).repeat(self.N)                           # tensor of indexes corresponding to values of each variable
        new_w[x,x,a,a] += (-unary_lagrangian_reshape + binary_sum).flatten()        # calculating new unary weights

        # Correction of unary lagrangians
        mask = new_w[x,x,a,a] < 0
        correction = new_w.amin(dim=(2,3)) - epsilon  # (N,N)
        x_bis = torch.arange(self.N, device=self.w.device)
        self.correct_unary_lagrangian = unary_lagrangian_reshape.clone()
        self.correct_unary_lagrangian[x_bis,0] += correction.diagonal()[x_bis]
        
        # Update unary weights after correction of unary lagrangians
        self.correct_w[x,x,a,a] += (-self.correct_unary_lagrangian + binary_sum).flatten()

        #############################
        ### New zero arity weight ###
        #############################

        self.correct_w_min = self.w_min + torch.sum(self.correct_unary_lagrangian)

        # Check feasability of the new solution
        if (not self.test_feasibility()) or (not torch.all(self.correct_w >= 0)):
            assert False, "The TwoStepsLoss doesn't provide a feasible solution."
        
        return self.correct_w_min
    
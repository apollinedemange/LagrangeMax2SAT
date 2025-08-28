""" Debug loss which calculate the L2 norm between predicted and optimal lagrangians. """

class L2Loss():        
    def __init__(self, unary_lagrangian_pred, binary_lagrangian_pred, unary_lagrangian_ref, binary_lagrangian_ref):
        self.unary_lagrangian_pred = unary_lagrangian_pred      # unary lagrangians that possibily lead to unfeasible solution
        self.binary_lagrangian_pred = binary_lagrangian_pred    # binary lagrangians that possibily lead to unfeasible solution
        self.unary_lagrangian_ref = unary_lagrangian_ref        # unary lagrangians of reference
        self.binary_lagrangian_ref = binary_lagrangian_ref      # binary lagrangians of reference
        self.N = binary_lagrangian_pred.shape[0]                # number of varibles of the Weighted Max2SAT problem
        self.D = binary_lagrangian_pred.shape[2]                # size of the domain of variables

    def loss(self):
        """ Loss is the L2 norm between optimal and predicted lagrangians. """
        diff_unary = (self.unary_lagrangian_pred - self.unary_lagrangian_ref)**2
        diff_binary = (self.binary_lagrangian_pred - self.binary_lagrangian_ref)**2
        loss_l2 = diff_unary.sum() + diff_binary.sum()
        return loss_l2
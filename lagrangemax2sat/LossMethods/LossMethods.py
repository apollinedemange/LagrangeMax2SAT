""" Parent class for loss classes."""

import torch

class LossMethods():
    # Attributs that will be defined in child class
    correct_w = None                  # weights of the reformulate problem corrected to respect feasibility conditions
    correct_w_min = None              # minimal weight of the reformulate problem corrected to respect feasibility conditions
    correct_unary_lagrangian = None   # unary lagrangian multipliers corrected to respect feasibility conditions
    correct_binary_lagrangian = None  # binary lagrangian multipliers corrected to respect feasibility conditions
        
    def __init__(self, w, w_min, unary_lagrangian, binary_lagrangian, osac_opt_sol=None):
        self.w = w                                  # original weights of the problem
        self.w_min = w_min                          # original minimal weight of the problem
        self.unary_lagrangian = unary_lagrangian    # unary lagrangians that possibily lead to unfeasible solution
        self.binary_lagrangian = binary_lagrangian  # binary lagrangians that possibily lead to unfeasible solution
        self.osac_opt_sol = osac_opt_sol            # optimal solution of the problem
        self.N = w.shape[0]                         # number of varibles of the Max2SAT problem
        self.D = w.shape[2]                         # size of the domain of variables
        self.device = self.w.device

    def test_feasibility(self):
        ''' Method which checks if the new minimal weight is smaller than the optimal solution. '''
        if self.osac_opt_sol == None:
            print("Missing OSAC solution")
            return True
        if self.correct_w_min <= self.osac_opt_sol :
            return True
        print("The negative weights are:",self.correct_w[self.correct_w < 0])
        print("The minimal weight is:", float(self.correct_w_min))
        return False

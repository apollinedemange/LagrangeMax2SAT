''' Project: Max2SAT lagrangian generation. '''

import torch
import os
import argparse

from lagrangemax2sat.osac import osac
from lagrangemax2sat.update_weights import update_weights_torch
from lagrangemax2sat.build_weights import build_weights_matrix

# io
from lagrangemax2sat.io.ReadWcnf import ReadWcnf
from lagrangemax2sat.io.WriteWcnf import WriteWcnf
from lagrangemax2sat.io.WriteCfn import WriteCfn

# LossMethods
from lagrangemax2sat.LossMethods.TwoStepsLoss import TwoStepsLoss
from lagrangemax2sat.LossMethods.SumMinLoss import SumMinLoss


def main(filepath_raw, D=2):
    ''' 
    Parameters: 
        - filepath_raw: full path of the file containing the problem, can be a .wcnf or .cnf
        - D: size of the varibles domain 
    Additional information: 
        - class 'ReadWcnf' and function 'build_weights_matrix' only works for boolean variables (i.e. D=2) 
    '''
    ''' Open the WCNF file or create it if we have a CNF file at the input. Find the size number of variables, 
    the minimal weight and create the matrix containing the weights of the problem. '''
    wcnf_file = ReadWcnf(filepath_raw)
    wcnf_filepath = wcnf_file.wcnf_filepath
    data_size = wcnf_file.N
    weights_matrix, weight_min = build_weights_matrix(data_size, D, wcnf_filepath)
    #print("The minimal weight is:", weight_min, "\nThe weight matrix is:", weights_matrix)
    
    ''' Formulate the OSAC problem and create a pyscipopt model to solve it:
    - unary_lagrangian: vector contaning the unary lagrangians
    - binary_lagrangian: tensor contaning the binary lagrangians'''
    unary_lagrangian, binary_lagrangian, osac_opt_sol = osac(weights_matrix, weight_min)
    #print("The unary lagrangians:", unary_lagrangian, "\nThe binary lagrangians:", binary_lagrangian)

    ''' Generation of the new weight matrix corresponding to the reformulate problem, in a differentiable way. '''
    new_w, new_w_min = update_weights_torch(weights_matrix, 
                                            weight_min, 
                                            unary_lagrangian, 
                                            binary_lagrangian)
    ''' Generation of the CFN file. '''
    cfn_file = WriteCfn(wcnf_filepath, 
                        new_w, 
                        new_w_min)
    cfn_file.write()
    print("The name of the CFN file is:", cfn_file.cfn_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .wcnf or .cnf files")
    parser.add_argument("filepath",help="The .wcnf or the .cnf file to process")
    parser.add_argument("loss",help="The loss method: T or S")
    args = parser.parse_args()
    main(args.filepath, args.loss)
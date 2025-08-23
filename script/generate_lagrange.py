''' Solve Max2SAT problems and store optimal lagrangians in .cfn file. '''

import json
import argparse

from pathlib import Path

from lagrangemax2sat.osac import osac
from lagrangemax2sat.build_weights import build_weights_matrix
from lagrangemax2sat.io.ReadWcnf import ReadWcnf


def generate_lagrange(filepath, D=2):
    path = Path(filepath)
    filetype = str(path.suffix)
    filename = str(path.stem)
    if filetype != ".wcnf":
        assert False, "Error: file should be .wcnf"

    N = ReadWcnf(filepath).N  # Number of variables

    # Solving the problem
    weights_matrix, weight_min = build_weights_matrix(N, D, filepath)
    unary_lagrangian, binary_lagrangian, opt_sol = osac(weights_matrix, weight_min)
    
    # Model creation
    cfn_model = {
        "problem": {"name":f"lagrangians_solutions_of_{filename}"},
        "optimal_solution" : {"c0": opt_sol},        
        "unary_lagrangians": {},
        "binary_lagrangians": {}
    }

    # Add unary_lagrangians
    for x in range(N):
        cfn_model["unary_lagrangians"][f"u_l_{x+1}"] = float(unary_lagrangian[x])
    
    # Add binary_lagrangians
    for x in range(N):
        for y in range(x):
            b_l = binary_lagrangian[x,y].tolist() + binary_lagrangian[y,x].tolist()
            cfn_model["binary_lagrangians"][f"b_l_{x+1}_{y+1}"] = [float(l) for l in b_l]
    
    # Writing inside cfn file
    cfn_filename = "data/osac_solution/" + filename + ".cfn"
    with open(cfn_filename, 'w') as file:
        json.dump(cfn_model, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .wcnf or .cnf files")
    parser.add_argument("filepath",help="The full path of the wcnf file of the problem to solve")
    args = parser.parse_args()
    print(generate_lagrange(args.filepath))
import numpy as np
import torch
from pyscipopt import Model, quicksum


def osac(w, w_min):
    ''' OSAC -- Optimal soft arc consistency with PyScipOpt
    Parameters:
        - w     : matrix containing weights of the Max2SAT problem
        - w_min : minimal weight of the Max2SAT problem
    Outputs:
        - unary_lagrangian_tensor  : tensor of the unary langrangian multipliers determined by pyscipopt
        - binary_lagrangian_tensor : tensor of the binary langrangian multipliers determined bby pyscipopt
        - optimal_sol              : optimal solution found by pyscipopt on OSAC problem
    '''
    N = w.shape[0]   # number of varibles of the Max2SAT problem
    D =  w.shape[2]  # size of the domain of variables
    model = Model("OSAC")
    unary_lagrangian = np.zeros(N, dtype=object)
    binary_lagrangian = np.zeros((N,N,D), dtype=object)
    
    # Variables
    for x in range(N):
        unary_lagrangian[x] = model.addVar(vtype='C', lb=None, name=f"unary_lagrangian[{x}]")
        for y in range(N):
            for a in range(D):
                binary_lagrangian[x,y,a] = model.addVar(vtype='C', lb=None, name=f"binary_lagrangian[{x},{y},{a}]")

    # Constraints
    for x in range(N):
        for a in range(D):
            model.addCons(w[x,x,a,a] - unary_lagrangian[x] + quicksum(binary_lagrangian[x,y,a] for y in range(x)) + quicksum(binary_lagrangian[x,y,a] for y in range(x+1, N)) >= 0)
    for x in range(1,N):
        for y in range(x):
            for a in range(D):
                for b in range(D):
                    model.addCons(w[x,y,a,b] - (binary_lagrangian[x,y,a] + binary_lagrangian[y,x,b]) >= 0)

    # Objective function
    model.setObjective(quicksum(unary_lagrangian[x] for x in range(N)), "maximize")

    # Optimization and optimal solution
    model.optimize()
    sol = model.getBestSol()
    optimal_sol = model.getObjVal() + w_min
    print("\nThe optimal solution is:", optimal_sol,"\n")

    # Conversion in pytorch tensor
    unary_lagrangian_tensor = torch.tensor([sol[unary_lagrangian[i]] for i in range(N)], dtype=torch.float)
    binary_lagrangian_tensor = torch.zeros(N,N,D)
    for x in range(N):
        for y in range(N):
            for a in range(D):
                if x != y:
                    binary_lagrangian_tensor[x,y,a] = sol[binary_lagrangian[x,y,a]]

    return unary_lagrangian_tensor, binary_lagrangian_tensor, optimal_sol

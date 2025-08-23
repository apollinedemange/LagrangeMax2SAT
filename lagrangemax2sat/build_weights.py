import torch


def build_weights_matrix(N, D, filepath):
    ''' Builds the matrix containing unary and binary weights.
    Parameters:
        - N : number of varibles of the Max2SAT problem
        - D : size of the domain of variables
        - filepath : full path of the file containing the weights of the problem
    Outputs:
        - weights : matrix containing weights of the Max2SAT problem
        - weight_min : minimal weight of the Max2SAT problem
    '''
    weight_min = 0
    weights = torch.zeros(N,N,D,D)
    with open(filepath,"r") as file:
        for line in file:
            split_line = line.strip().split()
            line_size = len(split_line)
            if line.startswith("p") or line.startswith("c"):
                continue
            # Binary cost function.
            elif line_size == 4: 
                w, var1, var2, _, = split_line
                w, var1, var2 = float(w), int(var1), int(var2)
                pos1 = 0 if var1 > 0 else 1
                pos2 = 0 if var2 > 0 else 1
                # In order to fill only the lower half of the matrix.
                if abs(var1) < abs(var2): 
                    weights[abs(var2)-1, abs(var1)-1, pos2, pos1] += w
                elif abs(var1) > abs(var2):
                    weights[abs(var1)-1, abs(var2)-1, pos1, pos2] += w
            # Unary cost function.
            elif line_size == 3: 
                w, var, _, = split_line
                w, var = float(w), int(var)
                pos = 0 if var > 0 else 1
                weights[abs(var)-1, abs(var)-1, pos, pos] += w
            # Minimal cost.
            elif line_size == 2: 
                w, _, = split_line
                weight_min += float(w)
    return weights, weight_min

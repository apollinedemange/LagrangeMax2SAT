''' Two methods to update de weights matrix thanks to lagrangians. '''

import torch


def update_weights(w, w_min, unary_lagrangian, binary_lagrangian):
    ''' Builds the matrix of the reformulated problem. Code with loops. '''
    N = w.shape[0]   # number of varibles of the Max2SAT problem
    D = w.shape[2]   # size of the domain of variables

    # New zero arity weight.
    new_w_min = w_min + sum(unary_lagrangian)

    # New unary weights.
    new_w = torch.zeros_like(w)
    for x in range(N):
        for a in range(D):
            new_w[x,x,a,a] = w[x,x,a,a] - unary_lagrangian[x] + sum(binary_lagrangian[x,:,a]) - binary_lagrangian[x,x,a]
    # New binary weights.
    for x in range(1,N):
        for y in range(x):
            for a in range(D):
                for b in range(D):
                    new_w[x,y,a,b] = w[x,y,a,b] - binary_lagrangian[x,y,a] - binary_lagrangian[y,x,b]
    return new_w, new_w_min


def update_weights_torch(w, w_min, unary_lagrangian, binary_lagrangian):
    ''' Builds the matrix of the reformulated problem using torch functions only. '''
    N = w.shape[0]   # number of varibles of the Max2SAT problem
    D = w.shape[2]   # size of the domain of variables
    
    # New zero arity weight.
    new_w_min = w_min + torch.sum(unary_lagrangian)

    # New unary weights.
    new_w = torch.zeros_like(w)
    binary_sum = torch.sum(binary_lagrangian, dim=1)                        # (N,D), in (x,a) sum of weights projected on value a of variable x
    diag_vals = torch.diagonal(binary_lagrangian, dim1=0, dim2=1)           # (D,N), data on the diagonal of binary_lagrangian (supposed to be zero)
    binary_sum = binary_sum - diag_vals.t()                                 # (N,D), binary_sum without the data on the diagonal
    unary_lagrangian_reshape = unary_lagrangian.view(N, 1)                  # reshaping the unary lagrangians into a column
    x = torch.arange(N).view(-1, 1).repeat(1, D).flatten()                  # tensor of indexes corresponding to variables
    a = torch.arange(D).repeat(N)                                           # tensor of indexes corresponding to values of each variable
    new_w[x,x,a,a] = w[x,x,a,a] + (-unary_lagrangian_reshape + binary_sum).flatten()  # calculating new unary weights

    # New binary weights.
    x_idx, y_idx = torch.tril_indices(N, N, offset=-1)                      # (x,y) indexes of each pair of variables (lower half of the w matrix)
    x = x_idx.view(-1, 1).repeat(1, D * D).flatten()                        # index x duplicated D*D times
    y = y_idx.view(-1, 1).repeat(1, D * D).flatten()                        # index y duplicated D*D times
    a_idx = torch.arange(D).view(-1, 1).repeat(1, D).flatten()              # index a corresponding to the value of the variable x
    b_idx = torch.arange(D).repeat(D)                                       # index b corresponding to the value of the variable y
    a = a_idx.repeat(len(x_idx))                                            # index a duplicated in order to have the same length as x and y
    b = b_idx.repeat(len(x_idx))                                            # index b duplicated in ordre to have the same length as x and y
    new_w[x,y,a,b] = w[x,y,a,b] - binary_lagrangian[x,y,a] - binary_lagrangian[y,x,b]  # calculating new binary weights

    return new_w, new_w_min

""" Debug file to print gradients during optimisation without GNN structure."""

import torch
import matplotlib.pyplot as plt

from lagrangemax2sat.io.ReadWcnf import ReadWcnf

from lagrangemax2sat.build_weights import build_weights_matrix
from lagrangemax2sat.osac import osac

from lagrangemax2sat.LossMethods.TwoStepsLoss import TwoStepsLoss
from lagrangemax2sat.LossMethods.SumMinLoss import SumMinLoss
from lagrangemax2sat.LossMethods.L2Loss import L2Loss

from script.add_noise import add_noise


def get_grad(filepath, LossMethod, std_unary, std_binary, epoch, D=2):
    # Store weights of the problem in tensor
    wcnf_file = ReadWcnf(filepath)
    data_size = wcnf_file.N
    weights_matrix, weight_min = build_weights_matrix(data_size, D, filepath)

    # Geneation of reference lagrangians and optimal solution
    unary_lagrangian, binary_lagrangian, osac_opt_sol = osac(weights_matrix, weight_min)

    # Add noise on unary and/or binary lagrangians
    unary_lagrangian_noisy, binary_lagrangian_noisy = add_noise(unary_lagrangian, 
                                                                binary_lagrangian, 
                                                                std_unary, 
                                                                std_binary)
    
    # Usefull for grandient calculation
    unary_lagrangian_noisy.requires_grad_(True)
    binary_lagrangian_noisy.requires_grad_(True)

    # Configure optimizer
    parameters = [unary_lagrangian_noisy, binary_lagrangian_noisy]
    optimizer  = torch.optim.SGD(parameters, lr=1e-3)

    loss_history = []

    for i in range(epoch):
        # Set gradients to zero
        optimizer.zero_grad()

        # Loss calculation
        loss_func = LossMethod(weights_matrix,
                            weight_min,   
                            unary_lagrangian_noisy,
                            binary_lagrangian_noisy,
                            osac_opt_sol)
        
        loss_func2 = L2Loss(unary_lagrangian,
                                binary_lagrangian,   
                                unary_lagrangian_noisy,
                                binary_lagrangian_noisy)

        loss_value = -loss_func.loss() #+ loss_func2.loss()
        if i%100==0:
            print(f"The loss method {LossMethod} provides a loss : {loss_value}")

        loss_history.append(loss_value.item())

        # Backward propagation
        loss_value.backward()

        # Gradient descent
        optimizer.step()
        
        if i%100==0:
            diff_unary = (unary_lagrangian_noisy.clone().detach() - unary_lagrangian.clone().detach())**2
            diff_binary = (binary_lagrangian_noisy.clone().detach() - binary_lagrangian.clone().detach())**2
            loss_l2 = diff_unary.sum() + diff_binary.sum()
            print()
            print("losl2:",loss_l2.numpy())

            def pretty_print_vector(tensor, precision=4, columns=5):
                if tensor.dim() != 1:
                    raise ValueError("Only 1D tensors are supported.")

                tensor = tensor.float()
                for i in range(0, tensor.size(0), columns):
                    row = tensor[i:i+columns]
                    formatted_row = ' '.join(f"{val:.{precision}f}" for val in row)
                    print(formatted_row)
            print("noisy binary:")
            pretty_print_vector(binary_lagrangian_noisy.flatten())
            print(" binary:")
            pretty_print_vector(binary_lagrangian.flatten())


    

    plt.figure()
    plt.plot(range(len(loss_history)), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"osac_sol={osac_opt_sol} | std_unary={std_unary} | std_binary={std_binary}")
    plt.tight_layout()
    plt.show()
    plt.savefig("results/loss_get_grad.png")


    # Get gradients
    return unary_lagrangian_noisy.grad, binary_lagrangian_noisy.grad


if __name__ == "__main__":
    filepath = "/home/apdemange/lagrangemax2sat/data/wcnf/instance_50c_10v_1.wcnf"
    LossMethod = TwoStepsLoss # TwoStepsLoss SumMinLoss
    std_unary = 0.5           # Noise on unary lagrangian
    std_binary = 0.5          # Noise on binary lagrangian
    epoch = 5000
    grad_unary, grad_binary = get_grad(filepath, LossMethod, std_unary, std_binary, epoch)
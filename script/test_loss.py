''' test_loss allow to check if we obtain the same solution with 
osac or with a loss in which we put optimal lagrangians'''


from lagrangemax2sat.io.ReadWcnf import ReadWcnf

from lagrangemax2sat.build_weights import build_weights_matrix
from lagrangemax2sat.osac import osac

from lagrangemax2sat.LossMethods.TwoStepsLoss import TwoStepsLoss
from lagrangemax2sat.LossMethods.SumMinLoss import SumMinLoss
from lagrangemax2sat.LossMethods.L2Loss import L2Loss


def test_loss(filepath, D=2):
    # Store weights of the problem in tensor
    wcnf_file = ReadWcnf(filepath)
    data_size = wcnf_file.N
    weights_matrix, weight_min = build_weights_matrix(data_size, D, filepath)

     # Geneation of reference lagrangians and optimal solution
    unary_lagrangian, binary_lagrangian, osac_opt_sol = osac(weights_matrix, weight_min)

    # SumMinLoss calculation
    loss_func = SumMinLoss(weights_matrix,
                           weight_min,   
                           unary_lagrangian,
                           binary_lagrangian,
                           osac_opt_sol)
    loss_1 = loss_func.loss()

    # TwoStepsLoss calculation
    loss_func = TwoStepsLoss(weights_matrix,
                            weight_min,   
                            unary_lagrangian,
                            binary_lagrangian,
                            osac_opt_sol)
    loss_2 = loss_func.loss()

    # L2Loss calculation
    loss_func = L2Loss(unary_lagrangian,
                            binary_lagrangian,   
                            unary_lagrangian,
                            binary_lagrangian)
    loss_3 = loss_func.loss()

    # Tests on SumMinLoss
    if loss_1 == osac_opt_sol:
        print("The loss SumMinLoss is good.")
    elif loss_1 < osac_opt_sol:
        print("The loss SumMinLoss is feasible but doesn't provide the optimal solution.")
        print("The SumMinLoss is:", loss_1, "The optimal solution:", osac_opt_sol)
    else :
        print("The loss SumMinLoss is impossible.")
        print("The SumMinLoss is:", loss_1, "The optimal solution:", osac_opt_sol)

    # Tests on TwoStepsLoss
    if loss_2 == osac_opt_sol:
        print("The loss TwoStepsLoss is good.")
    elif loss_2 < osac_opt_sol:
        print("The loss TwoStepsLoss is feasible but doesn't provide the optimal solution.")
        print("The TwoStepsLoss is:", loss_2, "The optimal solution:", osac_opt_sol)
    else :
        print("The loss TwoStepsLoss is impossible.")
        print("The TwoStepsLoss is:", loss_2, "The optimal solution:", osac_opt_sol)
    
    # Tests on L2Loss
    if float(loss_3) != 0:
        print("The loss L2Loss isn't working.")
    else :
        print("The loss L2Loss is OK.")
        
    return


if __name__ == "__main__":
    filepath = "/home/apdemange/lagrangemax2sat/data/wcnf/instance_50c_10v_1.wcnf"
    test_loss(filepath)
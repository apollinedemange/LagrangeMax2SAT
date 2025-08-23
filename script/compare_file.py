from script.add_noise import add_noise
from lagrangemax2sat.LossMethods.TwoStepsLoss import TwoStepsLoss
from lagrangemax2sat.LossMethods.SumMinLoss import SumMinLoss


def compare_file(weights_matrix, weight_min, unary_lagrangian, binary_lagrangian, osac_opt_sol, std_noise, num_tests):
    ''' This function tests several times if TwoStepsLoss is better than SumMinLoss for a given problem
        where random noise were added and returns the percentage of success of each method. 
    Parameters:
        - weights_matrix : tensor containing the original weights of the problem
        - weight_min : original minimal weight of the problem
        - unary_lagrangian : unary lagrangian multipliers, obtained with OSAC
        - binary_lagrangian : binary lagrangian multipliers, obtained with OSAC
        - osac_opt_sol : optimal solution of the problem, obtained with OSAC
        - std_noise : standard deviation of the gaussian noise applied to lagrangians
        - num_tests : number of times TSL and SML methods are compared on input problem
    Outputs:
        - percentage of sucess of TSL method
        - percentage of sucess of SML method
        '''
    TSL_counter = 0  # TwoStepsLoss counter
    SML_counter = 0  # SumMinLoss counter
    
    for _ in range(num_tests):
        # Add noise to unary and binary lagrangian
        unary_lagrangian_noisy, binary_lagrangian_noisy = add_noise(unary_lagrangian, 
                                                                    binary_lagrangian, 
                                                                    std_noise)
        # TwoStepsLoss
        noisy_pb_TSL = TwoStepsLoss(weights_matrix, 
                                      weight_min, 
                                      unary_lagrangian_noisy, 
                                      binary_lagrangian_noisy, 
                                      osac_opt_sol)
        loss_TSL = float(noisy_pb_TSL.loss())
        #print("loss_TSL:", loss_TSL)
        
        # SumMinLoss
        noisy_pb_SML = SumMinLoss(weights_matrix, 
                                    weight_min, 
                                    unary_lagrangian_noisy, 
                                    binary_lagrangian_noisy, 
                                    osac_opt_sol)
        loss_SML = float(noisy_pb_SML.loss())
        #print("loss_SML:", loss_SML)
        
        # Compare the 2 methods
        if loss_TSL > loss_SML:
            TSL_counter += 1
        elif loss_TSL < loss_SML:
            SML_counter += 1
    return TSL_counter*100/num_tests, SML_counter*100/num_tests
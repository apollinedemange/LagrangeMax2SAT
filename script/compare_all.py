''' Comparing TwoStepsLoss and SumMinLoss on several files. '''
 
import argparse

from lagrangemax2sat.osac import osac
from lagrangemax2sat.build_weights import build_weights_matrix
from lagrangemax2sat.io.ReadWcnf import ReadWcnf
from script.compare_file import compare_file


def compare_all(num_files, data_dir, std_noise=0.01, num_tests=10, D=2):
    TSL_counter, SML_counter = 0, 0
    num_files = int(num_files)
    for instance in range(num_files):
        filepath_raw = data_dir + f"{instance+1}.cnf"

        ''' Open the WCNF file or create it if we have a CNF file at the input. Find the size number of variables, 
        the minimal weight and create the matrix containing the weights of the problem. '''
        wcnf_file = ReadWcnf(filepath_raw)
        wcnf_filepath = wcnf_file.wcnf_filepath
        data_size = wcnf_file.N
        weights_matrix, weight_min = build_weights_matrix(data_size, D, wcnf_filepath)

        ''' Formulate the OSAC problem and create a pyscipopt model to solve it:
        - unary_lagrangian  : vector contaning the unary lagrangians
        - binary_lagrangian : tensor contaning the binary lagrangians'''
        unary_lagrangian, binary_lagrangian, osac_opt_sol = osac(weights_matrix, weight_min)

        ''' Compare which method works better '''
        TSL_percent, SML_percent = compare_file(weights_matrix, 
                                                weight_min, 
                                                unary_lagrangian, 
                                                binary_lagrangian, 
                                                osac_opt_sol, 
                                                std_noise, 
                                                num_tests)
        TSL_counter += TSL_percent/100
        SML_counter += SML_percent/100

    print(f"\nIn {TSL_counter*100/num_files}% of the cases TwoStepsLoss gave a better result than SumMinLoss")
    print(f"\nIn {SML_counter*100/num_files}% of the cases SumMinLoss gave a better result than TwoStepsLoss")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add number of files, number of variables and clauses and data direction")
    parser.add_argument("num_files",help="Number of files to process")
    parser.add_argument("data_dir",help="Direction of files to process. Ex: data/cnf/instances_500c_100v_")
    args = parser.parse_args()
    compare_all(args.num_files, args.data_dir)
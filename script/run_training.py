''' This file allows to do several training but on only one file each time. '''

from lagrangemax2sat.train import main, arg_parser

import matplotlib.pyplot as plt

def run_training(nb_file, path, name):
    loss_tot, gap_tot, osac_tot = 0, 0, 0
    gap_values = []
    x_values = []  

    for i in range(nb_file):
        filepath = path + name + str(i+26) + ".wcnf"
        print("the filename is:", filepath)
        args = arg_parser()
        args.training_split = filepath
        args.validation_split = filepath
        args.test_split = filepath
        args.wandb_logger_name = "2steps_pb" + str(i+1) 
        args.do_train = True

        loss_i, gap_i, osac_i = main(args)
        loss_tot += loss_i
        gap_tot += gap_i
        osac_tot += osac_i

        gap_values.append(gap_i)
        x_values.append(i+1)

    loss_tot = loss_tot/nb_file
    osac_tot = osac_tot/nb_file
    print("The mean of osac is : ", osac_tot)
    print("The mean of loss is : ", loss_tot)
    print("The sum of all gap is:", gap_tot)

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, gap_values, marker='o', linestyle='-', color='blue', label='Gap')
    plt.xlabel("File number")
    plt.ylabel("Gap")
    plt.title("Gap for each file")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("results/gap_training.png")


if __name__ == '__main__':
    nb_file = 70
    path = "data/wcnf/"
    name = "instance_50c_10v_"
    run_training(nb_file, path, name)

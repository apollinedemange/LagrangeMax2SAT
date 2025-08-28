''' Comparison between initial lower bound from Toulbar2 and GNN prediction.
Also comparison between execution times.'''

import subprocess
import time
import re
import json
import numpy as np
import matplotlib.pyplot as plt

def run_inference(filepath, ckpt_path):
    """ GNN solution and execution time."""
    result = subprocess.run(
        ["python3", "lagrangemax2sat/inference.py", "--filepath", filepath, "--ckpt_path", ckpt_path],
        capture_output=True, text=True
    )
    time_match = re.search(r"TIME=([0-9.]+)", result.stdout)
    solution_match = re.search(r"SOLUTION=(-?[0-9]+(?:\.[0-9]+)?)", result.stdout)

    if time_match and solution_match:
        inference_time = float(time_match.group(1))
        solution_value = float(solution_match.group(1))
        print("Inference time:", inference_time)
        print("Solution:", solution_value)
    else:
        print("Expected output of inference.py:", result.stdout)

    return solution_value, inference_time


def run_toulbar2(filepath):
    #VAC solution and execution time
    result_VAC = subprocess.run(
        ["toulbar2", filepath, "-nopre", "-A", "-bt=0"],
        capture_output=True, text=True
    )
    if result_VAC.returncode != 0:
        print("Error in toulbar2 (VAC):", result_VAC.stderr)
        return None, None
    match = re.search(r"Dual bound:\s*([0-9]+)", result_VAC.stdout)
    match_time = re.search(r"Preprocessing time:\s+([0-9.]+)\s+seconds", result_VAC.stdout)
    time_VAC = float(match_time.group(1))
    if match:
        lower_bound_VAC = float(match.group(1))
    else:
        match = re.search(r"Initial lower and upper bounds:\s*\[(\d+)", result_VAC.stdout)
        if match :
            lower_bound_VAC = float(match.group(1))
        else:
            print("Intial lower bound not found in Toulbar2 (VAC).")
            return None, None

    # EDAC solution and execution time
    result_EDAC = subprocess.run(
        ["toulbar2", filepath, "-nopre", "-k=4", "-bt=0"],
        capture_output=True, text=True
    )
    if result_EDAC.returncode != 0:
        print("Error in toulbar2 (EDAC):", result_EDAC.stderr)
        return None, None
    match = re.search(r"Dual bound:\s*([0-9]+)", result_EDAC.stdout)
    match_time = re.search(r"Preprocessing time:\s+([0-9.]+)\s+seconds", result_EDAC.stdout)
    time_EDAC = float(match_time.group(1))
    if match:
        lower_bound_EDAC = float(match.group(1))
    else:
        match = re.search(r"Initial lower and upper bounds:\s*\[(\d+)", result_EDAC.stdout)
        if match :
            lower_bound_EDAC = float(match.group(1))
        else:
            print("Intial lower bound not found in Toulbar2 (EDAC).")
            return None, None
    
    # FDAC solution and execution time
    result_FDAC = subprocess.run(
        ["toulbar2", filepath, "-nopre", "-k=3", "-bt=0"],
        capture_output=True, text=True
    )
    if result_FDAC.returncode != 0:
        print("Error in toulbar2 (FDAC):", result_FDAC.stderr)
        return None, None
    match = re.search(r"Dual bound:\s*([0-9]+)", result_FDAC.stdout)
    match_time = re.search(r"Preprocessing time:\s+([0-9.]+)\s+seconds", result_FDAC.stdout)
    time_FDAC = float(match_time.group(1))
    if match:
        lower_bound_FDAC = float(match.group(1))
    else:
        match = re.search(r"Initial lower and upper bounds:\s*\[(\d+)", result_FDAC.stdout)
        if match :
            lower_bound_FDAC = float(match.group(1))
        else:
            print("Intial lower bound not found in Toulbar2 (FDAC).")
            return None, None

    return lower_bound_VAC, time_VAC, lower_bound_EDAC, time_EDAC, lower_bound_FDAC, time_FDAC


def toulbar_vs_gnn(filepath, ckpt_path, osacpath, nb_file):
    # Init sum of each gap
    osac_zero = 0
    gaps_gnn, gaps_vac, gaps_edac, gaps_fdac = [], [], [], []
    times_gnn, times_vac, times_edac, times_fdac = [], [], [], []
    # Loop on all the files
    for i in range (nb_file):
        full_filepath = filepath + str(i+1) + ".wcnf"
        full_osacpath = osacpath + str(i+1) + ".cfn"
        # Get GNN solution and execution time
        gnn_lb, gnn_time = run_inference(full_filepath, ckpt_path)
        # Get VAC, FDAC and EDAC solutions and execution times
        vac_lb, vac_time, edac_lb, edac_time, fdac_lb, fdac_time = run_toulbar2(full_filepath)

        with open(full_osacpath, 'r') as file:
            cfn_model = json.load(file)
        osac_solution = cfn_model.get("optimal_solution", {}).get("c0")

        if osac_solution != 0:
            # Gap calculation
            gap_gnn  = abs(gnn_lb - osac_solution) / osac_solution *100
            gap_vac  = abs(vac_lb - osac_solution) / osac_solution *100
            gap_edac = abs(edac_lb - osac_solution) / osac_solution *100
            gap_fdac = abs(fdac_lb - osac_solution) / osac_solution *100
            gaps_gnn.append(gap_gnn)
            gaps_vac.append(gap_vac)
            gaps_edac.append(gap_edac)
            gaps_fdac.append(gap_fdac)

            times_gnn.append(gnn_time)
            times_vac.append(vac_time)
            times_edac.append(edac_time)
            times_fdac.append(fdac_time)
        else :
            osac_zero += 1
        
    print(f"There is {osac_zero} problems with osac=0.")
    
    # Function to print mean, variance and std
    def print_stats(name, data):
        print(f"{name} - Mean: {np.mean(data):.4f}, Variance: {np.var(data):.4f}, Std: {np.std(data):.4f}")

    print_stats("gap GNN", gaps_gnn)
    print_stats("gap VAC", gaps_vac)
    print_stats("gap EDAC", gaps_edac)
    print_stats("gap FDAC", gaps_fdac)

    print_stats("time GNN", times_gnn)
    print_stats("time VAC", times_vac)
    print_stats("time EDAC", times_edac)
    print_stats("time FDAC", times_fdac)

    # Gap distributions all together
    plt.hist(gaps_vac, bins=10, alpha=0.5, label="VAC")
    plt.hist(gaps_edac, bins=10, alpha=0.5, label="EDAC")
    plt.hist(gaps_fdac, bins=10, alpha=0.5, label="FDAC")
    plt.hist(gaps_gnn, bins=10, alpha=0.5, label="GNN")
    plt.xlabel("Gap")
    plt.ylabel("frequency")
    plt.legend()
    plt.title("Gap distribution all together")
    plt.savefig("results/gap_distribution_all.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    # GNN gap distribution
    plt.hist(gaps_gnn, bins=10, alpha=0.7, color="blue")
    plt.xlabel("Gap")
    plt.ylabel("Frequency")
    plt.title(f"Gap distribution - GNN - {filepath}")
    plt.savefig("results/gap_distribution_gnn.png", dpi=300, bbox_inches="tight")
    plt.close()

    # VAC gap distribution
    plt.hist(gaps_vac, bins=10, alpha=0.7, color="orange")
    plt.xlabel("Gap")
    plt.ylabel("Frequency")
    plt.title(f"Gap distribution - VAC - {filepath}")
    plt.savefig("results/gap_distribution_vac.png", dpi=300, bbox_inches="tight")
    plt.close()

    # EDAC gap distribution
    plt.hist(gaps_edac, bins=10, alpha=0.7, color="green")
    plt.xlabel("Gap")
    plt.ylabel("Frequency")
    plt.title(f"Gap distribution - EDAC - {filepath}")
    plt.savefig("results/gap_distribution_edac.png", dpi=300, bbox_inches="tight")
    plt.close()

    # FDAC gap distribution
    plt.hist(gaps_fdac, bins=10, alpha=0.7, color="red")
    plt.xlabel("Gap")
    plt.ylabel("Frequency")
    plt.title(f"Gap distribution - FDAC - {filepath}")
    plt.savefig("results/gap_distribution_fdac.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    file = "instance_1000c_10v_"
    filepath = "data/wcnf/" + file
    osacpath = "data/osac_solution/" + file
    ckpt_path = "models/last_6l_128hd.ckpt"
    nb_file = 25
    toulbar_vs_gnn(filepath, ckpt_path, osacpath, nb_file)

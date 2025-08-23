import random

def cnf2wcnf(filename, filepath, max_weight=10):
    ''' Convert a .cnf file into a .wcnf.
    Parameters:
        - filename : name of the file without extension or direction
        - filepath : full path of the file
        - max_weight : weights of .wcnf will be between [1, max_weight]
    Output:
        - new_filepath : full path of the wcnf file we created
    '''
    new_filepath = "data/wcnf/" + filename + ".wcnf"                   # new .wcnf file is in folder data/wcnf/
    with open(filepath,"r") as file, open(new_filepath,'w') as new_file:
        for line in file:
            if line.startswith("p"):
                new_line = line.rstrip('\n') + f' 100000000000.000\n'  # Add the maximal weight
                new_file.write(f"{new_line.replace("cnf", "wcnf")}")   # Replace cnf by wcnf
            elif line.startswith("c"):
                new_file.write(f"{line}")
            else:
                random_weight = random.randint(1, max_weight)          # Add a random weight at the beggining of each line
                new_file.write(f"{random_weight} {line}")
    return new_filepath
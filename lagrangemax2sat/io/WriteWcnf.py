import torch
from pathlib import Path


class WriteWcnf():
    """ Class to write a Weighted Max 2-SAT problem into a .wcnf. """
    def __init__(self, filepath, new_w, new_w_min, unary_lagrangian, binary_lagrangian):
        self.filepath = filepath
        self.N, self.D = new_w.shape[0], new_w.shape[2]
        self.new_w = new_w
        self.new_w_min = new_w_min
        self.unary_lagrangian = unary_lagrangian
        self.binary_lagrangian = binary_lagrangian

    @property
    def new_wcnf_filepath(self):
        path = Path(self.filepath)
        filename = str(path.stem)
        return "data/wcnf_reformulate/" + filename + "_reformulate.wcnf"  # new file in folder data/wcnf_reformulate/
    
    def is_writable(self):
        unary_int = torch.all(self.unary_lagrangian == self.unary_lagrangian.to(torch.int))
        binary_int = torch.all(self.binary_lagrangian == self.binary_lagrangian.to(torch.int))
        if unary_int and binary_int:
            return True
        return False

    def write(self):
        ''' Creates a new wcnf file corresponding to the reformulated problem. '''
        with open(self.filepath, "r") as file, open(self.new_wcnf_filepath, 'w') as new_file:
            # Comments at the beggining of the file.
            new_file.write(f"c this problem is a reformulation of the {self.filepath} problem\n")
            for line in file:
                if line.startswith("p"):
                    p, wncf, nb_var, _, max, = line.strip().split()
                    num_clauses = str(int(torch.count_nonzero(self.new_w) + 1))
                    new_line = p + " " + wncf + " " + nb_var + " " + num_clauses + " " + max + "\n"
                    new_file.write(f"{new_line}")
                elif line.startswith("c"):
                    new_file.write(f"{line}")
            # Zero arity weight.
            new_file.write(f"{int(self.new_w_min)} 0\n")
            # Unary weights.
            for x in range(self.N):
                for a in range(self.D):
                    weight = int(self.new_w[x, x, a, a])
                    if weight != 0:
                        new_line = f"{weight} {x + 1} 0\n" if a == 0 else f"{weight} -{x + 1} 0\n"
                        new_file.write(new_line)
            # Binary weights.
            for x in range(1, self.N):
                for y in range(x):
                    for a in range(self.D):
                        for b in range(self.D):
                            weight = int(self.new_w[x, y, a, b])
                            if weight != 0:
                                if a == 0 and b == 0:
                                    new_file.write(f"{weight} {x + 1} {y + 1} 0\n")
                                elif a == 0 and b == 1:
                                    new_file.write(f"{weight} {x + 1} -{y + 1} 0\n")
                                elif a == 1 and b == 1:
                                    new_file.write(f"{weight} -{x + 1} -{y + 1} 0\n")
                                else:
                                    new_file.write(f"{weight} -{x + 1} {y + 1} 0\n")
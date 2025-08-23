from pathlib import Path

from lagrangemax2sat.io.cnf2wcnf import cnf2wcnf


class ReadWcnf():
    ''' Class to open a .wcnf and find its number of variables.
    If the file is a .cnf, it converts it into a .wcnf.'''
    def __init__(self, filepath_raw, D=2):
        # TODO: is it possible to infer D as well ?
        self.filepath_raw = filepath_raw                        # full path to input file (.wcnf or .cnf)
        self.wcnf_filepath = self.test_wcnf(self.filepath_raw)  # full path to .wcnf file (already existing or created if necessary)
        self.N, self.clauses_size = self.find_data_size()       # number of variables and clauses of the problem
        self.D = D                                              # size of the domain of variables

    @staticmethod
    def test_wcnf(filepath):
        ''' Checks whether the input file is .wcnf otherwise calls cnf_to_wcnf function. Returns .wcnf file name. '''
        path = Path(filepath)
        filetype = str(path.suffix)                             # extension of the file to test
        filename = str(path.stem)                               # name of the file to test, without direction and extension
        if filetype == ".cnf":
            return cnf2wcnf(filename, filepath)                 # return the path of the .wcnf file created
        elif filetype == ".wcnf":
            return filepath                                     # return the input, since it is already a .wcnf file
        assert False, "File should be .wcnf or .cnf"

    def find_data_size(self):
        ''' Returns the number of variables of the problem. '''
        with open(self.wcnf_filepath, "r") as file:
            for line in file:
                if line.startswith("p"):
                    if len(line.strip().split()) == 5:
                        _, _, data_size, clauses_size, _, = line.strip().split()
                        data_size = int(data_size)
                        clauses_size = int(clauses_size)
        return data_size, clauses_size
    
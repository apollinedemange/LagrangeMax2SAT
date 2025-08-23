import json
from pathlib import Path

class WriteCfn():
    """ Class to write a Weighted Max 2-SAT problem into a .cfn. """
    def __init__(self, filepath, new_w, new_w_min):
        self.filepath = filepath
        self.N, self.D = new_w.shape[0], new_w.shape[2]
        self.new_w = new_w
        self.new_w_min = new_w_min

    @property
    def cfn_filepath(self):
        path = Path(self.filepath)
        filename = str(path.stem)
        return "data/cfn/" + filename + ".cfn"  # new file .cfn is in folder data/cfn/

    def write(self):
        ''' Creates a cfn file corresponding to the reformulated problem. '''
        # Definition of the model
        cfn_model = {
            "problem": {"name":f"cfn_file_of_{self.filepath}", "mustbe": "<100000000000.000"},
            "variables": {},
            "functions": {}
        }
        # Add variables
        domain = [f"pos{a}" for a in range(self.D)]
        for x in range(self.N):
            cfn_model["variables"][f"Var{x+1}"] = domain
        # Add arity zero cost
        cfn_model["functions"][f"v0"] = {"scope":[],"costs": [float(self.new_w_min)]}
        # Add unary costs
        for x in range(self.N):
            unary_w = [float(self.new_w[x,x,a,a]) for a in range(self.D)]
            cfn_model["functions"][f"v{x+1}"] = {"scope":[f"Var{x+1}"],"costs":unary_w}
        # Add binary costs
        for x in range(self.N):
            for y in range(x):
                list_new_w = self.new_w[x, y].reshape(-1).tolist()
                binary_w = [float(w) for w in list_new_w]
                cfn_model["functions"][f"v{x+1}v{y+1}"] = {"scope": [f"Var{x+1}", f"Var{y+1}"],"costs": binary_w}
        # Writing inside the file
        with open(self.cfn_filepath, 'w') as file:
            json.dump(cfn_model, file, indent=2)
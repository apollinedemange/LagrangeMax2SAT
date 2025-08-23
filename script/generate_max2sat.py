''' Convert 2SAT problems (.cnf) in Max2SAT problems (.wcnf) '''

import argparse

from lagrangemax2sat.io.cnf2wcnf import cnf2wcnf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .wcnf or .cnf files")
    parser.add_argument("filename",help="The name of the file to convert, without extension or direction")
    parser.add_argument("filepath",help="The full path of the cnf file to convert")
    args = parser.parse_args()
    print(cnf2wcnf(args.filename, args.filepath))
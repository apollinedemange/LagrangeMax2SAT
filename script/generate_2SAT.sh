#!/bin/bash

# Generation of 2SAT problems in .cnf files with Power Law Random SAT Generator
# and conversion in Weighted Max 2-SAT and generation of OSAC solutions

# Configuration
NUM_FILES=25                                          # number of files to generate
NUM_VARS=10                                            # number of variables
NUM_CLAUSES=1000                                         # number of clauses
D=2                                                     # size of clauses
GEN_TYPE=u                                              # uniform generation
UNIQUE=1                                                # differents variables in the same clause
OUTPUT_DIR="data/cnf"                                   # folder containing generated files
CONVERSION_TO_WCNF_SCRIPT="script/generate_max2sat.py"  # path to script which converts 2SAT problems in Max2SAT problems
SOLVER_SCRIPT="script/generate_lagrange.py"             # path to script which solves Max2SAT problem and stores solutions
START_INDEX=1
END_INDEX=$((START_INDEX + NUM_FILES - 1))


# Create a folder
mkdir -p $OUTPUT_DIR

# Generation of the instances
for i in $(seq $START_INDEX $END_INDEX); do
    FILENAME="${OUTPUT_DIR}/instance_${NUM_CLAUSES}c_${NUM_VARS}v_$i"
    SEED=$((RANDOM))
    echo "Generation of 2SAT problem in $FILENAME.cnf"
    ~/Power-Law-Random-SAT-Generator/CreateSAT -g $GEN_TYPE -v $NUM_VARS -c $NUM_CLAUSES -k $D -f $FILENAME -u $UNIQUE -s $SEED
    
    echo "Generation of Max2SAT problem (.wcnf) of $FILENAME.cnf"
    WCNF_FILENAME=$(uv run $CONVERSION_TO_WCNF_SCRIPT "instance_${NUM_CLAUSES}c_${NUM_VARS}v_$i" "$FILENAME.cnf")

    echo "Generation of optimal lagrangians (.cfn) of $FILENAME.cnf"
    uv run $SOLVER_SCRIPT "$WCNF_FILENAME"
done

echo "End of generation of $NUM_FILES random files in the folder '$OUTPUT_DIR'."


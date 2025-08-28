# LagrangeMax2SAT

This project aims at computing lower bounds for the Weighted MAX-2-SAT problem by learning Lagrangian multipliers with anisotropic Graph Neural Networks, combining combinatorial optimization and deep learning techniques.

## Overview

Weighted Max-2SAT is a special case of the MaxSAT problem where each clause contains exactly two literals and are associted with a cost. It is NP-hard and appears in many optimization and AI contexts.

This project proposes a novel perspective: finding Weighted Max-2SAT lower bound by predicting Lagrangians multipliers of dual problem.

## Deployed Models

We provide a pretrained model trained on Weighted Max-2SAT instances of identical size.

| Model Name              | Variables | Clauses | Layers | Embedding size | Description                                                   |
|-------------------------|-----------|---------|--------|----------------|---------------------------------------------------------------|
| `400c_50v_6l_128hd`     | 50        | 400     | 6      | 128            | Training big Weighted Max-2SAT problems and small model |

## Inference

You can run inference on new wcnf files to predict lower bound approximately.

### Run Inference 

```bash
python lagrangemax2sat/inference.py --ckpt_path models/mode_name.ckpt \
                                    --filepath data/wcnf/example.wcnf \
```

## Weighted Max-2SAT Problem Generator
```bash
script/generate_2SAT.sh
```

This script allows you to generate random Weighted Max-2SAT problems and their optimal solution with customizable parameters.

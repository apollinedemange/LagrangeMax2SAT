# LagrangeMax2SAT

This project aims at computing lower bounds for the Weighted MAX-2-SAT problem by learning Lagrangian multipliers with anisotropic Graph Neural Networks, combining combinatorial optimization and deep learning techniques.

## Overview

Weighted Max-2SAT is a special case of the MaxSAT problem where each clause contains exactly two literals and are associted with a cost. It is NP-hard and appears in many optimization and AI contexts.

This project proposes a novel perspective: finding Weighted Max-2SAT lower bound by predicting Lagrangians multipliers of dual problem.

## Deployed Models

We provide three pretrained models of different sizes and trained on different Weighted Max-2SAT instance configurations.

| Model Name              | Variables | Clauses | Description |
|-------------------------|-----------|---------|-------------|
| `50c_10v_12l_256hd`     | 10        | 50      | Training on easy Weighted Max-2SAT problems and large model |
| `400c_50v_12l_256hd`    | 50        | 400     | Training on bigger Weighted Max-2SAT problems and large model |
| `400c_50v_6l_128hd`     | 50        | 400     | Training on bigger Weighted Max-2SAT problems and small model |

Each model can be used for inference with the corresponding config and checkpoint

## Inference

You can run inference on new wcnf files to predict lower bound approximately.

### Run Inference
In the case of Dense Encoding approach, 

```bash
python lagrangemax2sat/inference.py --ckpt_path models/mode_name.ckpt \
                                    --filepath data/wcnf/example.wcnf \
```

## Weighted Max-2SAT Problem Generator
```bash
script/generate_2SAT.sh
```

This script allows you to generate random Weighted Max-2SAT problems and their optimal solution with customizable parameters.
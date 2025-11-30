# DiffLifting
This repository is the official implementation of the methods in the publication **Differentiable Lifting for Topological Neural Networks**. The code runs on python 3.10 and uses CUDA-12.1. 

<p align="center">
  <img src="static/difflift.png" />
</p>

## Description
We propose $\partial\text{lift}$ (DiffLift), a general framework for learning graph liftings to hypergraphs and cellular- and simplicial complexes in an end-to-end fashion. In particular, our approach leverages learned vertex-level latent representations to identify and parameterize distributions over candidate higher-order cells for inclusion. This results in a scalable model which can be readily integrated into any TNN. Our experiments show that $\partial\text{lift}$ outperforms existing lifting methods on multiple benchmarks for graph and node classification across different TNN architectures. Notably, our approach leads to gains of up to 45\% over static liftings, including both connectivity- and feature-based ones.

## How to run
```bash
docker build .
```

## Experiments
Graph classification experiments are in `main_graph_classification.py`, and node classification experiments are in `main.py`.

## License
This software is provided under the MIT License.


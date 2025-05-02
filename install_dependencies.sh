#!/bin/sh
pip install torch==2.5.1
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.2.1+cu121.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.2.1+cu121.html
pip install torch-geometric==2.5.3
pip install ogb==1.3.6
pip install rdkit
pip install scikit-learn>=1.3
pip install deepchem==2.8.0 # only used for scaffold splitting

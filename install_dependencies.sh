#!/bin/sh
pip install torchvision==0.20.0 # invoked by deepchem dependency
# torchvision e.g. 0.21 can be an issue if left at the higher version after downgrading torch to 2.5.1 
# must downgrade torch to 2.5.1 to stay compatible with ogb 1.3.6, see https://github.com/snap-stanford/ogb/issues/497
pip install torch==2.5.1
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.2.1+cu121.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.2.1+cu121.html
pip install torch-geometric==2.5.3
pip install ogb==1.3.6
pip install rdkit
pip install scikit-learn>=1.3
pip install deepchem==2.8.0 # only used for scaffold splitting

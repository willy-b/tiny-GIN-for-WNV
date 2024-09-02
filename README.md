# tiny-GIN-for-PCBA-577-WNV

**WORK IN PROGRESS REPO, NOT FOR ANY COMPETITION**

Trying to apply a similar network to https://github.com/willy-b/tiny-GIN-for-ogbg-molhiv/ to a West Nile Virus related dataset.

It is fun to participate in the Open Graph Benchmark ogbg-molhiv competition to build a predictor for identifying molecules with antiviral activity for HIV ( https://web.archive.org/web/20240822032633/https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molhiv ) for which https://github.com/willy-b/tiny-GIN-for-ogbg-molhiv/ got 22nd place, but many antivirals for HIV are already available.

It is interesting to apply similar techniques to widespread viruses for which there are not yet any approved antiviral drugs available.
West Nile Virus (WNV) would be an example of such a virus.

OGB has no dataset available at this time for West Nile Virus, but PubChem BioAssay from NCBI has many.

I will start by considering the West Nile Virus related dataset "HTS to identify Inhibitors of West Nile Virus NS2bNS3 Proteinase" and predicting such NS2bNS3 protease inhibitors as they were considered likely at the time of that datasets publication to be WNV antiviral drug candidates:
![PubChem BioAssay AID 577 - HTS to identify Inhibitors of West Nile Virus NS2bNS3 Proteinase](pcba_aid_577_description_snippet.png)

We can put that into OGB format for consistency and test a similar tiny Graph Isomorphism Network (GIN) on it.

NOTE THERE IS NO COMPETITION ASSOCIATED WITH THIS DATASET THAT I AM AWARE OF, THIS IS JUST FOR INTEREST IN THE PROBLEM UNLIKE THE OGBG-MOLHIV REPO.

# Trying it out

(work in progress, no official statistics available yet, splits are randomized per run, and reporting on test split is disabled by hand for now)

1. Install dependencies (run `install_dependencies.sh` this comes with or commands below):

```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.2.1+cu121.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.2.1+cu121.html
pip install torch-geometric # I'm using 2.5.3 right now
pip install ogb # I'm using 1.3.6 right now
```

2. Run this script `python main_gin.py` (I'm using python 3.10.12 but should be flexible)

# References

### PubChem BioAssay: HTS to identify Inhibitors of West Nile Virus NS2bNS3 Proteinase

PubChem Identifier: AID 577
URL: https://pubchem.ncbi.nlm.nih.gov/bioassay/577

### PyTorch Geometric

- Fey, Matthias and Lenssen, Jan E. Fast Graph Representation Learning with PyTorch Geometric. ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019. (Graph Isomorphism Network (GIN) implementation used)

### GraphNorm

- Tianle Cai and Shengjie Luo and Keyulu Xu and Di He and Tie-Yan Liu and Liwei Wang. GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training. Proceedings of the 38th International Conference on Machine Learning, 2021.

### Graph Isomorphism Network (GIN)

- Xu, Keyulu and Hu, Weihua and Leskovec, Jure and Jegelka, Stefanie. How Powerful Are Graph Neural Networks? International Conference on Learning Representations, 2019. https://openreview.net/forum?id=ryGs6iA5Km , https://arxiv.org/pdf/1810.00826 . (Graph Isomorphism Network (GIN) original paper)

### Open Graph Benchmark (OGB)

(note only the OGB data format and evaluator are used here, this is NOT an ogb dataset and not to be confused with their ogbg-molpcba which is 128 selected pubchem bioassays NOT including AID 577 we consider here)

- Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure. Open Graph Benchmark: Datasets for Machine Learning on Graphs. arXiv preprint arXiv:2005.00687, 2020.

## Acknowledgements

Credit to Stanford XCS224W (certificate of completion with link to course and program info can be found at https://digitalcredential.stanford.edu/check/27C7D07B3EF8511E8B9BBA720E9A7C51BE3CBC49F80B7F32D1839B5D24442250U3BuVnNsVW9ldVdCQURiRXFZSXo2d3ZlOW5BSDJWVzUrVit1VGFQRHN2UVhVYjQ3 ) for getting me started with Graph Neural Networks.

Credit to "Massively Multitask Networks for Drug Discovery" (Ramsundar et al 2015, https://arxiv.org/abs/1502.02072 ) and "Discovery of a structural class of antibiotics with explainable deep learning" (Wong et al 2023, https://www.nature.com/articles/s41586-023-06887-8 ) for inspiring me to pursue this challenge and to the authors of Wong et al 2023 in particular for making the world a better place by discovering new antibiotics for MRSA using deep learning.

Credit to "Keeping Neural Networks Simple by Minimizing the Description Length of the Weights" ( Hinton et al 1993, https://www.cs.toronto.edu/~fritz/absps/colt93.pdf ), and "Representation Learning on Graphs with Jumping Knowledge Networks" ( Xu et al 2018, https://arxiv.org/abs/1806.03536 ) for giving me ideas that helped me reduce the parameter count.

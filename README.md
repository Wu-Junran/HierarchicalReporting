#A Simple yet Effective Method for Graph Classification
This repository is the official PyTorch implementation of the experiments in the following paper:

Junran Wu, Shangzhe Li$^+$, Jianhao Li, Yicheng Pan\*, Ke Xu\*. A Simple yet Effective Method for Graph Classification. IJCAI 2022 (Long Oral).

[IJCAI 2022](https://www.ijcai.org/proceedings/2022/497) [arXiv](https://arxiv.org/abs/2206.02404)

## Installation

Install PyTorch following the instuctions on the [official website](https://pytorch.org/). The code has been tested over Python 3.7.4, PyTorch 1.3.1 and CUDA 10.1.


## Data preparation
Experiment datasets are adopted from [GIN-repo](https://github.com/weihua916/powerful-gnns).


* Unfold data

```
tar -xvzf datasets.tar.gz
```

* Load graphs from original dataset file for better code.

```
python load_graph.py
```

* Coding tree generation (Transform graphs to encoding trees).

```
python treeGeneration.py
```

## Test with tree kernel
Default 10-fold corss-validation is conducted with ```cross_val_score``` in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html).

```
usage: treeKernel.py [-h] [-d DATASET] [-k {2,3,4,5}]

Tree kernel with SVM for whole-tree classification

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        name of dataset (default: None, search all datasets)
  -k {2,3,4,5}, --tree_depth {2,3,4,5}
                        the depth of coding tree (default: None, search all depthes)
```


## Hierarchical Reporting Network

The default parameters are not the best performing-hyper-parameters used to reproduce our results in the paper. Hyper-parameters need to be specified through the commandline arguments. Please refer to our paper for the details of how we set the hyper-parameters.

To learn hyper-parameters to be specified, please see

```
usage: hrn_args.py [-h] [-d DATASET] [-k TREE_DEPTH] [-b BATCH_SIZE]
                   [-e EPOCHS] [-lr LEARNING_RATE]
                   [--iters_per_epoch ITERS_PER_EPOCH] [-s SEED]
                   [-fi FOLD_IDX] [-lm NUM_MLP_LAYERS] [-hd HIDDEN_DIM]
                   [-fd FINAL_DROPOUT] [-tp {root,sum,average}] [-gd]
                   [--device DEVICE]

PyTorch tree convolutional neural net for whole-tree classification

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        name of dataset (default: MUTAG)
  -k TREE_DEPTH, --tree_depth TREE_DEPTH
                        the depth of coding tree (default: 2)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        input batch size for training (default: 32)
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train (default: 350)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate (default: 0.01)
  --iters_per_epoch ITERS_PER_EPOCH
                        number of iterations per each epoch (default: 50)
  -s SEED, --seed SEED  random seed for splitting the dataset into 10
                        (default: 0)
  -fi FOLD_IDX, --fold_idx FOLD_IDX
                        the index of fold in 10-fold validation. Should be less than 10.
  -lm NUM_MLP_LAYERS, --num_mlp_layers NUM_MLP_LAYERS
                        number of layers for MLP EXCLUDING the input one
                        (default: 2). 1 means linear model.
  -hd HIDDEN_DIM, --hidden_dim HIDDEN_DIM
                        number of hidden units (default: 64)
  -fd FINAL_DROPOUT, --final_dropout FINAL_DROPOUT
                        final layer dropout (default: 0.5)
  -tp {root,sum,average}, --tree_pooling_type {root,sum,average}
                        Pooling for over nodes in a tree: root, sum or average
  -gd, --global_degree  add the degree of nodes to features
  --device DEVICE       which gpu to use if any (default: 0)
```

An example of training process is as follows:

```
python hrn_args.py -d IMDBBINARY -k 3 -b 32 -hd 32 -tp average -gd
```

## Cross-validation strategy in the paper

The cross-validation in our paper only uses training and validation sets (no test set) due to small dataset size. Specifically, after obtaining 10 validation curves corresponding to 10 folds, we first took average of validation curves across the 10 folds (thus, we obtain an averaged validation curve), and then selected a single epoch that achieved the maximum averaged validation accuracy. Finally, the standard devision over the 10 folds was computed at the selected epoch.

## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work. </br>

```BibTex
@inproceedings{wu2022simple,
  title     = {A Simple yet Effective Method for Graph Classification},
  author    = {Wu, Junran and Li, Shangzhe and Li, Jianhao and Pan, Yicheng and Xu, Ke},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {3580--3586},
  year      = {2022},
  month     = {7},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2022/497},
  url       = {https://doi.org/10.24963/ijcai.2022/497},
}
```




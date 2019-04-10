QANet-pytorch
===
# Introduction

An implementation of [QANet](https://arxiv.org/pdf/1804.09541.pdf) with PyTorch. 
refer to [hengruo/QANet-pytorch](https://github.com/hengruo/QANet-pytorch)
Any contributions are welcome!

# cowork-rules
* branch name: <type>/<brief>
    > type: feature, bug, etc.
    > brief: brief content
    e.g., feature/init
* commit message: <[type]><description>
    > type: FEATURE, BUG, CLEANUP, etc.
    > description: description the changes in this commit
    e.g., [FEATURE]init  

# Usage
1. Install pytorch 0.4 for Python 3.6+
2. Run `pip install -r requirements.txt` to install python dependencies.
3. Run `download.sh` to download the dataset.
4. Run `python preproc.py` to build tensors from the raw dataset.
5. Run `python main.py --mode train` to train the model. After training, `log/model.pt` will be generated.
6. Run `python main.py --mode test` to test an pretrained model. Default model file is `log/model.pt`

## Structure
preproc.py: builds input tensors.
main.py: program entry; functions about training and testing.
models.py: QANet structure.
config.py: configurations.

## Differences from the paper
1. The paper doesn't mention which activation function they used. I use relu.
2. I don't set the embedding of `<UNK>` trainable.
3. The connector between embedding layers and embedding encoders may be different from the implementation of Google, since the description in the paper is inconsistent (residual block can't be used because the dimensions of input and output are different) and they don't say how they implemented it.

## TODO
- [x] Reduce memory usage
- [ ] **Improve converging speed (to reach 60 F1 scores in 1000 iterations)**
- [ ] Reach state-of-art scroes of the original paper
- [ ] Performance analysis
- [ ] Test on SQuAD 2.0

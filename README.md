QANet-pytorch
===
# Introduction

An implementation of [QANet](https://arxiv.org/pdf/1804.09541.pdf) with PyTorch.   
refer to [hengruo/QANet-pytorch](https://github.com/hengruo/QANet-pytorch)  
Any contributions are welcome!

# cowork-rules
* branch name: \<type\>/\<brief\>
    > type: feature, bug, etc.  
    > brief: brief content  
    
    e.g., feature/init
* commit message: \<[type]\>\<description\>
    > type: FEATURE, BUG, CLEANUP, etc.  
    > description: description the changes in this commit  
    
    e.g., [FEATURE]init  

# Usage
1. Install pytorch 1.0+ for Python 3.6+
2. Run `pip install -r requirements.txt` to install python dependencies.
3. Run `download.sh` to download the dataset.
4. Start Coding

# Preprocessing data in paper
> We use the NLTK tokenizer to preprocess the data. The maximum
context length is set to 400 and any paragraph longer than that would 
be discarded. During training, we batch the examples by length and 
dynamically pad the short sentences with special symbol <PAD>. The 
maximum answer length is set to 30. We use the pretrained 300-D 
word vectors GLoVe (Pennington et al., 2014), and all the 
out-of-vocabulary words are replace with <UNK>, whose embedding is 
updated during training. Each character embedding is randomly 
initialized as a 200-D vector, which is updated in training as well. 
1. NLTK tokenizer
    * use spacy instead
2. The maximum context length is set to 400 and any paragraph longer than that would be discarded
3. The maximum answer length is set to 30  and any answer longer than that would be discarded
4. pretrained 300-D word vectors GLoVe
5. character embedding is randomly initialized as a 200-D vector

> We generate two additional augmented datasets obtained from Section 3, 
which contain 140K and 240K examples and are denoted as “data 
augmentation × 2” and “data augmentation × 3” respectively, including 
the original data.

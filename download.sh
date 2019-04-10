#!/usr/bin/env bash

# Download SQuAD
SQUAD_DIR=./data/squad
mkdir -p $SQUAD_DIR
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/dev-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O $SQUAD_DIR/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O $SQUAD_DIR/dev-v2.0.json

# Download GloVe
GLOVE_DIR=./data/glove
mkdir -p $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $GLOVE_DIR/glove.840B.300d.zip
unzip $GLOVE_DIR/glove.840B.300d.zip -d $GLOVE_DIR

python -m spacy download en

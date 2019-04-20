#!/usr/bin/env bash

# Download SQuAD
SQUAD_DIR=./data/squad
mkdir -p $SQUAD_DIR
mkdir -p $SQUAD_DIR/v1.1
mkdir -p $SQUAD_DIR/v2.0
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/v1.1/train.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/v1.1/dev.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O $SQUAD_DIR/v2.0/train.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O $SQUAD_DIR/v2.0/dev.json

# Download GloVe
GLOVE_DIR=./data/glove
mkdir -p $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $GLOVE_DIR/glove.840B.300d.zip
unzip $GLOVE_DIR/glove.840B.300d.zip -d $GLOVE_DIR

python -m spacy download en

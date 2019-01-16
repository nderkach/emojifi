#!/bin/bash

#download pretrained gloVe embeddings from Stanford NLP
#wget http://nlp.stanford.edu/data/glove.6B.zip
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip

#unzip
unzip glove.27B.zip

#we only need the 50 dimensional embedding
rm glove.6B.100*
rm glove.6B.200*
rm glove.6B.300*
rm glove.twitter.27B.25*
rm glove.twitter.27B.100*
rm glove.twitter.27B.200*
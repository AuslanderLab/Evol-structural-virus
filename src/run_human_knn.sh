#!/bin/bash

# the number is the hyperparameter for the KNN classifier
# arg number 1 needs to not end with /
# arg number 1 is outpath
# arg number 2 is the input file
# arg number 3 is the family file
python3 ./src/human_knn.py 5  $1 $2 $3

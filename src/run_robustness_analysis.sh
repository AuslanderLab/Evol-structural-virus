#!/bin/bash

pwd
# arg number 2 needs to not end with /
# arg 1 is the removal percentage
# arg 2 is the out path 
# arg 3 is the input df
# arg 4 is  family annotation
# 5 %
cd results
mkdir percent_5
cd percent_5
python3 ../../src/robustness.py 11 0.05 . $1 $2
# 10% 
cd .. 
mkdir percent_10
cd percent_10
python3 ../../src/robustness.py 11 0.1 . $1 $2
# 25%
cd .. 
mkdir percent_25
cd percent_25
python3 ../../src/robustness.py 11 0.25 . $1 $2
# 50%
cd .. 
mkdir percent_50
cd percent_50
python3 ../../src/robustness.py 11 0.5 . $1 $2
# 75%
cd .. 
mkdir percent_75
cd percent_75
python3 ../../src/robustness.py 11 0.75 . $1 $2
# 90%
cd .. 
mkdir percent_90
cd percent_90
python3 ../../src/robustness.py 11 0.9 . $1 $2

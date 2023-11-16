#!/bin/bash

cd src

echo "### Running the training script ###"
python training_lgbm.py
echo "### Running the training script ###"
python scoring_lgbm.py

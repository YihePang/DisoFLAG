# @Author: Yihe Pang
# @Date:   2023-06-12 11:39:36
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2023-06-14 22:38:04

#!/bin/bash
# run.sh -i ./data/test.fasta -o p

# echo "$1"   # -i
# echo "$2"   #data/test.fasta
# echo "$3"   # -o
# echo "$4"   # p/b  prob or binary


echo "Running predictor ......"
echo "Input file:  $2"
python run_pretrained_PLM.py $2
echo "1 -- Embeddings prepared -- "


echo "2 -- Results prepared -- "
python run_pred.py -i $2 -o $4
echo "3 -- Finished ---"
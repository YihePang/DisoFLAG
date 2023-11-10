# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2023-06-13 09:51:38
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2023-06-13 12:33:05
import sys
import os
import numpy as np
import datetime
import random

from model_running import FLAG_model_running


import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='predictor_configs')
parser.add_argument('--input','-i',type=str, default = "./data/test.fasta",required=True,help="input fasta file")
parser.add_argument('--output','-o',type=str, default = "p",required=True,help="Results type (p or b) :'p'-propensity score, 'b'-binary result")



if __name__ == '__main__':
    args = parser.parse_args()
    # print(args.input)
    # print(args.output)

    input_data_file = args.input
    output_results_path = './temp/results/'
    if not os.path.isdir(output_results_path):
    	os.makedirs(output_results_path)

    if args.output == 'p':
    	output_file_name = output_results_path+'DisoFLAG_prediction_propensity.txt'
    elif args.output == 'b':
    	output_file_name = output_results_path+'DisoFLAG_prediction_binary.txt'

    output_type = args.output

    FLAG_model_running(input_data_file, output_file_name, output_type)






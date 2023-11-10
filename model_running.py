# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2023-06-13 10:08:51
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2023-06-14 22:43:30
import numpy as np 
import random
import os
from args import Args_config
from prepare_model_data import data_2_samples, Batches_data
from load_data import load_file_2_data
import torch as t
from torch import nn
from model import Seq2FUN
import sys
from evaluator import write_2_file 

def FLAG_model_running(input_data_file, output_file_name, output_type):
	args = Args_config()

	test_data = data_2_samples(args = args, 
								data_file_name = input_data_file,
								is_slice = True)

	for root, dirs, files in os.walk(args.model_path):
		for one_file in files:
			model_file = args.model_path+'/'+one_file
			# print("model_file:",model_file)	
	model = t.load(model_file, map_location='cpu')
	# print("Model : ------",model)
	model.eval()



	if len(test_data) < args.batch_size:
		input_data = []
		for i in range(args.batch_size):
			if i < len(test_data):
				input_data.append(test_data[i])
			else:
				input_data.append(test_data[0])
	else:
		input_data = test_data

	test_batches = Batches_data(test_data, args.batch_size, is_train=False)	

	IDR_probs = []
	PB_probs = []
	DB_probs = []
	RB_probs = []
	IB_probs = []
	LB_probs = []
	Link_probs  = []
	for t_batch in test_batches:   #一个batch
		t_input_featues = t.tensor(np.array(t_batch.seq_T5_feature))
		# seq_mask
		one_seq_mask = t.tensor(np.array(t_batch.seq_mask), dtype=t.float32)

		one_IDR_probs, one_PB_probs, one_DB_probs, one_RB_probs, one_IB_probs, one_LB_probs, one_Link_probs = model(t_input_featues)

		# logits
		one_IDR_logits = one_IDR_probs * one_seq_mask
		one_PB_logits = one_PB_probs * one_seq_mask
		one_DB_logits = one_DB_probs * one_seq_mask
		one_RB_logits = one_RB_probs * one_seq_mask
		one_IB_logits = one_IB_probs * one_seq_mask
		one_LB_logits = one_LB_probs * one_seq_mask
		one_Link_logits = one_Link_probs * one_seq_mask

		IDR_probs.append(one_IDR_probs.detach().numpy())
		PB_probs.append(one_PB_logits.detach().numpy())
		DB_probs.append(one_DB_logits.detach().numpy())
		RB_probs.append(one_RB_logits.detach().numpy())
		IB_probs.append(one_IB_logits.detach().numpy())
		LB_probs.append(one_LB_logits.detach().numpy())
		Link_probs.append(one_Link_logits.detach().numpy())

	test_file = load_file_2_data(input_data_file)
	write_2_file(test_file, test_data, test_batches, IDR_probs, PB_probs, DB_probs, RB_probs, IB_probs, LB_probs, Link_probs, output_file_name, output_type)






















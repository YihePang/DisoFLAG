# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-05-05 15:47:31
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2023-06-14 22:39:30
import torch
from transformers import T5Tokenizer, T5Model, T5EncoderModel
import re
import os
import numpy as np
import gc
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()



def protT5(model_path,sequences_Example):
	"""
		input: 
		model_path = "Rostlab/prot_bert"
		sequences_Example = ["A E T C Z A O","S K T Z P"]
	"""
	sequences_Example = [' '.join(sequences_Example[0])]

	# print("T5 get input :",sequences_Example)

	tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)

	model = T5EncoderModel.from_pretrained(model_path)
	gc.collect()

	# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	device = torch.device('cpu')


	# if torch.cuda.device_count() > 1:
	# 	model = torch.nn.DataParallel(model)

	model = model.to(device)
		
	model = model.eval()

	sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

	ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, pad_to_max_length=True)

	input_ids = torch.tensor(ids['input_ids']).to(device)
	attention_mask = torch.tensor(ids['attention_mask']).to(device)

	with torch.no_grad():
	    embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]

	embedding = embedding.cpu().numpy()
	# print("embedding:",embedding.shape)

	features = [] 
	for seq_num in range(len(embedding)):
	    seq_len = (attention_mask[seq_num] == 1).sum()
	    seq_emd = embedding[seq_num][1:seq_len]
	    # print("seq_emd:",seq_emd.shape)         #[len,1024]
	    
	    features.append(seq_emd)
	return features

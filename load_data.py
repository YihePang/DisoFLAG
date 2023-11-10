# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2023-02-27 10:43:42
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2023-06-14 22:41:25

import numpy as np 
import os
import random


def load_file_2_data(file_path):
	loadfile = open(file_path,"r") 	
	load_f = []
	for line in loadfile:
		line=line.strip('\n')
		load_f.append(line)
	loadfile.close()

	load_data = []
	for i in range(len(load_f)):
		if i % 2 == 0:
			load_data.append(load_f[i:i+2])    #one data:  [0]--id  [1]--seq   
	# print("load_file: ",file_path,"    data length: ",len(load_data))  
	return load_data

def file_2_data(data_file_name):
	seq_id = []     	
	seq = []        	

	seq_label_IDP = []  	 

	seq_label_F1 = []  	 
	seq_label_F2 = []  	  
	seq_label_F3 = []  	 

	seq_label_F4 = []  	 
	seq_label_F5 = []  	 
	seq_label_F6 = []  	  

	seq_T5_feature = []     
	seq_BERT_feature = []     
	seq_IDP_feature = []     

 	data_list = load_file_2_data(data_file_name)

	for i in range(len(data_list)):
		one_seq_id = data_list[i][0][1:].replace('\r', '')

		seq_id.append(one_seq_id)                               
		seq.append(data_list[i][1].replace('\r', ''))            

		seq_label_IDP.append(['1']*len(data_list[i][1].replace('\r', '')))      

		seq_label_F1.append(['1']*len(data_list[i][1].replace('\r', '')))       
		seq_label_F2.append(['1']*len(data_list[i][1].replace('\r', '')))       
		seq_label_F3.append(['1']*len(data_list[i][1].replace('\r', '')))       

		seq_label_F4.append(['1']*len(data_list[i][1].replace('\r', '')))       
		seq_label_F5.append(['1']*len(data_list[i][1].replace('\r', '')))     
		seq_label_F6.append(['1']*len(data_list[i][1].replace('\r', '')))      

		# embeddings
		T5_embedding_path = './temp/embeddings/T5/'
		
		T5_feature_file = T5_embedding_path + one_seq_id + '.npy'
		
		one_T5_vec = np.load(T5_feature_file,allow_pickle=True)
		one_T5_vec = one_T5_vec.reshape(len(one_T5_vec),-1)
		seq_T5_feature.append(one_T5_vec)  

	return np.array(seq_id),np.array(seq),np.array(seq_label_IDP),np.array(seq_label_F1),np.array(seq_label_F2),np.array(seq_label_F3),np.array(seq_label_F4),np.array(seq_label_F5),np.array(seq_label_F6),np.array(seq_T5_feature)



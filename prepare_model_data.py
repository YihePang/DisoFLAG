# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2023-02-27 10:43:18
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2023-06-14 22:46:51
import numpy as np 
import random
from load_data import load_file_2_data, file_2_data


def residue_mask(seq_label):  
	mask = []
	for s in range(len(seq_label)):   
		lable_mask = []
		for i in range(len(seq_label[s])):
			if seq_label[s][i] == '1' or seq_label[s][i] == '0':
				lable_mask.append(1)
			else:
				lable_mask.append(0)
		mask.append(lable_mask)
	return mask

 
def sequence_mask(seq):
	mask = []
	for s in range(len(seq)):  
		lable_mask = []
		for i in range(len(seq[s])):
			lable_mask.append(1)
		mask.append(lable_mask)
	return mask


 
def lable_2_value(seq_label):    
	new_seq_label = []
	for s in range(len(seq_label)):
		lable = []
		for i in range(len(seq_label[s])):
			if seq_label[s][i] == '1':
				lable.append(1)
			else:
				lable.append(0)
		new_seq_label.append(lable)
	return new_seq_label


 
def slice_data(seq_id,seq,seq_label_0,seq_label_1,seq_label_2,seq_label_3,seq_label_4,seq_label_5,seq_label_6,seq_T5_feature,res_mask_0,res_mask_1,res_mask_2,res_mask_3,res_mask_4,res_mask_5,res_mask_6,seq_mask,max_seq_length):
	seq_id_new = []
	seq_new = []
	seq_label_0_new = []
	seq_label_1_new = []
	seq_label_2_new = []
	seq_label_3_new = []
	seq_label_4_new = []
	seq_label_5_new = []
	seq_label_6_new = []

	seq_T5_feature_new = []

	res_mask_0_new = []
	res_mask_1_new = []
	res_mask_2_new = []
	res_mask_3_new = []
	res_mask_4_new = []
	res_mask_5_new = []
	res_mask_6_new = []

	seq_mask_new = []

	for i in range(len(seq)): 
		s = 0
		for j in range(int(-(-len(seq[i])//max_seq_length))): 
			if s+max_seq_length >= len(seq[i]):
				end = len(seq[i]) - s
				seq_id_new.append(seq_id[i])
				seq_new.append(seq[i][s:s+end])

				seq_label_0_new.append(seq_label_0[i][s:s+end])
				seq_label_1_new.append(seq_label_1[i][s:s+end])
				seq_label_2_new.append(seq_label_2[i][s:s+end])
				seq_label_3_new.append(seq_label_3[i][s:s+end])
				seq_label_4_new.append(seq_label_4[i][s:s+end])
				seq_label_5_new.append(seq_label_5[i][s:s+end])
				seq_label_6_new.append(seq_label_6[i][s:s+end])

				seq_T5_feature_new.append(seq_T5_feature[i][s:s+end])

				res_mask_0_new.append(res_mask_0[i][s:s+end])
				res_mask_1_new.append(res_mask_1[i][s:s+end])
				res_mask_2_new.append(res_mask_2[i][s:s+end])
				res_mask_3_new.append(res_mask_3[i][s:s+end])
				res_mask_4_new.append(res_mask_4[i][s:s+end])
				res_mask_5_new.append(res_mask_5[i][s:s+end])
				res_mask_6_new.append(res_mask_6[i][s:s+end])

				seq_mask_new.append(seq_mask[i][s:s+end])

			elif s+max_seq_length < len(seq[i]):
				seq_id_new.append(seq_id[i])
				seq_new.append(seq[i][s:s+max_seq_length])

				seq_label_0_new.append(seq_label_0[i][s:s+max_seq_length])
				seq_label_1_new.append(seq_label_1[i][s:s+max_seq_length])
				seq_label_2_new.append(seq_label_2[i][s:s+max_seq_length])
				seq_label_3_new.append(seq_label_3[i][s:s+max_seq_length])
				seq_label_4_new.append(seq_label_4[i][s:s+max_seq_length])
				seq_label_5_new.append(seq_label_5[i][s:s+max_seq_length])
				seq_label_6_new.append(seq_label_6[i][s:s+max_seq_length])

				seq_T5_feature_new.append(seq_T5_feature[i][s:s+max_seq_length])
		
				res_mask_0_new.append(res_mask_0[i][s:s+max_seq_length])
				res_mask_1_new.append(res_mask_1[i][s:s+max_seq_length])
				res_mask_2_new.append(res_mask_2[i][s:s+max_seq_length])
				res_mask_3_new.append(res_mask_3[i][s:s+max_seq_length])
				res_mask_4_new.append(res_mask_4[i][s:s+max_seq_length])
				res_mask_5_new.append(res_mask_5[i][s:s+max_seq_length])
				res_mask_6_new.append(res_mask_6[i][s:s+max_seq_length])

				seq_mask_new.append(seq_mask[i][s:s+max_seq_length])
			s = s+max_seq_length
	return seq_id_new, seq_new, seq_label_0_new,seq_label_1_new,seq_label_2_new,seq_label_3_new,seq_label_4_new, seq_label_5_new, seq_label_6_new, seq_T5_feature_new, res_mask_0_new, res_mask_1_new, res_mask_2_new, res_mask_3_new, res_mask_4_new, res_mask_5_new, res_mask_6_new, seq_mask_new

 
def padding_list(input_list, max_seq_length):
	pad = 0                            # zero-padding
	out_list = []
	if len(input_list) < max_seq_length:
		for i in range(len(input_list)):
			out_list.append(input_list[i])
		for j in range(max_seq_length-len(input_list)):
			out_list.append(pad)
	else:
		for i in range(max_seq_length):
			out_list.append(input_list[i])
	return np.array(out_list)


 
def padding_matrix(input_mat, max_seq_length):
	input_mat = np.array(input_mat)
	mat_dim = input_mat.shape[-1]
	pad_vector = np.zeros([mat_dim])  # zero-padding
	out_mat = []
	if len(input_mat) < max_seq_length:
		for i in range(len(input_mat)):
			out_mat.append(input_mat[i])
		for j in range(max_seq_length-len(input_mat)):
			out_mat.append(pad_vector)
	else:
		for i in range(max_seq_length):
			out_mat.append(input_mat[i])
	return np.array(out_mat)


def seq_lable_padding(seq_label, max_seq_length):
	out_list = []
	for i in range(len(seq_label)):
		new_list = padding_list(seq_label[i], max_seq_length)
		# print(new_list)
		out_list.append(new_list)
	return np.array(out_list)


def seq_feature_padding(seq_feature, max_seq_length):
	out_mat = []
	for i in range(len(seq_feature)):
		new_f = padding_matrix(seq_feature[i], max_seq_length)
		out_mat.append(new_f)
	return np.array(out_mat)


def mask_padding(res_mask,max_seq_length):
	out_list = []
	for i in range(len(res_mask)):
		new_list = padding_list(res_mask[i], max_seq_length)
		# print(new_list)
		out_list.append(new_list)
	return np.array(out_list)


def data_2_samples(args, data_file_name, is_slice):

	seq_id,seq,seq_label_IDP,seq_label_F1,seq_label_F2,seq_label_F3,seq_label_F4,seq_label_F5,seq_label_F6,seq_T5_feature = file_2_data(data_file_name)

	# 标签处理
	res_mask_0 = residue_mask(seq_label_IDP)   
	res_mask_1 = residue_mask(seq_label_F1)     
	res_mask_2 = residue_mask(seq_label_F2)   
	res_mask_3 = residue_mask(seq_label_F3)   
	res_mask_4 = residue_mask(seq_label_F4)   
	res_mask_5 = residue_mask(seq_label_F5)   
	res_mask_6 = residue_mask(seq_label_F6)   

	seq_mask = sequence_mask(seq)       # 

	seq_label_0 = lable_2_value(seq_label_IDP)  
	seq_label_1 = lable_2_value(seq_label_F1)   
	seq_label_2 = lable_2_value(seq_label_F2)  
	seq_label_3 = lable_2_value(seq_label_F3)  
	seq_label_4 = lable_2_value(seq_label_F4)  
	seq_label_5 = lable_2_value(seq_label_F5)   
	seq_label_6 = lable_2_value(seq_label_F6)   


	 
	if is_slice == True:
		seq_id,seq,seq_label_0,seq_label_1,seq_label_2,seq_label_3,seq_label_4,seq_label_5,seq_label_6,seq_T5_feature,res_mask_0,res_mask_1,res_mask_2,res_mask_3,res_mask_4,res_mask_5,res_mask_6,seq_mask = slice_data(seq_id,seq,seq_label_0,seq_label_1,seq_label_2,seq_label_3,seq_label_4,seq_label_5,seq_label_6,
																												seq_T5_feature,
																												res_mask_0,res_mask_1,res_mask_2,res_mask_3,res_mask_4,res_mask_5,res_mask_6,
																												seq_mask,args.max_seq_length)
		# print("after slice lengths: ",len(seq_id))

	# padding
	pad_seq_label_0 = seq_lable_padding(seq_label_0, args.max_seq_length)

	pad_seq_label_1 = seq_lable_padding(seq_label_1, args.max_seq_length)
	pad_seq_label_2 = seq_lable_padding(seq_label_2, args.max_seq_length)
	pad_seq_label_3 = seq_lable_padding(seq_label_3, args.max_seq_length)

	pad_seq_label_4 = seq_lable_padding(seq_label_4, args.max_seq_length)
	pad_seq_label_5 = seq_lable_padding(seq_label_5, args.max_seq_length)
	pad_seq_label_6 = seq_lable_padding(seq_label_6, args.max_seq_length)

	pad_seq_T5_feature = seq_feature_padding(seq_T5_feature, args.max_seq_length)
	pad_res_mask_0 = mask_padding(res_mask_0,args.max_seq_length)

	pad_res_mask_1 = mask_padding(res_mask_1,args.max_seq_length)
	pad_res_mask_2 = mask_padding(res_mask_2,args.max_seq_length)
	pad_res_mask_3 = mask_padding(res_mask_3,args.max_seq_length)

	pad_res_mask_4 = mask_padding(res_mask_4,args.max_seq_length)
	pad_res_mask_5 = mask_padding(res_mask_5,args.max_seq_length)
	pad_res_mask_6 = mask_padding(res_mask_6,args.max_seq_length)

	pad_seq_mask = mask_padding(seq_mask,args.max_seq_length)


	data_samples = []
	for i in range(len(seq_id)):
		one_sample = []   

		one_sample.append(seq_id[i])          
		one_sample.append(seq[i])              
		
		# label
		one_sample.append(pad_seq_label_0[i]) #  (padding)-----------------------2   IDP
		one_sample.append(pad_seq_label_1[i]) #  (padding)-----------------------3   PB
		one_sample.append(pad_seq_label_2[i]) #  (padding)-----------------------4   DB
		one_sample.append(pad_seq_label_3[i]) #  (padding)-----------------------5   RB
		one_sample.append(pad_seq_label_4[i]) #  (padding)-----------------------6   IB
		one_sample.append(pad_seq_label_5[i]) #  (padding)-----------------------7   LB
		one_sample.append(pad_seq_label_6[i]) #  (padding)-----------------------8   Link

		# length
		one_sample.append(len(seq[i]))        # -----------------------------9

		
		one_sample.append(pad_seq_T5_feature[i])   #  (padding)--------------------10
		# one_sample.append(pad_seq_BERT_feature[i])   #  (padding)------------------11
		# one_sample.append(pad_seq_IDP_feature[i])   #  (padding)-------------------12

	 
		one_sample.append(seq_label_0[i]) # ---------13
		one_sample.append(seq_label_1[i]) # ---------14
		one_sample.append(seq_label_2[i]) # ---------15
		one_sample.append(seq_label_3[i]) # ---------16
		one_sample.append(seq_label_4[i]) # ---------17
		one_sample.append(seq_label_5[i]) # ---------18
		one_sample.append(seq_label_6[i]) # ---------19

		# mask
		one_sample.append(pad_res_mask_0[i])          #0,1   mask----------------------20
		one_sample.append(pad_res_mask_1[i])          #0,1   mask----------------------21
		one_sample.append(pad_res_mask_2[i])          #0,1   mask----------------------22
		one_sample.append(pad_res_mask_3[i])          #0,1   mask----------------------23
		one_sample.append(pad_res_mask_4[i])          #0,1   mask----------------------24
		one_sample.append(pad_res_mask_5[i])          #0,1   mask----------------------25
		one_sample.append(pad_res_mask_6[i])          #0,1   mask----------------------26
 
		one_sample.append(pad_seq_mask[i])            #seq  -----------------------27

		data_samples.append(one_sample)


	return data_samples


 
class Batch:  
	def __init__(self):
		self.seq_id = []              # 
		self.seq = []                 # 

		self.seq_label_0 = []           #    IDR
		self.seq_label_1 = []           #    PB
		self.seq_label_2 = []           #    DB 
		self.seq_label_3 = []           #    RB 
		self.seq_label_4 = []           #    IB
		self.seq_label_5 = []           #    LB
		self.seq_label_6 = []           #    Link

		self.seq_length = []          # 

		self.seq_T5_feature = []           #  特征


 
		self.org_seq_label_0 = []        
		self.org_seq_label_1 = []       
		self.org_seq_label_2 = []       
		self.org_seq_label_3 = []        
		self.org_seq_label_4 = []       
		self.org_seq_label_5 = []       
		self.org_seq_label_6 = []       


		self.res_mask_0 = []           
		self.res_mask_1 = []            
		self.res_mask_2 = []           
		self.res_mask_3 = []           
		self.res_mask_4 = []             
		self.res_mask_5 = []            
		self.res_mask_6 = []            

		self.seq_mask = []            


def one_batch_data(one_data_samples):
	one_batch = Batch()
	for i in range(len(one_data_samples)):
		one_batch.seq_id.append(one_data_samples[i][0])           #   
		one_batch.seq.append(one_data_samples[i][1])                # 

		one_batch.seq_label_0.append(one_data_samples[i][2])          # 
		one_batch.seq_label_1.append(one_data_samples[i][3])          # 
		one_batch.seq_label_2.append(one_data_samples[i][4])          # 
		one_batch.seq_label_3.append(one_data_samples[i][5])          # 
		one_batch.seq_label_4.append(one_data_samples[i][6])          # 
		one_batch.seq_label_5.append(one_data_samples[i][7])          # 
		one_batch.seq_label_6.append(one_data_samples[i][8])          # 

		one_batch.seq_length.append(one_data_samples[i][9])         # 

		one_batch.seq_T5_feature.append(one_data_samples[i][10])        #   
		# one_batch.seq_BERT_feature.append(one_data_samples[i][11])        #   
		# one_batch.seq_IDP_feature.append(one_data_samples[i][12])        #   

		one_batch.org_seq_label_0.append(one_data_samples[i][11])     # 
		one_batch.org_seq_label_1.append(one_data_samples[i][12])     # 
		one_batch.org_seq_label_2.append(one_data_samples[i][13])     # 
		one_batch.org_seq_label_3.append(one_data_samples[i][14])     # 
		one_batch.org_seq_label_4.append(one_data_samples[i][15])     # 
		one_batch.org_seq_label_5.append(one_data_samples[i][16])     # 
		one_batch.org_seq_label_6.append(one_data_samples[i][17])     # 

		one_batch.res_mask_0.append(one_data_samples[i][18])
		one_batch.res_mask_1.append(one_data_samples[i][19])
		one_batch.res_mask_2.append(one_data_samples[i][20])
		one_batch.res_mask_3.append(one_data_samples[i][21])
		one_batch.res_mask_4.append(one_data_samples[i][22])
		one_batch.res_mask_5.append(one_data_samples[i][23])
		one_batch.res_mask_6.append(one_data_samples[i][24])

		one_batch.seq_mask.append(one_data_samples[i][25])


	return one_batch


def Batches_data(data_samples, batch_size, is_train): #all data samples  
	# if is_train == True:
	# 	random.shuffle(data_samples)
	batches = []
	data_len = len(data_samples)
	batch_nums = int(data_len/batch_size) 
	def genNextSamples():
		for i in range(0, batch_nums*batch_size, batch_size):
			yield data_samples[i: i + batch_size]
		if data_len % batch_size != 0:   
			last_num = data_len - batch_nums*batch_size
			up_num = batch_size - last_num
			l1 = data_samples[batch_nums*batch_size : data_len]
			l2 = data_samples[0: up_num]
			yield l1+l2
	
	for one_data_samples in genNextSamples():
		one_batch = one_batch_data(one_data_samples)
		batches.append(one_batch)	
	return batches  


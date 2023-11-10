# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2023-03-01 15:40:08
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2023-06-14 22:40:38
import numpy as np 
import os
from prepare_model_data import lable_2_value


def write_2_file(data_file, data_samples, data_batches, IDR_probs, PB_probs, DB_probs, RB_probs, IB_probs, LB_probs, Link_probs, file_name,output_type):
	batch_size = np.array(data_batches[0].seq_label_0).shape[0]
	max_length = np.array(data_batches[0].seq_label_0).shape[1]
	slice_length = len(data_samples)


	pred_logs_0 = []
	pred_logs_1 = []
	pred_logs_2 = []
	pred_logs_3 = []
	pred_logs_4 = []
	pred_logs_5 = []
	pred_logs_6 = []
	for b in range(len(IDR_probs)):
		# IDR
		pred_logs_0 += list(IDR_probs[b])
		# PB
		pred_logs_1 += list(PB_probs[b])
		# DB
		pred_logs_2 += list(DB_probs[b])
		# RB
		pred_logs_3 += list(RB_probs[b])
		# IB
		pred_logs_4 += list(IB_probs[b])
		# LB
		pred_logs_5 += list(LB_probs[b])
		# Link
		pred_logs_6 += list(Link_probs[b])
	pred_logs_0 = pred_logs_0[:slice_length]
	pred_logs_1 = pred_logs_1[:slice_length]
	pred_logs_2 = pred_logs_2[:slice_length]
	pred_logs_3 = pred_logs_3[:slice_length]
	pred_logs_4 = pred_logs_4[:slice_length]
	pred_logs_5 = pred_logs_5[:slice_length]
	pred_logs_6 = pred_logs_6[:slice_length]


	pred_seq_ids = []  
	for d in range(len(data_batches)):
		batch_data = data_batches[d]  
		for i in range(len(batch_data.seq_id)):  #[batch_size]
			pred_seq_ids.append(str(batch_data.seq_id[i]).replace('\r',''))  #  pred_seq_ids
	pred_seq_ids = pred_seq_ids[:slice_length]
	org_ids = list(set(pred_seq_ids))



	org_seq_pred_0 = []
	org_seq_pred_1 = []
	org_seq_pred_2 = []
	org_seq_pred_3 = []
	org_seq_pred_4 = []
	org_seq_pred_5 = []
	org_seq_pred_6 = []
	for i in range(len(org_ids)):
		find_id = org_ids[i]

		one_pred_0 = []
		one_pred_1 = []
		one_pred_2 = []
		one_pred_3 = []
		one_pred_4 = []
		one_pred_5 = []
		one_pred_6 = []
		for j in range(len(pred_seq_ids)):
			if pred_seq_ids[j] == find_id:
				one_pred_0 += list(pred_logs_0[j])
				one_pred_1 += list(pred_logs_1[j])
				one_pred_2 += list(pred_logs_2[j])
				one_pred_3 += list(pred_logs_3[j])
				one_pred_4 += list(pred_logs_4[j])
				one_pred_5 += list(pred_logs_5[j])
				one_pred_6 += list(pred_logs_6[j])
		org_seq_pred_0.append([find_id,one_pred_0])
		org_seq_pred_1.append([find_id,one_pred_1])
		org_seq_pred_2.append([find_id,one_pred_2])
		org_seq_pred_3.append([find_id,one_pred_3])
		org_seq_pred_4.append([find_id,one_pred_4])
		org_seq_pred_5.append([find_id,one_pred_5])
		org_seq_pred_6.append([find_id,one_pred_6])


	pred_final_ordered_0 = []
	pred_final_ordered_1 = []
	pred_final_ordered_2 = []
	pred_final_ordered_3 = []
	pred_final_ordered_4 = []
	pred_final_ordered_5 = []
	pred_final_ordered_6 = []
	for i in range(len(data_file)): 
		find_id = str(str(data_file[i][0]).replace('>','')).replace('\r','')
		for j in range(len(org_seq_pred_0)):
			if org_seq_pred_0[j][0] == find_id:
				pred_final_ordered_0.append(org_seq_pred_0[j][-1][:len(data_file[i][1])])
				pred_final_ordered_1.append(org_seq_pred_1[j][-1][:len(data_file[i][1])])
				pred_final_ordered_2.append(org_seq_pred_2[j][-1][:len(data_file[i][1])])
				pred_final_ordered_3.append(org_seq_pred_3[j][-1][:len(data_file[i][1])])
				pred_final_ordered_4.append(org_seq_pred_4[j][-1][:len(data_file[i][1])])
				pred_final_ordered_5.append(org_seq_pred_5[j][-1][:len(data_file[i][1])])
				pred_final_ordered_6.append(org_seq_pred_6[j][-1][:len(data_file[i][1])])

	
	write_file = open(file_name,"w")

	for i in range(len(data_file)):
		write_file.write(data_file[i][0]+'\n')
		write_file.write(data_file[i][1]+'\n')
		one_seq_len = len(data_file[i][1].replace('\r',''))
		pred_0 = [round(j,4) for j in pred_final_ordered_0[i]]
		pred_1 = [round(j,4) for j in pred_final_ordered_1[i]]
		pred_2 = [round(j,4) for j in pred_final_ordered_2[i]]
		pred_3 = [round(j,4) for j in pred_final_ordered_3[i]]
		pred_4 = [round(j,4) for j in pred_final_ordered_4[i]]
		pred_5 = [round(j,4) for j in pred_final_ordered_5[i]]
		pred_6 = [round(j,4) for j in pred_final_ordered_6[i]]
		pred_0 = pred_0[0:one_seq_len]
		pred_1 = pred_1[0:one_seq_len]
		pred_2 = pred_2[0:one_seq_len]
		pred_3 = pred_3[0:one_seq_len]
		pred_4 = pred_4[0:one_seq_len]
		pred_5 = pred_5[0:one_seq_len]
		pred_6 = pred_6[0:one_seq_len]

		if output_type == 'b':
			# best ROC performance
			pred_0 = [1 if p > 0.2340 else 0 for p in pred_0]
			pred_1 = [1 if p > 0.1678 else 0 for p in pred_1]
			pred_2 = [1 if p > 0.0163 else 0 for p in pred_2]
			pred_3 = [1 if p > 0.006 else 0 for p in pred_3]
			pred_4 = [1 if p > 0.0011 else 0 for p in pred_4]
			pred_5 = [1 if p > 0.0109 else 0 for p in pred_5]
			pred_6 = [1 if p > 0.0254 else 0 for p in pred_6]
			write_file.write("".join(str(j) for j in pred_0))
			write_file.write('\n')
			write_file.write("".join(str(j) for j in pred_1))
			write_file.write('\n')
			write_file.write("".join(str(j) for j in pred_2))
			write_file.write('\n')
			write_file.write("".join(str(j) for j in pred_3))
			write_file.write('\n')
			write_file.write("".join(str(j) for j in pred_4))
			write_file.write('\n')
			write_file.write("".join(str(j) for j in pred_5))
			write_file.write('\n')
			write_file.write("".join(str(j) for j in pred_6))
			write_file.write('\n')
		else:
			write_file.write(",".join(str(j) for j in pred_0))
			write_file.write('\n')
			write_file.write(",".join(str(j) for j in pred_1))
			write_file.write('\n')
			write_file.write(",".join(str(j) for j in pred_2))
			write_file.write('\n')
			write_file.write(",".join(str(j) for j in pred_3))
			write_file.write('\n')
			write_file.write(",".join(str(j) for j in pred_4))
			write_file.write('\n')
			write_file.write(",".join(str(j) for j in pred_5))
			write_file.write('\n')
			write_file.write(",".join(str(j) for j in pred_6))
			write_file.write('\n')
		
	print("Find results :  ",file_name)
	write_file.close()






		





































	



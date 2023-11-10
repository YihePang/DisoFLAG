# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2023-02-27 10:40:50
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2023-06-14 22:39:56

class Args_config:  
	def __init__(self):
		self.use_gpu = True
		self.max_seq_length = 128

		self.feature_dim = 1024
		self.encoder_hidden = 512

		self.decoder_hidden = 1024

		self.decoder_dropout = 0.3

		self.model_path = './saved_model'
		self.epochs = 50
		self.batch_size = 16
		self.learning_rate = 0.00005
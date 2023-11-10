# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2023-02-27 22:01:17
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2023-06-14 22:43:02
import torch as t
from torch import nn

import torch.nn.functional as F
import torch.nn.init as init

import numpy as np
import scipy.sparse as sp

class Encoder(nn.Module):
	def __init__(self,args):
		super().__init__()
		# 给模型赋个名
		self.model_name = 'Encoder'
		self.args = args

		self.Bi_GRU = nn.GRU(input_size = self.args.feature_dim,
								hidden_size = self.args.encoder_hidden,
								batch_first = True,
								bidirectional = True)  

	def forward(self, input_feature):
		input_feature = input_feature.to(t.float32)  #[batch_size, L ,1024]

		# Bi-GRU
		encoder_outputs, encoder_hiddens = self.Bi_GRU(input_feature)  # layer_outputs: [batch, seq_len, 2*hidden_size]
		# print("encoder_outputs:",encoder_outputs.shape)  # [B, L, 1024]
		# print("encoder_hiddens:",encoder_hiddens.shape)  # [2_layers, B, hidden]
		
		return encoder_outputs, encoder_hiddens
		

class Attention(nn.Module):
	def __init__(self,args):
		super().__init__()
		self.model_name = 'Decoder_attention'
		self.args = args

	def forward(self, decoder_state_t, encoder_outputs):
		# decoder_state_t [B, 1024]
		# encoder_outputs [B,L,1024]
		batch_size, seq_length, hidden  = encoder_outputs.shape
		state_trans = t.tile(decoder_state_t.unsqueeze(dim=1), dims=(seq_length, 1))
		# print("Atten_state_trans:",state_trans.shape)                              # [B, L , 1024]

		multip = t.matmul(state_trans, encoder_outputs.transpose(1, 2))
		# print("Atten_multip:",multip.shape)                                        # [B, L , L]

		attention_scores =  t.sum(multip, dim=-1)
		# print("attention_scores:",attention_scores.shape)                           # [B, L]

		attention_scores = t.softmax(attention_scores, dim=-1)
		# print("attention_scores:",attention_scores.shape)                           # [B, L]

		context = t.sum(attention_scores.unsqueeze(dim=-1) * encoder_outputs, dim=1)
		# print("attention_context:",context.shape)                                 # [B, 1024]

		return context, attention_scores


class GraphConvolution(nn.Module):
	def __init__(self, input_dim, output_dim, use_bias=True):
		super(GraphConvolution, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.use_bias = use_bias
		self.weight = nn.Parameter(t.Tensor(input_dim, output_dim))
		if self.use_bias:
			self.bias = nn.Parameter(t.Tensor(output_dim))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		init.kaiming_uniform_(self.weight)
		if self.use_bias:
			init.zeros_(self.bias)

	def forward(self, adjacency, input_feature):
		"""
		Args: 
		-------
			adjacency: torch.sparse.FloatTensor
				
			input_feature: torch.Tensor           			
		"""
		support = t.matmul(input_feature, self.weight)
		# print("GCN----support:",support.shape)  # [B, 6, 128]
		# print("GCN----adjacency:",adjacency.shape)  # [6, 6]

		# output = t.sparse.mm(adjacency, support)
		output = t.matmul(adjacency,support)
		# print("GCN----output:",output.shape)  # [B, 6, 128]

		if self.use_bias:
			output += self.bias
		# print("GCN----output:",output.shape)  # [B, 6, 1]

		return output, self.weight, self.bias, adjacency





class Graph_decoder(nn.Module):
	def __init__(self,args):
		super().__init__()
		self.model_name = 'Graph'
		self.args = args

		# adj = t.Tensor(np.array([[1, 0.385, 0.410, 0.377, 0.372, 0.366],
		# 						   [0.385, 1, 0.737, 0.683, 0.714, 0.649],
		# 						   [0.410, 0.737, 1, 0.721, 0.751, 0.681],
		# 						   [0.377, 0.683, 0.721, 1, 0.698, 0.634],
		# 						   [0.372, 0.714, 0.751, 0.698, 1, 0.663],
		# 						   [0.366, 0.649, 0.681, 0.634, 0.663, 1]]))
		adj = t.Tensor(np.array([[1, 0.363, 0.430, 0.356, 0.397, 0.367],
								   [0.363, 1, 0.741, 0.620, 0.698, 0.624],
								   [0.430, 0.741, 1, 0.724, 0.802, 0.724],
								   [0.356, 0.620, 0.724, 1, 0.681, 0.610],
								   [0.397, 0.698, 0.802, 0.681, 1, 0.684],
								   [0.367, 0.624, 0.724, 0.610, 0.684, 1]]))

		self.edgs = nn.Parameter(adj)

		self.conv1 = GraphConvolution(1024, 128)		


	def forward(self, IDR_vec, PB_vec, DB_vec, RB_vec, IB_vec, LB_vec, Link_vec):
		batch_size, seq_length, hidden_size  = IDR_vec.shape

		
		Graph_outputs = []
		Graph_Ws = []
		Graph_bs = []
		Graph_adjs = []

		for s in range(seq_length):
			one_IDR_input = IDR_vec[:, s, :]   # [B, 1024]

			one_PB_input = PB_vec[:, s, :]   # [B, 1024]
			one_DB_input = DB_vec[:, s, :]   # [B, 1024]
			one_RB_input = RB_vec[:, s, :]   # [B, 1024]
			one_IB_input = IB_vec[:, s, :]   # [B, 1024]
			one_LB_input = LB_vec[:, s, :]   # [B, 1024]
			one_Link_input = Link_vec[:, s, :]   # [B, 1024]
			X = []
			X.append(one_PB_input)
			X.append(one_DB_input)
			X.append(one_RB_input)
			X.append(one_IB_input)
			X.append(one_LB_input)
			X.append(one_Link_input)
			X = t.stack(X, dim=0)
			X = X.transpose(0,1)    # [B, 6, 1024]
			one_x, G_W, G_b, G_a = self.conv1(self.edgs, X)  # [B, 6, out_dim]
			Graph_outputs.append(one_x)       # [L, B, 6, out_dim]
			# Graph_Ws.append(G_W)  
			# Graph_bs.append(G_b) 
			# Graph_adjs.append(G_a) 

		Graph_outputs = t.stack(Graph_outputs, dim=1)   # [B, L, 6, out_dim]
		# Graph_Ws = t.stack(Graph_Ws, dim=1)   # [B, L, 6, out_dim]
		# Graph_bs = t.stack(Graph_bs, dim=1)   # [B, L, 6, out_dim]
		# Graph_adjs = t.stack(Graph_adjs, dim=1)   # [B, L, 6, out_dim]
		Graph_Ws = G_W   # [B, L, 6, out_dim]
		Graph_bs = G_b   # [B, L, 6, out_dim]
		Graph_adjs = G_a
		

		# Max pooling
		pool_out = t.nn.functional.max_pool2d(input=Graph_outputs, kernel_size=(6,1), stride=(6,1))
		pool_out = pool_out.squeeze(dim=2)
		# print("Max pooling----output:",pool_out.shape)   # [B, L, out_dim]

		IDR_trans = pool_out
		PB_trans = Graph_outputs[:, :, 0, :]
		DB_trans = Graph_outputs[:, :, 1, :]
		RB_trans = Graph_outputs[:, :, 2, :]
		IB_trans = Graph_outputs[:, :, 3, :]
		LB_trans = Graph_outputs[:, :, 4, :]
		Link_trans = Graph_outputs[:, :, 5, :]

		return IDR_trans, PB_trans, DB_trans, RB_trans, IB_trans, LB_trans, Link_trans, Graph_Ws, Graph_bs, Graph_adjs



class Decoder(nn.Module):
	def __init__(self,args):
		super().__init__()
		# 给模型赋个名
		self.model_name = 'Decoder'
		self.args = args
		self.GRU = nn.GRUCell( self.args.decoder_hidden, self.args.decoder_hidden)
		self.attention = Attention(self.args)
		self.decoder_projection = nn.Linear(2*self.args.decoder_hidden, self.args.decoder_hidden)
		self.dropout = nn.Dropout(self.args.decoder_dropout)

	def forward(self, encoder_outputs, encoder_hiddens):
		decoder_inputs = encoder_outputs
		batch_size, seq_length, hidden  = decoder_inputs.shape
		
		ht_0 = encoder_hiddens[0]  # [32, 512]
		ht_1 = encoder_hiddens[1]  # [32, 512]
		ht = t.cat([ht_0, ht_1], dim=-1)  # [32, 1024]
		# print("decoder_init_ht:",ht.shape)  

		decoder_outputs = []
		for s in range(seq_length):
			# 计算 attention context
			context, attention_scores = self.attention(ht, encoder_outputs)    # [B, 1024], [B, L]

			one_decoder_input = t.cat([decoder_inputs[:, s, :], context], dim=-1)   # [B, 1024] + [B, 1024] = [B, 2048]
			# print("one_decoder_input:",one_decoder_input.shape)  # [B, 2048]
			one_input = self.decoder_projection(one_decoder_input)
			
			ht = self.GRU(one_input, ht)
			# print("one_ht:",ht.shape)  # [B, 1024]
			
			decoder_outputs.append(ht)

		decoder_outputs = t.stack(decoder_outputs, dim=0)
		# print("FINAL decoder decoder_outputs:",decoder_outputs.shape)  # [L, B, 1024]
		
		decoder_outputs = decoder_outputs.transpose(0,1)
		# print("FINAL decoder decoder_outputs:",decoder_outputs.shape)  # [B, L, 1024]

		return decoder_outputs


class Seq2FUN(nn.Module):
	def __init__(self,args):			
		super().__init__()
		self.model_name = 'Model'
		self.args = args
		self.encoder = Encoder(self.args)
		self.decoder = Decoder(self.args)
		self.Graph_decoder = Graph_decoder(self.args)

		
		self.IDR_trans = nn.Linear(self.args.decoder_hidden, self.args.decoder_hidden)

		self.PB_trans = nn.Linear(self.args.decoder_hidden, self.args.decoder_hidden)
		self.DB_trans = nn.Linear(self.args.decoder_hidden, self.args.decoder_hidden)
		self.RB_trans = nn.Linear(self.args.decoder_hidden, self.args.decoder_hidden)
		self.IB_trans = nn.Linear(self.args.decoder_hidden, self.args.decoder_hidden)
		self.LB_trans = nn.Linear(self.args.decoder_hidden, self.args.decoder_hidden)
		self.Link_trans = nn.Linear(self.args.decoder_hidden, self.args.decoder_hidden)

		
		self.IDP_cal_prob = nn.Linear(in_features=128, out_features=1, bias =True)

		self.PB_cal_prob = nn.Linear(in_features=128, out_features=1, bias =True)
		self.DB_cal_prob = nn.Linear(in_features=128, out_features=1, bias =True)
		self.RB_cal_prob = nn.Linear(in_features=128, out_features=1, bias =True)
		self.IB_cal_prob = nn.Linear(in_features=128, out_features=1, bias =True)
		self.LB_cal_prob = nn.Linear(in_features=128, out_features=1, bias =True)
		self.Link_cal_prob = nn.Linear(in_features=128, out_features=1, bias =True)


		self.activate = nn.Sigmoid()


	def forward(self, input_feature):
		# print("input_feature:",input_feature.shape)  # [B, L, 1024]
		# Bi-GRU Encoder
		encoder_outputs, encoder_hiddens = self.encoder(input_feature)

		# Decoder_attention
		decoder_outputs = self.decoder(encoder_outputs, encoder_hiddens)   # [B, L, 1024]

		
		# IDR feature
		IDR_vec = self.IDR_trans(decoder_outputs)    # [B, L, 1024]

		PB_vec = self.PB_trans(decoder_outputs)
		DB_vec = self.DB_trans(decoder_outputs)
		RB_vec = self.RB_trans(decoder_outputs)
		IB_vec = self.IB_trans(decoder_outputs)
		LB_vec = self.LB_trans(decoder_outputs)
		Link_vec = self.Link_trans(decoder_outputs)

		# Gragh decoder
		IDR_F_vec, PB_F_vec, DB_F_vec, RB_F_vec, IB_F_vec, LB_F_vec, Link_F_vec, Graph_Ws, Graph_bs, Graph_adjs = self.Graph_decoder(IDR_vec, PB_vec, DB_vec, RB_vec, IB_vec, LB_vec, Link_vec)

		# cal_probs
		IDR_probs = t.squeeze(self.activate(self.IDP_cal_prob(IDR_F_vec)))  # [B, L]

		PB_probs = t.squeeze(self.activate(self.PB_cal_prob(PB_F_vec)))     # [B, L]
		DB_probs = t.squeeze(self.activate(self.DB_cal_prob(DB_F_vec)))     # [B, L]
		RB_probs = t.squeeze(self.activate(self.RB_cal_prob(RB_F_vec)))     # [B, L]
		IB_probs = t.squeeze(self.activate(self.IB_cal_prob(IB_F_vec)))     # [B, L]
		LB_probs = t.squeeze(self.activate(self.LB_cal_prob(LB_F_vec)))     # [B, L]
		Link_probs = t.squeeze(self.activate(self.Link_cal_prob(Link_F_vec)))  # [B, L]

		return IDR_probs, PB_probs, DB_probs, RB_probs, IB_probs, LB_probs, Link_probs

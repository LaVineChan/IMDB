import torch.nn as nn
import torch
from torch.autograd import Variable

class TextRNN(nn.Module):
	def __init__(self, args):
		super(TextRNN, self).__init__()
		self.args = args
		embedding_dim = args.embedding_dim
		label_num = 2
		vocab_size = args.vocab_size
		self.hidden_size = args.hidden_size
		self.layer_num = args.layer_num
		if args.static:
			self.embedding = nn.Embedding(vocab_size, embedding_dim)
			self.embedding = self.embedding.from_pretrained(args.vectors, freeze=False)
		else:
			self.embedding = nn.Embedding(vocab_size, embedding_dim)

		self.lstm = nn.LSTM(embedding_dim,
							self.hidden_size,
							self.layer_num,
							batch_first=True,# 第一个维度设为 batch, 即:(batch_size, seq_length, embedding_dim)
							bidirectional=True)
		self.dropout = nn.Dropout(args.dropout)
		self.fc = nn.Linear(self.hidden_size*2, label_num)

	# def init_weights(self):
	# 	initrange = 0.5
	# 	self.embedding.weight.data.uniform_(-initrange, initrange)
	# 	self.fc.weight.data.uniform_(-initrange, initrange)
	# 	self.fc.bias.data.zero_()

	def forward(self, x):
 	   # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大长度
		x = self.embedding(x)  #经过embedding,x的维度为(batch_size, time_step, input_size=embedding_dim)

		x = Variable(x)
		# 隐层初始化
		# h0维度为(num_layers*2, batch_size, hidden_size)
		# c0维度为(num_layers*2, batch_size, hidden_size)
		h0 = torch.zeros((self.layer_num * 2, x.size(0), self.hidden_size), dtype=torch.float).cuda()
		c0 = torch.zeros((self.layer_num * 2, x.size(0), self.hidden_size), dtype=torch.float).cuda()

		# LSTM前向传播，此时out维度为(batch_size, seq_length, hidden_size*2)
		# hn,cn表示最后一个状态,维度与h0和c0一样
		out, (hn, cn) = self.lstm(x, (h0, c0))
		out = self.dropout(out[:, -1, :])
		out = self.fc(out)
		return out






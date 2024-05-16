import torch
import torch.nn as nn

from typing import Tuple

class LSTM(nn.Module):
    
	def __init__(self, no_layers: int, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int) -> None:
		super(LSTM, self).__init__()

		self.output_dim = output_dim
		self.hidden_dim = hidden_dim

		self.no_layers = no_layers
		self.vocab_size = vocab_size

		# embedding layer
		self.embedding_layer = nn.Embedding(num_embeddings=vocab_size+1, embedding_dim=embedding_dim)

		# LSTM layer
		self.lstm_layer = nn.LSTM(
			input_size=embedding_dim, 
			hidden_size=hidden_dim, 
			num_layers=no_layers,
			batch_first=True,
		)

		# fully connected layer
		self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)
		self.sig = nn.Sigmoid()

	def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

		batch_size = x.shape[0]

		# if hidden shape is not the same as expected due to batch size
		# average the others
		# target is the batch_size
		# we expect hidden state to have (no_layers, target, hidden_dim)
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		current_hidden_batch_size = hidden[0].shape[1]
		if (batch_size != current_hidden_batch_size):
			new_h = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
			new_c = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
			for i in range(self.no_layers):

				n = current_hidden_batch_size // batch_size
				for j in range(batch_size):

					temp_h = hidden[0][i, n*j:n*(j+1), :].float()
					temp_c = hidden[1][i, n*j:n*(j+1), :].float()
					if (j == batch_size - 1):
						temp_h = hidden[0][i, n*j:, :].float()
						temp_c = hidden[1][i, n*j:, :].float()

					mean_temp_h = torch.mean(temp_h, dim = 0)
					mean_temp_c = torch.mean(temp_c, dim = 0)

					new_h[i][j] = mean_temp_h
					new_c[i][j] = mean_temp_c
			hidden = (new_h, new_c)

		# embeddings and lstm_out
		embeds = self.embedding_layer(x) # shape: B x S x Feature since batch = True
		lstm_out, hidden = self.lstm_layer(embeds, hidden)

		lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

		# fully connected layer
		out = self.fc(lstm_out)
		out = self.sig(out)

		# reshape to be batch size first
		out = out.view(batch_size, -1)

		# get last batch of labels
		out = out[:, -self.output_dim:]

		return out, hidden
	
	def init_hidden(self, batch_size: int):
		"""
		Initialize hidden state
		Create two new tensors with sizes n_layers x batch_size x hidden_dim,
		Initialized to zero, for hidden state and cell state of LSTM
		"""
		h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim))
		c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim))
		if (torch.cuda.is_available()):
			h0 = h0.to('cuda')
			c0 = c0.to('cuda')
		
		hidden = (h0, c0)
		return hidden
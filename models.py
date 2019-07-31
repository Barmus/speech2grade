import torch

class ThreeLayerNet_2LevelAttn(torch.nn.Module):
	def __init__(self, frame_dim, h1_dim, h2_dim, y_dim):

		super(ThreeLayerNet_2LevelAttn, self).__init__()
		self.attn1 = torch.nn.Linear(frame_dim, frame_dim, bias = False)
		self.attn2 = torch.nn.Linear(frame_dim, frame_dim, bias = False)
		self.linear1 = torch.nn.Linear(frame_dim, h1_dim)
		self.linear2 = torch.nn.Linear(h1_dim, h2_dim)
		self.linear3 = torch.nn.Linear(h2_dim, y_dim)

		self.frame_dim = frame_dim
		self.utt_weights = None


	def forward(self, X, mask):

		M = mask

		# Apply self-attention over the frames
		A1 = self.attn1(torch.eye(self.frame_dim))
		S_half = torch.einsum('buvi,ij->buvj', X, A1)
		S = torch.einsum('buvi,buvi->buv', X, S_half)
		T = torch.nn.Tanh()
		ST = T(S)
		# Use mask to convert padding scores to -inf (go to zero after softmax normalisation)
		# Note tanh function maintained 0s in ST in correct positions (as uAu^T = 0 when u = 0)
		ST_translated = ST + 1
		ST_masked = ST_translated * M
		# Normalise weights using softmax for each utterance of each speaker
		SM = torch.nn.Softmax(dim = 2)
		W = SM(ST_masked)
		# Perform weighted sum (using normalised scores above) along the words axes for X
		weights_extra_axis = torch.unsqueeze(W, 3)
		repeated_weights = weights_extra_axis.expand(-1, -1, -1, X.size(3))
		x_multiplied = X * repeated_weights
		x_attn1 = torch.sum(x_multiplied, dim = 2)



		# Apply self-attention over the utterances
		A2 = self.attn2(torch.eye(self.frame_dim))
		S_half2 = torch.einsum('bui,ij->buj',x_attn1, A2)
		S2 = torch.einsum('bui,bui->bu',x_attn1, S_half2)
		ST2 = T(S2)
		# Normalise using softmax along the utterances dimension
		SM2 = torch.nn.Softmax(dim = 1)
		W2 = SM2(ST2)
		self.utt_weights = W2
		# Perform weight sum (along the utts axes)
		weights_extra_axis2 = torch.unsqueeze(W2, 2)
		repeated_weights2 = weights_extra_axis2.expand(-1, -1, X.size(3))
		x_multiplied2 = x_attn1 * repeated_weights2
		X_final = torch.sum(x_multiplied2, dim = 1)



		# Pass through feed-forward DNN
		h1 = self.linear1(X_final).clamp(min=0)
		h2 = self.linear2(h1).clamp(min=0)
		y_pred = self.linear3(h2)
		return y_pred





	def get_utt_attn_weights(self):
		return self.utt_weights






class ThreeLayerNet_1LevelAttn(torch.nn.Module):
	def __init__(self, frame_dim, h1_dim, h2_dim, y_dim):

		super(ThreeLayerNet_1LevelAttn, self).__init__()
		self.attn1 = torch.nn.Linear(frame_dim, frame_dim, bias = False)
		self.linear1 = torch.nn.Linear(frame_dim*21, h1_dim)
		self.linear2 = torch.nn.Linear(h1_dim, h2_dim)
		self.linear3 = torch.nn.Linear(h2_dim, y_dim)

		self.frame_dim = frame_dim


	def forward(self, X, mask):

		M = mask

		# Apply self-attention over the words
		A1 = self.attn1(torch.eye(self.frame_dim))
		S_half = torch.einsum('buvi,ij->buvj', X, A1)
		S = torch.einsum('buvi,buvi->buv', X, S_half)
		T = torch.nn.Tanh()
		ST = T(S)
		# Use mask to convert padding scores to -inf (go to zero after softmax normalisation)
		# Note tanh function maintained 0s in ST in correct positions (as uAu^T = 0 when u = 0)
		ST_translated = ST + 1
		ST_masked = ST_translated * M
		# Normalise weights using softmax for each utterance of each speaker
		SM = torch.nn.Softmax(dim = 2)
		W = SM(ST_masked)
		# Perform weighted sum (using normalised scores above) along the words axes for X
		weights_extra_axis = torch.unsqueeze(W, 3)
		repeated_weights = weights_extra_axis.expand(-1, -1, -1, X.size(3))
		x_multiplied = X * repeated_weights
		x_attn1 = torch.sum(x_multiplied, dim = 2)



		# Concatenate the utterances
		X_final = x_attn1.view(x_attn1.size(0), -1)


		# Pass through feed-forward DNN
		h1 = self.linear1(X_final).clamp(min=0)
		h2 = self.linear2(h1).clamp(min=0)
		y_pred = self.linear3(h2)
		return y_pred


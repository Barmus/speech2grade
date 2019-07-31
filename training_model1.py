import json
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models import ThreeLayerNet_2LevelAttn

# NOTE: no validation data set is being used here

if torch.cuda.is_available():
	print("cuda available")
	device = torch.device("cuda")
else:
	print("no cuda")
	device = torch.device("cpu")


# Get all the data
target_file = 'data_padded_fb.txt'
with open(target_file, 'r') as f:
	data = json.load(f)

print("Got the data")

# Extract relevant parts of data
X = data[0]
y = data[1]
L = data[2]

# Convert to tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y)

X = X.to(device)
y = y.to(device)

# Make the mask from utterance lengths matrix L
M = [[([1]*utt_len + [-100000]*(X.size(2)- utt_len)) for utt_len in speaker] for speaker in L]
M = torch.FloatTensor(M)
M = M.to(device)

# Initialise constants
frame_dim = 40
h1_dim = 250
h2_dim = 150
y_dim = 1


bs = 32
epochs = 20

# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(X, y, M)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)


model = ThreeLayerNet_2LevelAttn(frame_dim, h1_dim, h2_dim, y_dim)
model = model.to(device)

print("made model")

criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(epochs):
	model.train()
	for xb, yb, mb in train_dl:
	
		# Forward pass
		y_pred = model(xb, mb)
	

		# Compute loss
		loss = criterion(y_pred[:,0], yb)
	
		# Zero gradients, backward pass, update weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	model.eval()
	y_pr = model(X,M)
	mse_loss = criterion(y_pr[:,0], y)
	print(epoch, mse_loss.item())	
	

# Save the model to a file
file_path = 'model1_trained.pt'
torch.save(model, file_path)





















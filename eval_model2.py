import json
import torch
from models import ThreeLayerNet_1LevelAttn

# Load all the data
target_file = 'eval_data_padded_fb.txt'
with open(target_file, 'r') as f:
	data = json.load(f)

print("Loaded data")

# Extract relevant parts of data
X = data[0]
y = data[1]
L = data[2]

# Convert to tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y)

size = y.size(0)

# Make mask from L
M = [[([1]*utt_len + [-100000]*(X.size(2)- utt_len)) for utt_len in speaker] for speaker in L]

M = torch.FloatTensor(M)


# Load trained model
model_path = 'model2_trained.pt'
model = torch.load(model_path)
model.eval()

criterion = torch.nn.MSELoss(reduction = 'mean')

y_pred = model(X, M)
y_pred_useful = y_pred[:, 0]
loss = criterion(y_pred_useful, y)
mse = loss.item()

# Caculate pcc
vy = y - torch.mean(y)
vyp = y_pred_useful - torch.mean(y_pred_useful)
pcc = 1/((size-1)*torch.std(vy) *torch.std(vyp))*(torch.sum(vy*vyp))
pcc = pcc.item()

# Find percent in <0.5 and <1
total05 = 0
total1 = 0
for a,b in zip(y, y_pred_useful):
        diff = abs(a-b)
        if diff < 1:
                total1 += 1
                if diff < 0.5:
                        total05 += 1

less1 = total1/size
less05 = total05/size


print("MSE: "+ str(mse) + " PCC: " + str(pcc)+ " < 1: " + str(less1) + " < 0.5: " + str(less05))


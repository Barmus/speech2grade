import json
import torch
from models import ThreeLayerNet_2LevelAttn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

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
model_path = 'model1_trained.pt'
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

# display attention weights distribution across utterances
weights_matrix = model.get_utt_attn_weights()
means = torch.mean(weights_matrix, dim = 0)
stds = torch.std(weights_matrix, dim = 0)

means = means.tolist()
stds = stds.tolist()

ind = np.arange(start = 1, stop = 22, dtype = int)


threshold = 1/21
colors = ['b', 'b', 'b', 'b', 'b', 'b', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'c', 'm', 'y', 'y', 'y', 'y', 'y']

plt.bar(ind, means, yerr=stds, color = colors)


plt.ylabel('Attention Weights')
plt.xlabel('Utterances')

plt.axhline(y = threshold, linewidth =1, color = 'red')

legend_colors = ['b', 'g', 'c', 'm', 'y']
lines = [Line2D([0], [0], color = c, linewidth = 3, linestyle='-') for c in legend_colors]
labels = ['Part 1', 'Part 2', 'Part 3', 'Part 4', 'Part 5']
plt.legend(lines, labels)


png_file = 'attn_distribution.png'
plt.savefig(png_file)

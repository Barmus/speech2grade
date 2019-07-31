import json

target_file = 'eval_fb_data_training.txt'

with open(target_file, 'r') as f:
	data_dict = json.load(f)

# Define the number of utterances per speaker
MAX_UTTS_PER_SPEAKER = 21

# Define threshold for max length of vector of utterance
MAX_LEN_UTT = 6379

# Define Filterbank size (vector length of frame)
FRAME_SIZE = 40

vals = list(data_dict.values())

# Initialise lists to hold all speakers' data
X = []
y = []
utt_lengths_matrix = []

for speaker in vals:
	try:
		grade = float(speaker[0])
	except:
		continue
	new_utts = []
	curr_utts = speaker[1]
	
	# Reject speakers with not exactly 21 utterances
	if len(curr_utts) != MAX_UTTS_PER_SPEAKER:
		continue
	
	# Create list to store the utterance lengths
	utt_lengths = []	

	for curr_utt in curr_utts:
		utt_len = len(curr_utt)

		if utt_len <= MAX_LEN_UTT:
			# append padding of zero vectors
			zeros_to_add = MAX_LEN_UTT - utt_len
			fb_zero_vec = [0]*FRAME_SIZE
			zero_vec = [fb_zero_vec]*zeros_to_add
			new_utt = curr_utt + zero_vec
			utt_lengths.append(utt_len)
		else:
			# shorten utterance from end
			new_utt = curr_utt[:MAX_LEN_UTT]
			utt_lengths.append(MAX_LEN_UTT)

		# Convert all values to float
		new_utt = [[float(i) for i in frame] for frame in new_utt]

					
		new_utts.append(new_utt)
	
	X.append(new_utts)
	y.append(grade)
	utt_lengths_matrix.append(utt_lengths)


# Store all data in a single list
D = [X, y, utt_lengths_matrix]

# Output data to a file
data_file = 'eval_data_padded_fb.txt'
with open(data_file, 'w') as f:
	f.truncate(0)
	f.write(json.dumps(D))














	

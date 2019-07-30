from rtk.data.kaldi.ark import read as read_ark

# Necessary to run command before running this script: source /home/dawna/ar527/tools/rtk/IMPORT
# Note rtk file written in python2

INPUT_FILE = "/home/alta/BLTSpeaking/exp-yw396/K5/kaldi-aug2017/feats/htk_fbk/BLXXXgrd02-fbk.ark"
grades_file_path = '/home/alta/BLTSpeaking/grd-graphemic-kmk-v2/GKTS4-D3/grader/BLXXXgrd02/score_10_mbr_rnnlm/data/grades.txt'
target_file = 'fb_data_training.txt'


def get_fb_vectors(INPUT_FILE):
	ark = read_ark(INPUT_FILE)

	# ark is a dictionary with key: speaker & utterance id, val: list of filter bank vectors - one for each frame

	return ark




# Get the dictionary of speaker&utt id to filter bank vectors
ark = get_fb_vectors(INPUT_FILE)

# 

# Create dictionary of keys as only speaker ids and the values as a list of utterances
speaker_dict = {}

for key in ark:
	speaker_id = key[:12]
	utt_id = key[22:28]
	
	
	if speaker_id in speaker_dict:
		speaker_dict[speaker_id][utt_id] = ark[key]
	else:
		speaker_dict[speaker_id] = {}
		speaker_dict[speaker_id][utt_id] = ark[key]

new_speaker_dict = {}
for speaker in speaker_dict:
	utt_dict = speaker_dict[speaker]
	utt_ids_list = list(utt_dict.keys())
	correct_order_utts = sorted(utt_ids_list)
	utts = [utt_dict[i] for i in correct_order_utts]
	new_speaker_dict[speaker] = utts



# Create dictionary with keys as speaker ids and the value a tuple: (grade, utts)

# Get grades
lines = [line.rstrip('\n') for line in open(grades_file_path)]

data_dict = {}
for line in lines:
	speaker_id = line[:12]
        grade = line[-3:]
        data_dict[speaker_id] = [grade, new_speaker_dict[speaker_id]]


# Write the data to a file
with open(target_file, 'w') as f:
                f.truncate(0)
                f.write(json.dumps(data_dict))




 



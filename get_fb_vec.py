from rtk.data.kaldi.ark import read as read_ark

# Necessary to run command before running this script: source /home/dawna/ar527/tools/rtk/IMPORT


INPUT_FILE = "/home/alta/BLTSpeaking/exp-yw396/K5/kaldi-aug2017/feats/htk_fbk/BLXXXgrd02-fbk.ark"

def get_fb_vectors():
	ark = read_ark(INPUT_FILE)

	# ark is a dictionary with key: speaker id, val: list of filter bank vectors - one for each frame

	return ark

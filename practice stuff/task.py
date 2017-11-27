from loadAndExtractAudio import parse_audio_files


parent_dir = 'sound_data'
tr_sub_dirs = ['train']
#ts_sub_dirs = ['fold3']
tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
print(tr_features)
#ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)
from loadAndExtractAudio import parse_audio_files


parent_dir = 'sound_data'
tr_sub_dirs = ['train']
ts_sub_dirs = ['test']
tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)
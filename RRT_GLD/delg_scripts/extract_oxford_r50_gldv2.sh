python extract_features.py \
  --delf_config_path r50delg_gldv2clean_config.pbtxt \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --images_dir data/oxford5k/jpg \
  --image_set query \
  --output_features_dir data/oxford5k/delg_r50_gldv2


python extract_features.py \
  --delf_config_path r50delg_gldv2clean_config.pbtxt \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --images_dir data/oxford5k/jpg \
  --image_set index \
  --output_features_dir data/oxford5k/delg_r50_gldv2
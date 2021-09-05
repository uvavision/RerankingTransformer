python extract_features.py \
  --delf_config_path r50delg_gld_config.pbtxt.pbtxt \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --images_dir data/oxford5k/jpg \
  --image_set query \
  --output_features_dir data/oxford5k/delg_r50_gldv1


python extract_features.py \
  --delf_config_path r50delg_gld_config.pbtxt.pbtxt \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --images_dir data/oxford5k/jpg \
  --image_set index \
  --output_features_dir data/oxford5k/delg_r50_gldv1
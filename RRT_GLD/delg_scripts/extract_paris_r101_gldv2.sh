python extract_features.py \
  --delf_config_path r101delg_gldv2clean_config.pbtxt \
  --dataset_file_path data/paris6k/gnd_rparis6k.mat \
  --images_dir data/paris6k/jpg \
  --image_set query \
  --output_features_dir data/paris6k/delg_r101_gldv2


python extract_features.py \
  --delf_config_path r101delg_gldv2clean_config.pbtxt \
  --dataset_file_path data/paris6k/gnd_rparis6k.mat \
  --images_dir data/paris6k/jpg \
  --image_set index \
  --output_features_dir data/paris6k/delg_r101_gldv2
python extract_features_gld.py \
  --delf_config_path r50delg_gld_config.pbtxt \
  --dataset_file_path $(GLDv2_ROOT)/train.txt \
  --images_dir $(GLDv2_ROOT) \
  --output_features_dir $(GLDv2_ROOT)/delg_r50_gldv1
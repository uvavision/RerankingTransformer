# Download the knn file if you haven't done so
# wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/gld_nn/nn_inds_r50_gldv2.pkl \
#     -P data/gldv2

python experiment.py -F logs/r50_gldv2 with temp_dir=logs/r50_gldv2 dataset.gldv2_roxford_r50_gldv2 model.RRT max_norm=0.0
# Download the knn file if you haven't done so
# wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/gld_nn/nn_inds_r101_gldv1.pkl \
#     -P data/gldv2

python experiment.py -F logs/r101_gldv1 with temp_dir=logs/r101_gldv1 dataset.gldv2_roxford_r101_gldv1 model.RRT max_norm=0.1
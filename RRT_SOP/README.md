# RerankingTransformers (RRTs): Experiments on Stanford Online Products

## About
This folder contains the code for training/evaluating RRTs on [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/).

The code is built on top of the [metric learning framework](https://github.com/jeromerony/dml_cross_entropy) provided by @jeromerony.

***
## Preparation
```
python prepare_data.py
```
This will download the [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/) (SOP) dataset to the ```data``` folder and create the split files.

<!-- If you'd like to run the experiment with ResNet50 as the backbone, please use the ```main``` branch.

```
git checkout main
```

If you'd like to run the experiment with SuperPoint as the backbone, please checkout the ```superpoint_sop``` branch:

```
git checkout superpoint_sop
``` -->

You can download the pretrained models and the KNN files from these links:

Pretrained models
```
https://www.dropbox.com/s/2gl17jrs3iozds4/rrt_sop_ckpts.zip?dl=0
```

KNN files (optional):
```
https://www.dropbox.com/s/ohrk1ew25ebdlfj/rrt_sop_caches.zip?dl=0
```

Please unzip them to the root directory (```RRT_SOP```).

***
## Experiments

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Repo branch</th>
<th valign="bottom">R@1</th>
<th valign="bottom">R@10</th>
<th valign="bottom">R@100</th>
<th valign="bottom">Evaluation</th>
<th valign="bottom">Training</th> 
<th valign="bottom">Log</th> 
<!-- TABLE BODY -->
<tr>
      <td align="left">Global retrieval</td>
      <td align="center">ResNet-50</td>
      <td align="center">main</td>
      <td align="center">80.7</td>
      <td align="center">91.9</td>
      <td align="center">96.6</td>
      <td align="center"><a href=#global-retrieval>script</a></td>
      <td align="center"><a href=#global-retrieval-1>script</a></td>
      <td align="center"></td>
</tr>
<tr>
      <td align="left">Reranking (frozen backbone)</td>
      <td align="center">ResNet-50</td>
      <td align="center">main</td>
      <td align="center">81.8</td>
      <td align="center">92.4</td>
      <td align="center">96.6</td>
      <td align="center"><a href=#reranking-with-a-frozen-backbone>script</a></td>
      <td align="center"><a href=#reranking-with-a-frozen-backbone-1>script</a></td>
      <td align="center"></td>
</tr>
<tr>
      <td align="left">Reranking (finetuned backbone)</td>
      <td align="center">ResNet-50</td>
      <td align="center">main</td>
      <td align="center">84.5</td>
      <td align="center">93.2</td>
      <td align="center">96.6</td>
      <td align="center"><a href=#reranking-with-a-finetuned-backbone>script</a></td>
      <td align="center"><a href=#reranking-with-a-finetuned-backbone-1>script</a></td>
      <td align="center"><a href=logs/finetune_log>log</a></td>
</tr>
</tbody></table>



<!-- ## Evaluation using ResNet50 as the backbone -->
## Evaluation

#### Global retrieval

```
python eval_global.py -F logs/eval_global_r50 with temp_dir=logs/eval_global_r50 \
      resume=rrt_sop_ckpts/rrt_r50_sop_global.pt dataset.sop_global model.resnet50
```

Please copy the knn file to the correct directory after the global retrieval evaluation:

```
cp logs/eval_global_r50/nn_inds.pkl rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl
```

#### Reranking with a frozen backbone

Run the evaluation:
```
python eval_rerank.py -F logs/eval_rerank_frozen_r50 with temp_dir=logs/eval_rerank_frozen_r50 \
      resume=rrt_sop_ckpts/rrt_r50_sop_rerank_frozen.pt dataset.sop_rerank model.resnet50 \
      cache_nn_inds=rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl
```

#### Reranking with a finetuned backbone

Run the evaluation:
```
python eval_rerank.py -F logs/eval_rerank_finetune_r50 with temp_dir=logs/eval_rerank_finetune_r50 \
      resume=rrt_sop_ckpts/rrt_r50_sop_rerank_finetune.pt dataset.sop_rerank model.resnet50 \
      cache_nn_inds=rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl
```

***
<!-- ## Training using ResNet50 as the backbone -->
## Training

#### Global retrieval
```
python experiment_global.py -F logs/train_global_r50 with temp_dir=logs/train_global_r50 \
            dataset.sop_global model.resnet50 dataset.batch_size=800 dataset.test_batch_size=800
```
You can consider reducing the batch size if it is too large for your experiment. 
A batch size of 400 can already achieve good performance.

#### Reranking with a frozen backbone

For each training image, we need its top-100 nearest neighbors from the global retrieval, you can download the [knn file](https://www.dropbox.com/s/ohrk1ew25ebdlfj/rrt_sop_caches.zip?dl=0), or generate the nearest neighbors file by running:
```
python eval_global.py -F logs/nn_file_for_training with temp_dir=logs/nn_file_for_training \
      resume=rrt_sop_ckpts/rrt_r50_sop_global.pt dataset.sop_global model.resnet50 \
      query_set='train'

cp logs/nn_file_for_training/nn_inds.pkl rrt_sop_caches/rrt_r50_sop_nn_inds_train.pkl
```

Run the training
```
python experiment_rerank.py -F logs/train_rerank_frozen_r50 with temp_dir=logs/train_rerank_frozen_r50 \
      dataset.sop_rerank model.resnet50 model.freeze_backbone=True \
      resume=rrt_sop_ckpts/rrt_r50_sop_global.pt
```

#### Reranking with a finetuned backbone
```
python experiment_rerank.py -F logs/train_rerank_finetune_r50 with temp_dir=logs/train_rerank_finetune_r50 \
      dataset.sop_rerank model.resnet50 model.freeze_backbone=False \
      resume=rrt_sop_ckpts/rrt_r50_sop_rerank_frozen.pt
```

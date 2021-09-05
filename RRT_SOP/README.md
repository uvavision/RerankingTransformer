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

If you'd like to run the experiment with ResNet50 as the backbone, please use the ```main``` branch.

```
git checkout main
```

If you'd like to run the experiment with SuperPoint as the backbone, please checkout the ```superpoint_sop``` branch:

```
git checkout superpoint_sop
```

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
</tr>
<tr>
      <td align="left">Global retrieval</td>
      <td align="center">SuperPoint</td>
      <td align="center">superpoint_sop</td>
      <td align="center">32.8</td>
      <td align="center">45.4</td>
      <td align="center">60.5</td>
      <td align="center"><a href=#global-retrieval-2>script</a></td>
      <td align="center">N.A.</td>
</tr>
<tr>
      <td align="left">Reranking (frozen backbone)</td>
      <td align="center">SuperPoint</td>
      <td align="center">superpoint_sop</td>
      <td align="center">50.2</td>
      <td align="center">57.9</td>
      <td align="center">60.5</td>
      <td align="center"><a href=#reranking-with-a-frozen-backbone-2>script</a></td>
      <td align="center"><a href=#reranking-with-a-frozen-backbone-3>script</a></td>
</tr>
<tr>
      <td align="left">Reranking (finetuned backbone)</td>
      <td align="center">SuperPoint</td>
      <td align="center">superpoint_sop</td>
      <td align="center">51.9</td>
      <td align="center">59.0</td>
      <td align="center">60.5</td>
      <td align="center"><a href=#reranking-with-a-finetuned-backbone-2>script</a></td>
      <td align="center"><a href=#reranking-with-a-finetuned-backbone-3>script</a></td>
</tr>
</tbody></table>



## Evaluation using ResNet50 as the backbone

#### Global retrieval
```
# Download the pretrained model
wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_ckpts/rrt_r50_sop_global.pt -P rrt_sop_ckpts/

python eval_global.py -F logs/eval_global_r50 with temp_dir=logs/eval_global_r50 \
      resume=rrt_sop_ckpts/rrt_r50_sop_global.pt dataset.sop_global model.resnet50
```

#### Reranking with a frozen backbone

Download the pretrained model:
```
wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_ckpts/rrt_r50_sop_rerank_frozen.pt \
      -P rrt_sop_ckpts/
```

If you have run the global retrieval evaluation:
```
cp logs/eval_global_r50/nn_inds.pkl rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl
```

Otherwise:
```
wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl \
      -P rrt_sop_caches/
```

Run the evaluation:
```
python eval_rerank.py -F logs/eval_rerank_frozen_r50 with temp_dir=logs/eval_rerank_frozen_r50 \
      resume=rrt_sop_ckpts/rrt_r50_sop_rerank_frozen.pt dataset.sop_rerank model.resnet50 \
      cache_nn_inds=rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl
```

#### Reranking with a finetuned backbone

Download the pretrained model:
```
wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_ckpts/rrt_r50_sop_rerank_finetune.pt \
      -P rrt_sop_ckpts/
```

If you have run the global retrieval evaluation:
```
cp logs/eval_global_r50/nn_inds.pkl rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl
```

Otherwise:
```
wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl \
      -P rrt_sop_caches/
```

Run the evaluation:
```
python eval_rerank.py -F logs/eval_rerank_finetune_r50 with temp_dir=logs/eval_rerank_finetune_r50 \
      resume=rrt_sop_ckpts/rrt_r50_sop_rerank_finetune.pt dataset.sop_rerank model.resnet50 \
      cache_nn_inds=rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl
```

***
## Training using ResNet50 as the backbone

#### Global retrieval
```
python experiment_global.py -F logs/train_global_r50 with temp_dir=logs/train_global_r50 \
            dataset.sop_global model.resnet50 dataset.batch_size=800 dataset.test_batch_size=800
```
You can consider reducing the batch size if it is too large for your experiment. 
A batch size of 400 can already achieve good performance.

#### Reranking with a frozen backbone

For each training image, we need its top-100 nearest neighbors from the global retrieval, you can generate the nearest neighbors file by running:
```
# Download the pretrained global model if you haven't  
# wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_ckpts/rrt_r50_sop_global.pt -P rrt_sop_ckpts/

python eval_global.py -F logs/nn_file_for_training with temp_dir=logs/nn_file_for_training \
      resume=rrt_sop_ckpts/rrt_r50_sop_global.pt dataset.sop_global model.resnet50 \
      query_set='train'

cp logs/nn_file_for_training/nn_inds.pkl rrt_sop_caches/rrt_r50_sop_nn_inds_train.pkl
```

Or download it:
```
wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_caches/rrt_r50_sop_nn_inds_train.pkl \
      -P rrt_sop_caches/
```

Run the training
```
# Download the pretrained global model if you haven't  
# wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_ckpts/rrt_r50_sop_global.pt -P rrt_sop_ckpts/

python experiment_rerank.py -F logs/train_rerank_frozen_r50 with temp_dir=logs/train_rerank_frozen_r50 \
      dataset.sop_rerank model.resnet50 model.freeze_backbone=True \
      resume=rrt_sop_ckpts/rrt_r50_sop_global.pt
```

#### Reranking with a finetuned backbone
```
# Download the pretrained model with a frozen backbone 
wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_ckpts/rrt_r50_sop_rerank_frozen.pt \
      -P rrt_sop_ckpts/


python experiment_rerank.py -F logs/train_rerank_finetune_r50 with temp_dir=logs/train_rerank_finetune_r50 \
      dataset.sop_rerank model.resnet50 model.freeze_backbone=False \
      resume=rrt_sop_ckpts/rrt_r50_sop_rerank_frozen.pt
```


***
## Evaluation using [SuperPoint](https://arxiv.org/abs/1712.07629) as the backbone

#### Global retrieval

Download the keypoint locations extracted by [SuperPoint](https://arxiv.org/abs/1712.07629):
```
wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_caches/superpoint_sop_320_500.pkl \
      -P rrt_sop_caches/
```

Run the evaluation:
```
python eval_global.py -F logs/eval_sop_global_super with temp_dir=logs/eval_sop_global_super
```

#### Reranking with a frozen backbone

Download the pretrained model:
```
wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_ckpts/rrt_superpoint_sop_rerank_nopos_frozen.pt \
      -P rrt_sop_ckpts/
```

If you have run the global retrieval evaluation:
```
cp logs/eval_sop_global_super/nn_inds.pkl rrt_sop_caches/rrt_superpoint_sop_nn_inds_test.pkl
```

Otherwise:
```
wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_caches/rrt_superpoint_sop_nn_inds_test.pkl \
      -P rrt_sop_caches/
```

Run the evaluation:
```
python eval_rerank.py -F logs/eval_sop_local_super_frozen with temp_dir=eval_sop_local_super_frozen \
            resume=rrt_sop_ckpts/rrt_superpoint_sop_rerank_nopos_frozen.pt
```

#### Reranking with a finetuned backbone

Download the pretrained model:
```
wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_ckpts/rrt_superpoint_sop_rerank_nopos_finetune.pt \
      -P rrt_sop_ckpts/
```

If you have run the global retrieval evaluation:
```
cp logs/eval_sop_global_super/nn_inds.pkl rrt_sop_caches/rrt_superpoint_sop_nn_inds_test.pkl
```

Otherwise:
```
wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_caches/rrt_superpoint_sop_nn_inds_test.pkl \
      -P rrt_sop_caches/
```

Run the evaluation:
```
python eval_rerank.py -F logs/eval_sop_local_super_finetune with temp_dir=eval_sop_local_super_finetune \
            resume=rrt_sop_ckpts/rrt_superpoint_sop_rerank_nopos_finetune.pt
```

***
## Training using [SuperPoint](https://arxiv.org/abs/1712.07629) as the backbone


#### Reranking with a frozen backbone

For each training image, we need its top-100 nearest neighbors from the global retrieval, you can generate the nearest neighbors file by running:
```
python eval_global.py -F logs/nn_file_for_training_super with temp_dir=logs/nn_file_for_training_super \
            query_set='train'
cp logs/nn_file_for_training_super/nn_inds.pkl rrt_sop_caches/rrt_superpoint_sop_nn_inds_train.pkl
```

Or download it:
```
wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_caches/rrt_superpoint_sop_nn_inds_train.pkl \
      -P rrt_sop_caches/
```

Run the training
```
python experiment_rerank.py -F logs/train_rerank_frozen_superpoint with temp_dir=logs/train_rerank_frozen_superpoint \
      model.freeze_backbone=True dataset.test_file=mini_test.txt \
      cache_nn_inds=rrt_sop_caches/rrt_superpoint_sop_nn_inds_mini_test.pkl
```

#### Reranking with a finetuned backbone
```
# Download the pretrained model with a frozen backbone 
wget www.cs.virginia.edu/~ft3ex/data/rrt_iccv2021_data/rrt_sop_ckpts/rrt_superpoint_sop_rerank_nopos_frozen.pt \
      -P rrt_sop_ckpts/
```
      
Run the training
```
python experiment_rerank.py -F logs/train_rerank_finetune_superpoint with temp_dir=logs/train_rerank_finetune_superpoint \
      model.freeze_backbone=False dataset.test_file=mini_test.txt \
      cache_nn_inds=rrt_sop_caches/rrt_superpoint_sop_nn_inds_mini_test.pkl \
      resume=rrt_sop_ckpts/rrt_superpoint_sop_rerank_nopos_frozen.pt
```


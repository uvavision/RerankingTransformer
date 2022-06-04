# RerankingTransformers (RRTs): Experiments on Google Landmarks v2, Revisited Oxford/Paris

## About
This folder contains the code for training/evaluating RRTs using the pretrained [DELG descriptors](https://arxiv.org/abs/2001.05027).

The code is built on top of the [metric learning framework](https://github.com/jeromerony/dml_cross_entropy) provided by @jeromerony.

***
## Preparing the descriptors

```diff
@@ Please create a separate python virtual environment for this task. @@
```

### Extraction scripts wrap-up
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Backbone</th>
<th valign="bottom">Desc. version</th>
<th valign="bottom">Google Landmarks v2</th>
<th valign="bottom">Revisited Oxford5k</th>
<th valign="bottom">Revisited Paris6k</th>
<!-- TABLE BODY -->
<tr>
      <td align="center">ResNet-50</td>
      <td align="center">v1</td>
      <td align="center"><a href=delg_scripts/extract_gldv2_r50_gldv1.sh>script</a></td>
      <td align="center"><a href=delg_scripts/extract_oxford_r50_gldv1.sh>script</a></td>
      <td align="center"><a href=delg_scripts/extract_paris_r50_gldv1.sh>script</a></td>
</tr>
<tr>
      <td align="center">ResNet-50</td>
      <td align="center">v2-clean</td>
      <td align="center"><a href=delg_scripts/extract_gldv2_r50_gldv2.sh>script</a></td>
      <td align="center"><a href=delg_scripts/extract_oxford_r50_gldv2.sh>script</a></td>
      <td align="center"><a href=delg_scripts/extract_paris_r50_gldv2.sh>script</a></td>
</tr>
<tr>
      <td align="center">ResNet-101</td>
      <td align="center">v1</td>
      <td align="center"><a href=delg_scripts/extract_gldv2_r101_gldv1.sh>script</a></td>
      <td align="center"><a href=delg_scripts/extract_oxford_r101_gldv1.sh>script</a></td>
      <td align="center"><a href=delg_scripts/extract_paris_r101_gldv1.sh>script</a></td>
</tr>
<tr>
      <td align="center">ResNet-101</td>
      <td align="center">v2-clean</td>
      <td align="center"><a href=delg_scripts/extract_gldv2_r101_gldv2.sh>script</a></td>
      <td align="center"><a href=delg_scripts/extract_oxford_r101_gldv2.sh>script</a></td>
      <td align="center"><a href=delg_scripts/extract_paris_r101_gldv2.sh>script</a></td>
</tr>
</tbody></table>

### Install the DELG package
Please follow the [instruction](https://github.com/tensorflow/models/blob/master/research/delf/INSTALL_INSTRUCTIONS.md) to install the DELG library. 

All the instructions below assume [the DELG package](https://github.com/tensorflow/models/tree/master/research/delf) is installed in `DELG_ROOT`.

### Extract the features of Revisited Oxford/Paris

```
cd $(DELG_ROOT)/delf/python/delg
```

Please follow the [instruction](https://github.com/tensorflow/models/blob/master/research/delf/delf/python/delg/DELG_INSTRUCTIONS.md) to extract the features of Revisited Oxford/Paris. 
The [table](#extraction-scripts-wrap-up) above also provides example scripts to help extract the features. The scripts will not work out-of-the-box, you will still need to set the paths of the input/output directories properly. Please refer to the [instruction](https://github.com/tensorflow/models/blob/master/research/delf/delf/python/delg/DELG_INSTRUCTIONS.md) for more details.

### Extract the features of Google Landmarks v2, if you'd like to train the model from scratch 

```
cd $(DELG_ROOT)/delf/python/delg
```

Download the training set of Google Landmarks v2 from [CVDF](https://github.com/cvdfoundation/google-landmark), or [Kaggle](https://www.kaggle.com/c/landmark-recognition-2021/data). We call the directory of the downloaded dataset as `GLDv2_ROOT`.

Copy-paste the file of the training split [`train.txt`](https://github.com/fwtan/RRT_ICCV2021/blob/main/RRT_GLD/data/gldv2/train.txt) to `GLDv2_ROOT`, and the python script [`extract_features_gld.py`](https://github.com/fwtan/RRT_ICCV2021/blob/main/RRT_GLD/delg_scripts/extract_features_gld.py) to `DELG_ROOT/delf/python/delg`.

Run the scripts shown in the [table](#extraction-scripts-wrap-up) above. Again, the scripts may not work out-of-the-box, you may still need to set the paths of the input/output directories properly.


<!-- *The users are suggested to extract the features themselves by following the instructions above. 
We provide the links of the extracted descriptors of `delg_r50_gldv1` in the [table](#extraction-scripts-wrap-up) (the 1st row) above as an example. 
Given the sizes of the descriptors and the network condition of our server, downloading/unziping the descriptors may be much slower than extracting the descriptors directly.*
 -->

### Dataset structure

The code assumes datasets in the structure described below. Note that, we don't need the image data for our experiments.

<h4>GLDv2<a class="headerlink" href="#gldv2" title="Permalink to this headline">¶</a></h4>
<div class="highlight-default notranslate">
      <div class="highlight">
      <pre>
      <span></span><span class="n">data</span><span class="o">/</span><span class="n">gldv2</span><span class="o">/</span>
          <span class="n">train.txt</span><span class="o">/</span>
          <span class="n">delg_r50_gldv1</span><span class="o">/</span>
          <span class="n">delg_r50_gldv2</span><span class="o">/</span>
          <span class="n">delg_r101_gldv1</span><span class="o">/</span>
          <span class="n">delg_r101_gldv2</span><span class="o">/</span>
      </pre>
      </div>
</div>

      
<h4>Revisited Oxford<a class="headerlink" href="#revisited-oxford" title="Permalink to this headline">¶</a></h4>
<div class="highlight-default notranslate">
      <div class="highlight">
      <pre>
      <span></span><span class="n">data</span><span class="o">/</span><span class="n">oxford5k</span><span class="o">/</span>
          <span class="n">test_query.txt</span><span class="o">/</span>
          <span class="n">test_gallery.txt</span><span class="o">/</span>
          <span class="n">gnd_roxford5k.pkl</span><span class="o">/</span>
          <span class="n">delg_r50_gldv1</span><span class="o">/</span>
          <span class="n">delg_r50_gldv2</span><span class="o">/</span>
          <span class="n">delg_r101_gldv1</span><span class="o">/</span>
          <span class="n">delg_r101_gldv2</span><span class="o">/</span>
      </pre>
      </div>
</div>

      
<h4>Revisited Paris<a class="headerlink" href="#revisited-paris" title="Permalink to this headline">¶</a></h4>
<div class="highlight-default notranslate">
      <div class="highlight">
      <pre>
      <span></span><span class="n">data</span><span class="o">/</span><span class="n">paris6k</span><span class="o">/</span>
          <span class="n">test_query.txt</span><span class="o">/</span>
          <span class="n">test_gallery.txt</span><span class="o">/</span>
          <span class="n">gnd_rparis6k.pkl</span><span class="o">/</span>
          <span class="n">delg_r50_gldv1</span><span class="o">/</span>
          <span class="n">delg_r50_gldv2</span><span class="o">/</span>
          <span class="n">delg_r101_gldv1</span><span class="o">/</span>
          <span class="n">delg_r101_gldv2</span><span class="o">/</span>
      </pre>
      </div>
</div>

Here, the [`gnd_roxford5k.pkl`](https://github.com/fwtan/RRT_ICCV2021/blob/main/RRT_GLD/data/oxford5k/gnd_roxford5k.pkl) and [`gnd_rparis6k.pkl`](https://github.com/fwtan/RRT_ICCV2021/blob/main/RRT_GLD/data/paris6k/gnd_rparis6k.pkl) files are already included in the repo, they can also be downloaded from the [Revisiting Oxford and Paris page](http://cmp.felk.cvut.cz/revisitop/). The `test_query.txt` and `test_gallery.txt` files are also included in the repo. They were generated by running (you do need the image data to run this):

```
python tools/prepare_data.py
```
***
      
## Experiments

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Desc. version</th>
<th valign="bottom">mAP<br/>medium / hard</th>
<th valign="bottom">Evaluation</th>
<th valign="bottom">Training</th>  
<th valign="bottom">Log</th>     
<!-- TABLE BODY -->
<tr>
      <td align="left">Global retrieval</td>
      <td align="center">ResNet-50</td>
      <td align="center">v1</td>
      <td align="center">ROxf: 69.7 / 45.1<br/>RPar: 81.6 / 63.4</td>
      <td align="center">ROxf: <a href=rrt_scripts/eval_r50_gldv1_roxf_global.sh>script</a><br/>RPar: <a href=rrt_scripts/eval_r50_gldv1_rpar_global.sh>script</a></td>
      <td align="center">N.A.</td>
      <td align="center"></td>
</tr>
<tr>
      <td align="left">Reranking</td>
      <td align="center">ResNet-50</td>
      <td align="center">v1</td>
      <td align="center">ROxf: 75.5 / 56.4<br/>RPar: 82.7 / 68.6</td>
      <td align="center">ROxf: <a href=rrt_scripts/eval_r50_gldv1_roxf_rerank.sh>script</a><br/>RPar: <a href=rrt_scripts/eval_r50_gldv1_rpar_rerank.sh>script</a></td>
      <td align="center"><a href=rrt_scripts/train_r50_gldv1.sh>script</a></td>
      <td align="center"><a href=logs/r50_gldv1_log_retrained>log (retrained)</a></td>
</tr>
<tr>
      <td align="left">Global retrieval</td>
      <td align="center">ResNet-50</td>
      <td align="center">v2-clean</td>
      <td align="center">ROxf: 73.6 / 51.0<br/>RPar: 85.7 / 71.5</td>
      <td align="center">ROxf: <a href=rrt_scripts/eval_r50_gldv2_roxf_global.sh>script</a><br/>RPar: <a href=rrt_scripts/eval_r50_gldv2_rpar_global.sh>script</a></td>
      <td align="center">N.A.</td>
      <td align="center"></td>
</tr>
<tr>
      <td align="left">Reranking</td>
      <td align="center">ResNet-50</td>
      <td align="center">v2-clean</td>
      <td align="center">ROxf: 78.1 / 60.2<br/>RPar: 86.7 / 75.1</td>
      <td align="center">ROxf: <a href=rrt_scripts/eval_r50_gldv2_roxf_rerank.sh>script</a><br/>RPar: <a href=rrt_scripts/eval_r50_gldv2_rpar_rerank.sh>script</a></td>
      <td align="center"><a href=rrt_scripts/train_r50_gldv2.sh>script</a></td>
      <td align="center"></td>
</tr>
<tr>
      <td align="left">Global retrieval</td>
      <td align="center">ResNet-101</td>
      <td align="center">v1</td>
      <td align="center">ROxf: 73.2 / 51.2<br/>RPar: 82.4 / 64.7</td>
      <td align="center">ROxf: <a href=rrt_scripts/eval_r101_gldv1_roxf_global.sh>script</a><br/>RPar: <a href=rrt_scripts/eval_r101_gldv1_rpar_global.sh>script</a></td>
      <td align="center">N.A.</td>
      <td align="center"></td>
</tr>
<tr>
      <td align="left">Reranking</td>
      <td align="center">ResNet-101</td>
      <td align="center">v1</td>
      <td align="center">ROxf: 78.8 / 62.5<br/>RPar: 83.2 / 68.4</td>
      <td align="center">ROxf: <a href=rrt_scripts/eval_r101_gldv1_roxf_rerank.sh>script</a><br/>RPar: <a href=rrt_scripts/eval_r101_gldv1_rpar_rerank.sh>script</a></td>
      <td align="center"><a href=rrt_scripts/train_r101_gldv1.sh>script</a></td>
      <td align="center"></td>
</tr>
<tr>
      <td align="left">Global retrieval</td>
      <td align="center">ResNet-101</td>
      <td align="center">v2-clean</td>
      <td align="center">ROxf: 76.3 / 55.6<br/>RPar: 86.6 / 72.4</td>
      <td align="center">ROxf: <a href=rrt_scripts/eval_r101_gldv2_roxf_global.sh>script</a><br/>RPar: <a href=rrt_scripts/eval_r101_gldv2_rpar_global.sh>script</a></td>
      <td align="center">N.A.</td>
      <td align="center"></td>
</tr>
<tr>
      <td align="left">Reranking</td>
      <td align="center">ResNet-101</td>
      <td align="center">v2-clean</td>
      <td align="center">ROxf: 79.9 / 64.1<br/>RPar: 87.6 / 76.1</td>
      <td align="center">ROxf: <a href=rrt_scripts/eval_r101_gldv2_roxf_rerank.sh>script</a><br/>RPar: <a href=rrt_scripts/eval_r101_gldv2_rpar_rerank.sh>script</a></td>
      <td align="center"><a href=rrt_scripts/train_r101_gldv2.sh>script</a></td>
      <td align="center"><a href=logs/r101_gldv2_log>log</a></td>
</tr>
</tbody></table>
  
***
## Evaluation

### global retrieval

```
python tools/prepare_topk_revisited.py with dataset_name=[oxford5k|paris6k] \
    feature_name=[r50_gldv1 | r50_gldv2 | r101_gldv1 | r101_gldv2] \
    gnd_name=[gnd_roxford5k.pkl | gnd_rparis6k.pkl]
```

Please specify the `dataset_name`, `feature_name`, and ground-truth filename `gnd_name` accordingly.

This command will generate the nearest neighbor file to the dataset folder. 

You can also check the specific command included in the [table](#experiments) above.


### Reranking

```
python evaluate_revisited.py with model.RRT
    dataset.[roxford|rparis]_[r50|r101]_[gldv1|gldv2] \
    resume=rrt_gld_ckpts/[r50|r101]_[gldv1|gldv2].pt
```

Please specify the dataset, desc. version, and the checkpoint accordingly. 

All the pretrained weights are included in the repo.

Note that reranking requires the nearest neighbor file generated from global retrieval, so please run the global retrieval script first.

You can also check the specific command included in the [table](#experiments) above.

***
## Training

In order to train RRTs, we need the top100 nearest neighbors for each training image. 

You can generate them by running (it may take hours):

```
python tools/prepare_topk_gldv2.py with feature_name=[r50|r101]_[gldv1|gldv2]
``` 

Run the training:

```
python experiment.py with dataset.gldv2_roxford_[r50|r101]_[gldv1|gldv2] model.RRT max_norm=[0.0|0.1]
```

The training scripts for the specific models are shown in the [table](#experiments) above. 
      

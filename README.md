# [Instance-level Image Retrieval using Reranking Transformers](https://arxiv.org/abs/2103.12236)
Fuwen Tan, Jiangbo Yuan, Vicente Ordonez, ICCV 2021.

## Abstract
Instance-level image retrieval is the task of searching in a large database for images that match an object in a query image. To address this task, systems usually rely on a retrieval step that uses global image descriptors, and a subsequent step that performs domain-specific refinements or reranking by leveraging operations such as geometric verification based on local features. In this work, we propose Reranking Transformers (RRTs) as a general model to incorporate both local and global features to rerank the matching images in a supervised fashion and thus replace the relatively expensive process of geometric verification. RRTs are lightweight and can be easily parallelized so that reranking a set of top matching results can be performed in a single forward-pass. We perform extensive experiments on the Revisited Oxford and Paris datasets, and the Google Landmark v2 dataset, showing that RRTs outperform previous reranking approaches while using much fewer local descriptors. Moreover, we demonstrate that, unlike existing approaches, RRTs can be optimized jointly with the feature extractor, which can lead to feature representations tailored to downstream tasks and further accuracy improvements. 

## Software required
The code is only tested on Linux 64:

```
  conda create -n rrt python=3.6
  conda activate rrt
  pip install -r requirements.txt
```

## Organization

To use the code for experiments on Google Landmarks v2, Revisited Oxford/Paris, please refer to the folder [RRT_GLD](RRT_GLD).

To use the code for experiments on Stanford Online Products, please refer to the folder [RRT_SOP](RRT_SOP).

To use the code for evaluating SuperGlue on Revisited Oxford/Paris and Stanford Online Products, please refer to the repo [SuperGlue](https://github.com/uvavision/SuperGluePretrainedNetwork).


## Citing

If you find our paper/code useful, please consider citing:
  
	@inproceedings{fwtan-instance-2021,
        author = {Fuwen Tan and Jiangbo Yuan and Vicente Ordonez},
        title = {Instance-level Image Retrieval using Reranking Transformers},
        year = {2021},
        booktitle = {International Conference on Computer Vision (ICCV)}
     }

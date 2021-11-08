# Home Action Genome

The repository for our work [Home Action Genome: Cooperative Contrastive Action Understanding](https://arxiv.org/abs/2105.05226) presented in CVPR '21.

This repository contains the implementation of Home Action Genome: Cooperative Compositional Contrastive Learning. We release a multi-view action dataset with multiple modalities and view-points supplemented with hierarchical activity and atomic action labels together with dense scene composition labels, along with a supplementary approach to leverage such rich annotations.

> Existing research on action recognition treats activities as monolithic events occurring in videos. Recently, the benefits of formulating actions as a combination of atomic-actions have shown promise in improving action understanding with the emergence of datasets containing such annotations, allowing us to learn representations capturing this information. However, there remains a lack of studies that extend action composition and leverage multiple viewpoints and multiple modalities of data for representation learning. To promote research in this direction, we introduce Home Action Genome (HOMAGE): a multi-view action dataset with multiple modalities and view-points supplemented with hierarchical activity and atomic action labels together with dense scene composition labels. Leveraging rich multi-modal and multi-view settings, we propose Cooperative Compositional Action Understanding (CCAU), a cooperative learning framework for hierarchical action recognition that is aware of compositional action elements. CCAU shows consistent performance improvements across all modalities. Furthermore, we demonstrate the utility of co-learning compositions in few-shot action recognition by achieving 28.6% mAP with just a single sample.

### Installation

Our implementation should work with python >= 3.6, pytorch >= 0.4, torchvision >= 0.2.2. The repo also requires cv2
 (`conda install -c menpo opencv`), tensorboardX >= 1.7 (`pip install tensorboardX`), tqdm.

A requirements.txt has been provided which can be used to create the exact environment required.
  ```
  pip install -r requirements.txt
  ```

### Prepare data

Follow the instructions [here](process_data/).

## Data

### Multi Camera Perspective Videos

![homage_dataset](https://user-images.githubusercontent.com/7645118/123186633-6a8b0c80-d44d-11eb-8928-82fbf3d06eb7.png)

### Atomic Action Annotations

![dataset_annotations](https://user-images.githubusercontent.com/7645118/123186626-65c65880-d44d-11eb-85b9-9bc1a15102a1.png)

### Frame-level Scene Graph Annotation

![scene_graph](https://user-images.githubusercontent.com/7645118/123186630-68c14900-d44d-11eb-84bb-523edc580ba1.png)

### Cooperative Compositional Contrastive Learning (CCAU)

Training scripts are present in `cd homage/train/`

Run `python model_trainer.py --help` to get details about the command lines args. The most useful ones are `--dataset` and `--modalities`, which are used to change the dataset we're supposed to run our experiments along with the input modalities to use.
 
Our implementation has been tested with Ego-view RGB Images, Third-Person view RGB Images and Audio. However, it is easy to extend it to custom views; look at `dataset_3d.py` for details.

* Single View Training: train CCAU using 2 GPUs, using only ego and third-person RGB inputs, with a 3D-ResNet18 backbone, with 224x224 resolution, for 100 epochs. Batch size is per-gpu.
  ```
  CUDA_VISIBLE_DEVICES="0,1" python model_trainer.py --net resnet18 --modalities imgs 
  --batch_size 16 --img_dim 224 --epochs 100
  ```

* Multi-View Training: train CCAU using 4 GPUs, using ego and third-person RGB views and Audio with a 3D-ResNet18 backbone, with 128x128 resolution, for 100 epochs
  ```
  CUDA_VISIBLE_DEVICES="0,1,2,3" python model_trainer.py --net resnet18 --modalities imgs_audio --batch_size 16 --img_dim 128 --epochs 100
  ```

### Evaluation: Video Action Recognition

Testing scripts are present as part of `homage/train/model_trainer.py` under the `--test` flag as well as in scripts present in `cd homage/test/`.

## Citing

If our paper or dataset was useful to you, please consider citing it using the below.
~~~
@InProceedings{Rai_2021_CVPR,
    author    = {Rai, Nishant and Chen, Haofeng and Ji, Jingwei and Desai, Rishi and Kozuka, Kazuki and Ishizaka, Shun and Adeli, Ehsan and Niebles, Juan Carlos},
    title     = {Home Action Genome: Cooperative Compositional Action Understanding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {11184-11193}
}
~~~

### Acknowledgements

Portions of code have been borrowed from [CoCon](https://github.com/nishantrai18/cocon). Feel free to refer to it as well if you're interested in the field.

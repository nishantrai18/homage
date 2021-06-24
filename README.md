# Home Action Genome

The repository for our work [Home Action Genome: Cooperative Contrastive Action Understanding](https://arxiv.org/abs/2105.05226) presented in CVPR '21.

Code and dataloaders for the dataset will be made available here.

> Existing research on action recognition treats activities as monolithic events occurring in videos. Recently, the benefits of formulating actions as a combination of atomic-actions have shown promise in improving action understanding with the emergence of datasets containing such annotations, allowing us to learn representations capturing this information. However, there remains a lack of studies that extend action composition and leverage multiple viewpoints and multiple modalities of data for representation learning. To promote research in this direction, we introduce Home Action Genome (HOMAGE): a multi-view action dataset with multiple modalities and view-points supplemented with hierarchical activity and atomic action labels together with dense scene composition labels. Leveraging rich multi-modal and multi-view settings, we propose Cooperative Compositional Action Understanding (CCAU), a cooperative learning framework for hierarchical action recognition that is aware of compositional action elements. CCAU shows consistent performance improvements across all modalities. Furthermore, we demonstrate the utility of co-learning compositions in few-shot action recognition by achieving 28.6% mAP with just a single sample.

## Data

### Multi Camera Perspective Videos

![homage_dataset](https://user-images.githubusercontent.com/7645118/123186633-6a8b0c80-d44d-11eb-8928-82fbf3d06eb7.png)

### Atomic Action Annotations

![dataset_annotations](https://user-images.githubusercontent.com/7645118/123186626-65c65880-d44d-11eb-85b9-9bc1a15102a1.png)

### Frame-level Scene Graph Annotation

![scene_graph](https://user-images.githubusercontent.com/7645118/123186630-68c14900-d44d-11eb-84bb-523edc580ba1.png)

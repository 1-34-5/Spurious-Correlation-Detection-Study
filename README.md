​
Spurious Correlation Study in Vision Models
This repository contains the code and experiments for a study on spurious correlations and bias in image classifiers trained on the Oxford-IIIT Pet dataset.
The project trains ResNet-50, Swin-Tiny, and DeiT-Small models and evaluates how strongly they rely on background vs foreground using counterfactual images and multiple explainability methods.


Project overview
Goal: Analyse to what extent modern CNNs and ViTs learn spurious background correlations instead of robust object-centric features.

Dataset: Oxford-IIIT Pet (37 classes) with trimap annotations for foreground masks.

Models:

ResNet-50 (CNN baseline)

Swin-Tiny (hierarchical ViT)

DeiT-Small (token-based ViT)

Key ideas:

Generate counterfactual images by compositing pet foregrounds onto 15 different backgrounds.

Measure accuracy drop (∆Acc) between original and counterfactual images as a quantitative signal of background dependence.

Use Grad-CAM, Saliency, Integrated Gradients, and Occlusion to compute foreground attention ratios (FAR) with respect to ground-truth trimaps.

Compare robustness and bias patterns across architectures.

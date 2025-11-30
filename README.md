
# Scalable Class-Incremental Learning Based on Parametric Neural Collapse (SCL-PNC)

## 🌟 Abstract

Class-Incremental Learning (CIL) often encounters challenges such as overfitting to new data and **catastrophic forgetting** of old data. Existing model expansion methods, while effective, can ignore structural efficiency, leading to **feature differences** between modules and **class distribution mapping bias**.

To address these issues, we propose the **Scalable Class-Incremental Learning based on Parametric Neural Collapse (SCL-PNC)** method. SCL-PNC enables demand-driven, minimal-cost backbone expansion through two core mechanisms:

1.  A **Dynamic Parametric Equiangular Tight Frame (ETF) Classifier** to mitigate the class distribution mapping bias.
2.  An **Adapt-layer** combined with a **parallel expansion framework** and **knowledge distillation** to align features across expanded modules, thereby counteracting **feature drift** and ensuring feature consistency.

SCL-PNC leverages the phenomenon of **Neural Collapse** to structurally guide the convergence of the incrementally expanded model. Our method demonstrates superior effectiveness and efficiency on standard benchmarks.

## 💡 Key Contributions

Our primary contributions are summarized as follows:

* **Adaptive Feature Alignment:** We introduce an **Adapt-layer** built on knowledge distillation to constrain feature vector prototypes from the backbone to align with classifier prototypes, effectively alleviating feature consistency deviation between modules.
* **Dynamic Classifier Design:** We propose a novel **Dynamic Parametric ETF Classifier** to address the distribution mapping bias, ensuring the classifier can scale its vectors dynamically in accordance with the increasing number of incremental classes.
* **Superior Performance:** SCL-PNC significantly outperforms State-of-the-Art (SOTA) methods on both small-scale (CIFAR-100) and large-scale (ImageNet-100) datasets. Our **parallel expansion framework** also shows an inherent architectural advantage over serial expansion methods.

## 🧠 Methodology

SCL-PNC is built around an incrementally expandable model backbone, guided by three critical components:

### 1. Expandable Model Backbone (EM)

The backbone is composed of a **frozen Base-layer** (for general features) and **trainable Expand-layers** (for task-specific information).

* Uses a **parallel expansion framework** with a **dual-source input strategy** for each new Expand-layer.
* Incorporates **Knowledge Distillation ($\mathcal{L}_{distill}$)** between continuous expansion modules to maintain the continuity of old task knowledge.

### 2. Adaptive Layer (Adapt-layer)

The Adapt-layer acts as a feature space transformation bridge, guiding the features towards the vertices defined by the ETF classifier.

* It imposes constraints on the feature vector prototypes to align with the corresponding classifier prototypes, thus **inducing Neural Collapse** within the backbone.
* Implemented using an **MLP (Multi-Layer Perceptron)** for high efficiency.

### 3. Parametric ETF Classifier

The classifier replaces the traditional fully-connected layer to leverage the properties of Neural Collapse.

* It is **dynamically expandable**, meaning the number of class prototypes ($K$) scales automatically with the arrival of new incremental classes.

## 📊 Experimental Results

### Datasets and Metrics

* **Datasets:** CIFAR-100 and ImageNet-100.
* **Evaluation:** Incremental Accuracy Curve and **Average Recognition Accuracy ($\bar{A}$)**.

### Performance Highlights

SCL-PNC demonstrates significant performance gains. For instance, on the CIFAR-100 dataset using the `B50Inc10` strategy (Base 50 classes, 10 classes incremental per step), SCL-PNC achieves an **Average Accuracy of 70.92%**.

| Strategy | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | **Average ($\bar{A}$)** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| B50Inc10 | 78.62 | 74.67 | 73.04 | 68.20 | 66.28 | 64.69 | **70.92** |
| B50Inc5 | 78.62 | 75.76 | 74.80 | 73.02 | 72.61 | ... | **70.82** |
| B10Inc10 | 91.10 | 79.40 | 74.37 | 68.45 | ... | ... | **64.90** |

## 💻 Environment and Reproduction

### Prerequisites

* Python (Recommended version 3.8+)
* PyTorch (Tested with recent versions)
* PyCIL library (for standardized benchmarking)
* A memory buffer of 3312 examples is required for the exemplar replay component.

### Training Settings

* **Optimizer:** SGD
* **Initial Learning Rate:** 0.1
* **SGD Momentum:** 0.9
* **Batch Size:** 128 (for both base and incremental tasks)
* **Total Epochs:** 200
* **LR Schedule:** Decayed by 0.01 every 20 training epochs
* **Augmentation:** Standard data augmentation including random crop, horizontal flip, and color jitter.
* **Training Scripts:**The main training script is main_memo.py
  * **Instructions:**
  *  After unzipping the downloaded ImageNet-100 dataset, place the `train` and `test` folders directly in the `data/imagenet100/` directory.
  *  After unzipping the downloaded CIFAR100 dataset, place the `train` and `test` folders directly in the `data/cifar-100-python/` directory.
* A typical command to train on CIFAR100 would be:

  python main_memo.py --dataset cifar100 --convnet_type memo_resnet32 --init_cls 50 --increment 10 --memory_size 3312 --batch_size 128 --init_epoch 200 --epochs 200



* A typical command to train on imagenet100 would be:

  python main_memo.py --dataset imagenet100 --convnet_type memo_resnet34_imagenet --init_cls 50 --increment 10 --memory_size 3312 --batch_size 128 --init_epoch 200 --epochs 200
## ✍️ Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{zhao2025scalable,
  title={Scalable Class-Incremental Learning Based on Parametric Neural Collapse},
  author={Zhao, Enhui and Lin, Guangfeng and Zhang, Chuangxin and Liao, Kaiyang and Chen, Yajun},
  journal={},
  year={2025},
  month={}
}
>>>>>>> origin/main

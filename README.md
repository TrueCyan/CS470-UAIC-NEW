# Uncertainty-Aware Image Captioning (UAIC)

PyTorch implementation of the paper "Uncertainty-Aware Image Captioning (UAIC)".

## Data Requirement
Download data and put it in right path.
- Karpathy Splits [https://www.kaggle.com/datasets/shtvkumar/karpathy-splits]
  - data/karpathy_split/dataset_coco.json
- MSCOCO 2014 Dataset [https://cocodataset.org/#download]
  - data/mscoco/train2014
  - data/mscoco/val2014

## Execution Order
1.  src/utils/vocabulary.py
2.  src/train.py
3.  src/evaluate.py

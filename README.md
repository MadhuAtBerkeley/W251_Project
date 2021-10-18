# W251_Project
Final Project for W251 MIDS Program

## 1. Download vggFace2 Dataset

Download dataset with 500 faces from (https://www.kaggle.com/greatgamedota/vggface2-test). The dataset is stored in W251_Project/data/test_no_mask folder

## 2. Dataset Augmentation with masks

Install packages in conda environment given in W251_project/conda_env.sh
```
python mask_the_face.py --inpath data/test_no_mask --outpath data/test_with_mask --mask_type <type-of-mask> --verbose --write_original_image
```

## 3. Facenet training without masks

Run jupyter notebook
```
jupyter notebook --allow-root
```

Run finetune_no_augmentation.ipynb

## 4. Facenet training with masks

Run finetune_with_augmentation.ipynb

## 5. Masked face Inference on Jetson

Run PyTorch docker container for inference
```
docker/run.sh
```
Run jetson-inference/infer.ipynb inisde the container
```
jupyter notebook --allow-root
```


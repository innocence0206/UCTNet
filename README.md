# UCTNet: Uncertainty-Guided CNN-Transformer Hybrid Networks for Medical Image Segmentation
This repo is the official implementation for UCTNet.

## Prerequisites
Install packages
```
cd nnUNet
pip install -e .

cd UCTNet
pip install -e .
```

## Datasets processing
Download datasets [BCV(Synapse)](https://www.dropbox.com/sh/z4hbbzqai0ilqht/AAARqnQhjq3wQcSVFNR__6xNa?dl=0https://www.dropbox.com/sh/z4hbbzqai0ilqht/AAARqnQhjq3wQcSVFNR__6xNa?dl=0), [ACDC](https://acdc.creatis.insa-lyon.fr/description/databases.html) and [ISIC2018](https://challenge.isic-archive.com/data/). 
The dataset partitioning of BCV(Synapse) and ACDC follows TransUNet and the ISIC 2018 is divided according to dataset_ISIC2018.json.

```
UCTNet_BCV -dataset_path DATASET_PATH
UCTNet_ACDC -dataset_path DATASET_PATH
UCTNet_ISIC2018 -dataset_path DATASET_PATH
```
## Training
BCV
```
UCTNet_train -task 17 --fold 0 --custom_network UCTNet_3D -ei UCTNet --deterministic -pretrained_weights "/UCTNet/pretrained_weight/BCV/model_final_checkpoint.model"
```

ACDC
```
UCTNet_train -network 2d -task 27 --fold 0 --custom_network UCTNet_2D -ei UCTNet --deterministic -pretrained_weights "/UCTNet/pretrained_weight/ACDC/model_final_checkpoint.model"
```

ISIC2018
```
UCTNet_train -network 2d -task 100 --fold 0 --custom_network UCTNet_2D -ei UCTNet --deterministic
```

## Testing
BCV
```
UCTNet_predict --task_name 17 --f 0 -ei UCTNet -chk model_final_checkpoint
```

ACDC
```
UCTNet_predict --task_name 27 -m 2d --f 0 -ei UCTNet -chk model_final_checkpoint
```

ISIC_2018
```
UCTNet_predict --task_name 100 -m 2d --f 0 -ei UCTNet -chk model_final_checkpoint
```

## Acknowledgements


Part of codes are reused from the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [PHTrans](https://github.com/lseventeen/PHTrans). Thanks to Wentao Liu and Fabian Isensee for the open source code.

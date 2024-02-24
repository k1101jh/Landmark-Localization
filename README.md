# Landmark Localization

## Dataset
- [**ISBI 2015 Cephalometric**](https://figshare.com/s/37ec464af8e81ae6ebbf) dataset
- [**Digital Hand Atlas**](https://ipilab.usc.edu/computer-aided-bone-age-assessment-of-children-using-a-digital-hand-atlas-2/) dataset

## Pre-work
- Download dataset
- Decompress downloaded files in ./original_data/[dataset_name] folder.
- Run this code
```
### ISBI 2015
python codes/ISBI_2015_cephalometric_preprocess.py

### Digital Hand Atlas
python codes/digital_hand_atlas_preprocess.py
```
경로 수정 시, ./configs/dataset에 있는 yaml 파일에 적힌 경로를 수정할 것


## Train
```
python codes/train.py dataset=[dataset name]
```
dataset name은 configs/dataset의 파일명과 일치해야 한다. (Digital_Hand_Atlas, ISBI_2015_Cephalometric)

## Test
```
python codes/test.py dataset=[dataset name]
```
dataset name은 configs/dataset의 파일명과 일치해야 한다. (Digital_Hand_Atlas, ISBI_2015_Cephalometric)

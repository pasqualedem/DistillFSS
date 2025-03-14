# FSSWeedMap


## Download the dataset

Script:

```bash
wget https://www.phenobench.org/data/PhenoBench-v110.zip
unzip PhenoBench-v110.zip
```

EVICAN
https://paperswithcode.com/sota/cell-segmentation-on-evican

Nucleus
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3332-1

```bash
cd data
kaggle competitions download -c data-science-bowl-2018
unzip data-science-bowl-2018.zip -d data-science-bowl
unzip data-science-bowl/stage1_train.zip -d Nucleus
```

Pothole Mix
https://data.mendeley.com/datasets/kfth5g2xk3/2

Lab2Wild
https://www.kaggle.com/datasets/sergeynesteruk/apple-rotting-segmentation-problem-in-the-wild/code

KVASIR
https://datasets.simula.no/kvasir/


```bash
cd data
wget https://datasets.simula.no/downloads/kvasir-seg.zip
unzip kvasir-seg.zip
```

LungCancer
```bash
cd data
wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5rr22hgzwr-1.zip
unzip 5rr22hgzwr-1.zip
mv "lungcancer/Lung cancer segmentation dataset with Lung-RADS class/"* lungcancer
rm -r "lungcancer/Lung cancer segmentation dataset with Lung-RADS class/"
```

WeedMap
```bash
mkdir data/Weedmap
cd data/WeedMap
# Download the zip
unzip 0_rotations_processed_003_test.zip
```

ISIC
```bash
mkdir data/ISIC
cd data/ISIC
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip
unzip ISIC2018_Task1-2_Training_Input.zip
unzip ISIC2018_Task1_Training_GroundTruth.zip
```

Industrial
```bash
mkdir data/Industrial
cd data/Industrial
wget https://download.scidb.cn/download?fileId=6396c900bae2f1393c118ada -O data.zip
wget https://download.scidb.cn/download?fileId=6396c900bae2f1393c118ad9 -O data.json
unzip data.zip
mv data/* .
rm -r data
```

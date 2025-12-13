# <img src="figures/favicon.svg" alt="icon" width="40" style="vertical-align: middle;"/> [DistillFSS](https://pasqualedem.github.io/DistillFSS/)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2512.05613)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **DistillFSS: Synthesizing Few-Shot Knowledge into a Lightweight Segmentation Model**  
> Official implementation of DistillFSS

<p align="center">
  <img src="figures/DistillFSS.svg" alt="DistillFSS Framework" width="800"/>
</p>

## 🔥 Highlights

- **🚀 Efficient Inference**: No support images needed at test time—knowledge is distilled directly into the model
- **🎯 Strong Performance**: Competitive or superior results compared to state-of-the-art CD-FSS methods
- **📊 Comprehensive Benchmark**: New evaluation protocol spanning medical imaging, industrial inspection, and agriculture
- **⚡ Scalable**: Handles large support sets without computational explosion

## 📋 Abstract

Cross-Domain Few-Shot Semantic Segmentation (CD-FSS) seeks to segment unknown classes in unseen domains using only a few annotated examples. This setting is inherently challenging: source and target domains exhibit substantial distribution shifts, label spaces are disjoint, and support images are scarce—making standard episodic methods unreliable and computationally demanding at test time.

**DistillFSS** addresses these constraints through a teacher-student distillation process that embeds support-set knowledge directly into the model's parameters. By internalizing few-shot reasoning into a dedicated layer, our approach eliminates the need for support images during inference, enabling fast, lightweight deployment while maintaining the ability to adapt to novel classes through rapid specialization.

## 🏗️ Framework Overview

DistillFSS consists of two main components:

1. **Teacher Network**: Processes the support set and encodes class-specific knowledge
2. **Student Network**: Learns to segment without direct access to support images by distilling knowledge from the teacher

The distillation process embeds support-set information into the student's parameters, allowing efficient inference without episodic sampling.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/pasqualedem/DistillFSS.git
cd DistillFSS

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## 📊 Dataset Preparation

Our benchmark includes datasets from diverse domains. Follow the instructions below to download and prepare each dataset:

### 🌱 Agriculture Domain

#### WeedMap
```bash
mkdir -p data/WeedMap
cd data/WeedMap
# Download the zip from the official source
unzip 0_rotations_processed_003_test.zip
```

### 🏥 Medical Imaging Domain

#### Nucleus Dataset
```bash
cd data
kaggle competitions download -c data-science-bowl-2018
unzip data-science-bowl-2018.zip -d data-science-bowl
unzip data-science-bowl/stage1_train.zip -d Nucleus
```

#### KVASIR (Gastrointestinal)
```bash
cd data
wget https://datasets.simula.no/downloads/kvasir-seg.zip
unzip kvasir-seg.zip
```

#### Lung Cancer
```bash
cd data
wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5rr22hgzwr-1.zip
unzip 5rr22hgzwr-1.zip
mv "lungcancer/Lung cancer segmentation dataset with Lung-RADS class/"* lungcancer
rm -r "lungcancer/Lung cancer segmentation dataset with Lung-RADS class/"
```

#### ISIC (Skin Lesions)
```bash
mkdir -p data/ISIC
cd data/ISIC
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip
unzip ISIC2018_Task1-2_Training_Input.zip
unzip ISIC2018_Task1_Training_GroundTruth.zip
```

### 🏭 Industrial & Infrastructure Domain

#### Pothole Mix
Download from [Mendeley Data](https://data.mendeley.com/datasets/kfth5g2xk3/2)

#### Industrial Defects
```bash
mkdir -p data/Industrial
cd data/Industrial
wget https://download.scidb.cn/download?fileId=6396c900bae2f1393c118ada -O data.zip
wget https://download.scidb.cn/download?fileId=6396c900bae2f1393c118ad9 -O data.json
unzip data.zip
mv data/* .
rm -r data
```

## 🚀 Getting Started

DistillFSS provides three main entry points for running grid search experiments:

### 1. Distillation (`distill.py`)

Train a student model by distilling knowledge from a teacher network that processes support examples.

```bash
python distill.py grid --parameters parameters/distill/DATASET_NAME.yaml
```

The distillation process:
- Creates a teacher-student architecture
- Trains the student to mimic the teacher's outputs
- Embeds support-set knowledge into the student's parameters
- Evaluates on the test set after distillation

### 2. Refinement (`refine.py`)

Fine-tune a pre-trained model on support examples for improved performance.

```bash
# Sequential execution
python refine.py grid --parameters parameters/refine/DATASET_NAME.yaml

# Parallel execution (creates SLURM scripts)
python refine.py grid --parameters parameters/refine/DATASET_NAME.yaml --parallel

# Only create SLURM scripts without running
python refine.py grid --parameters parameters/refine/DATASET_NAME.yaml --parallel --only_create
```

### 3. Speed Benchmarking

Evaluate the inference speed and efficiency of different models.

```bash
python distill.py grid --parameters parameters/speed.yaml
```

### Configuration Files

The repository includes pre-configured parameter files organized by experiment type:

#### 📊 Baseline Configurations (`parameters/baselines/`)
Standard baseline experiments for each dataset:
- `Industrial.yaml` - Industrial defect segmentation
- `ISIC.yaml` - Skin lesion segmentation
- `KVASIR.yaml` - Gastrointestinal polyp segmentation
- `LungCancer.yaml` - Lung nodule segmentation
- `Nucleus.yaml` / `Nucleus_hdmnet.yaml` - Cell nucleus segmentation
- `Pothole.yaml` - Road defect detection
- `WeedMap.yaml` - Weed segmentation

#### 🎓 Distillation Configurations (`parameters/distill/`)
Teacher-student distillation experiments:
- Configurations for: Industrial, ISIC, KVASIR, LungCancer, Nucleus, Pothole, WeedMap

#### 🔧 Refinement Configurations (`parameters/refine/`)
Fine-tuning experiments on support sets:
- Configurations for: Industrial, ISIC, KVASIR, LungCancer, Nucleus, Pothole, WeedMap, deepglobe

#### ⚡ Speed Benchmark Configuration (`parameters/speed.yaml`)
Benchmarking inference speed across models and datasets.

### Example Usage

```bash
# Run baseline experiments on Industrial dataset
python refine.py grid --parameters parameters/baselines/Industrial.yaml

# Run distillation on KVASIR dataset
python distill.py grid --parameters parameters/distill/KVASIR.yaml

# Run refinement on WeedMap with parallel execution
python refine.py grid --parameters parameters/refine/WeedMap.yaml --parallel

# Run efficiency benchmarks
python distill.py grid --parameters parameters/speed.yaml

# Run experiments on additional datasets
python refine.py grid --parameters parameters/other/EVICAN.yaml
```

## 📈 Results

DistillFSS achieves competitive or superior performance across multiple domains while significantly reducing computational costs:

### Experimental Results

Performance comparison (mIoU) against state-of-the-art methods on Medical and Industrial datasets.

<table>
<thead>
  <tr>
    <th rowspan="2">Dataset (Shot k)</th>
    <th colspan="3" align="center">Low Shot<br><i>(k=5, 9, or 10)</i></th>
    <th colspan="3" align="center">High Shot<br><i>(k=50, 60, or 80)</i></th>
  </tr>
  <tr>
    <th align="center">BAM</th>
    <th align="center">Transfer</th>
    <th align="center">Distill</th>
    <th align="center">BAM</th>
    <th align="center">Transfer</th>
    <th align="center">Distill</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><b>Lung Nodule</b> (5/50)</td>
    <td align="center">0.17</td>
    <td align="center">3.43</td>
    <td align="center">3.31</td>
    <td align="center">0.19</td>
    <td align="center">2.51</td>
    <td align="center"><b>4.87</b></td>
  </tr>
  <tr>
    <td><b>ISIC</b> (9/60)</td>
    <td align="center">9.67</td>
    <td align="center"><b>14.35</b></td>
    <td align="center">13.31</td>
    <td align="center">8.69</td>
    <td align="center">22.15</td>
    <td align="center"><b>23.41</b></td>
  </tr>
  <tr>
    <td><b>KVASIR-Seg</b> (5/50)</td>
    <td align="center">18.96</td>
    <td align="center"><b>45.18</b></td>
    <td align="center">37.29</td>
    <td align="center">23.03</td>
    <td align="center"><b>59.97</b></td>
    <td align="center">57.09</td>
  </tr>
  <tr>
    <td><b>Nucleus</b> (5/50)</td>
    <td align="center">11.03</td>
    <td align="center"><b>73.12</b></td>
    <td align="center">69.57</td>
    <td align="center">11.05</td>
    <td align="center">79.39</td>
    <td align="center"><b>79.96</b></td>
  </tr>
  <tr>
    <td><b>WeedMap</b> (5/50)</td>
    <td align="center">6.63</td>
    <td align="center"><b>51.01</b></td>
    <td align="center">44.43</td>
    <td align="center">6.16</td>
    <td align="center"><b>64.18</b></td>
    <td align="center">61.96</td>
  </tr>
  <tr>
    <td><b>Pothole</b> (5/50)</td>
    <td align="center">1.46</td>
    <td align="center"><b>17.36</b></td>
    <td align="center">17.01</td>
    <td align="center">2.23</td>
    <td align="center">31.77</td>
    <td align="center"><b>31.96</b></td>
  </tr>
  <tr>
    <td><b>Industrial</b> (10/80)</td>
    <td align="center">4.98</td>
    <td align="center">4.09</td>
    <td align="center">3.50</td>
    <td align="center">4.86</td>
    <td align="center"><b>48.19</b></td>
    <td align="center">46.09</td>
  </tr>
</tbody>
</table>

*Detailed results and ablation studies are available in the paper.*

## 🔧 Project Structure

```
DistillFSS/
├── distill.py              # Main distillation entry point
├── refine.py               # Main refinement entry point
├── configs/                # Configuration files
├── distillfss/
│   ├── data/              # Dataset implementations
│   ├── models/            # Model architectures
│   ├── utils/             # Utilities (logging, tracking, etc.)
│   └── substitution.py    # Support set substitution strategies
├── data/                  # Dataset storage
└── out/                   # Output directory (logs, models, results)
```

## 📊 Experiment Tracking

DistillFSS integrates with [Weights & Biases](https://wandb.ai) for experiment tracking. Configure your W&B credentials before running:

```bash
wandb login
```

Training metrics, predictions, and model checkpoints are automatically logged to W&B.

## 📚 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@misc{marinisDistillFSSSynthesizingFewShot2025,
	title = {{DistillFSS}: {Synthesizing} {Few}-{Shot} {Knowledge} into a {Lightweight} {Segmentation} {Model}},
	shorttitle = {{DistillFSS}},
	url = {http://arxiv.org/abs/2512.05613},
	doi = {10.48550/arXiv.2512.05613},
	publisher = {arXiv},
	author = {Marinis, Pasquale De and Blok, Pieter M. and Kaymak, Uzay and Brussee, Rogier and Vessio, Gennaro and Castellano, Giovanna},
	month = dec,
	year = {2025},
	note = {arXiv:2512.05613 [cs]},
```

## 🙏 Acknowledgements

This work builds upon several excellent open-source projects and datasets. We thank the authors for making their code and data publicly available.

## 📝 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## 📧 Contact

For questions or collaborations, please contact:
- **Pasquale De Marinis** - [pasquale.demarinis@uniba.it](mailto:pasquale.demarinis@uniba.it)
- **GitHub Issues** - For bug reports and feature requests

---

<p align="center">
  Made with ❤️ for the Few-Shot Learning community
</p>

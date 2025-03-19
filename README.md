# Introduction

This repository contains the code implementation for the paper **"Chromosomal Mutation-Inspired Radio Augmentation for Enhanced Automatic Modulation Classification"**, published in the **IEEE Internet of Things Journal**. Inspired by chromosomal mutations, the paper proposes six novel I/Q modulation signal augmentation methods: **Interstitial Deletion**, **Terminal Deletion**, **Inversion**, **Breakage**, **Ring**, and **Translocation**, aimed at enhancing the performance of automatic modulation classification (AMC). This repository provides the implementation of these six data augmentation methods. If you use this work, please cite the following paper:

```bibtex
@article{pu2024chromosomal,
  title={Chromosomal Mutation-Inspired Radio Augmentation for Enhanced Automatic Modulation Classification},
  author={Pu, Xitong and Luo, Chunbo and Yin, Yihao and Liu, Zijian and Luo, Yang},
  journal={IEEE Internet of Things Journal},
  volume={11},
  number={24},
  pages={41124--41136},
  year={2024},
  publisher={IEEE}
}
```

## Environment
The code requires the following dependencies:

• **PyTorch** ≥ 1.10.0

• **Numpy** ≥ 1.20.0

• **Python** ≥ 3.9.0

## Directory Structure
```
- checkpoints/       # Pre-trained model files
- datasets/          # Dataset files
- models/            # Model scripts
- dataset2016a.py    # Script for loading RML2016.10A dataset
- dataset2016b.py    # Script for loading RML2016.10B dataset
- main_cldnn.py      # Main script for running CLDNN2
- radioaug.py        # Implementation of radio augmentation methods
```

### Notes:
• The implementations of the five augmentation methods (**Interstitial Deletion**, **Terminal Deletion**, **Inversion**, **Breakage**, and **Ring**) are centralized in the `radioaug.py` script.

• The **Translocation** method is embedded within the dataset loading scripts (`dataset2016a.py` and `dataset2016b.py`). Users can refer to the comments in these scripts for detailed usage instructions.

## AMC Performance Validation
Empowered by these six augmentation methods, we achieved **state-of-the-art (SOTA)** performance on the **RML2016.10A** and **RML2016.10B** datasets using the **CLDNN2** model, with accuracies of **67.11%** and **69.02%**, respectively. This repository provides the pre-trained models for both datasets. Follow the steps below to verify the results:


1. **Download the Repository and Datasets**:

   • Clone this repository to your local machine.

   • Download the **RML2016.10A** and **RML2016.10B** datasets from the [DEEPSIG](https://www.deepsig.ai/datasets/) website.

   • Place the downloaded datasets into the `<datasets>` directory.

2. **Run the Script**:

   • Execute the following command in the terminal to verify the SOTA results for the **"All"** scenario on the **RML2016.10A** dataset (as shown in Table 1(a) of the paper):
   ```bash
   python main_cldnn.py --dataset a --checkpoint ./checkpoints/CLDNN_2016a_all.pth
   ```

   • Execute the following command in the terminal to verify the SOTA results for the **"All"** scenario on the **RML2016.10B** dataset (as shown in Table 1(b) of the paper):
   ```bash
   python main_cldnn.py --dataset b --checkpoint ./checkpoints/CLDNN_2016b_all.pth
   ```

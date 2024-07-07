## Instructions：
This repository provides the implementation code for the six radio augmentation methods proposed in the article, as well as the pre-trained models that achieve state-of-the-art (SOTA) performance. These models are used to verify the authenticity of the experimental results presented in Table 1 of the article. For privacy reasons, the complete code and pre-trained models will be fully available once our article is accepted for publication.

## Environment
- PyTorch ≥ 1.10.0
- Numpy ≥ 1.20.0
- Python ≥ 3.9.0

## Verification of Experimental Results
Before running the scripts, you need to download this repository to your local host, and download the RML2016.10A and RML2016.10B datasets from the [DEEPSIG网站](https://www.deepsig.ai/datasets/) website. Move these two datasets to the <datasets> directory.

### Verification of Experimental Results on CLDNN
Run the following command in the terminal:
```
python main_cldnn.py --dataset a --checkpoint ./checkpoints/CLDNN_2016a_all.pth
```
This verifies the SOTA results of CLDNN in the "All" scenario on RML2016.10A as shown in Table 1(a) of the article.

Run the following command in the terminal:
```
python main_cldnn.py --dataset b --checkpoint ./checkpoints/CLDNN_2016b_all.pth
```
This verifies the SOTA results of CLDNN in the "All" scenario on RML2016.10B as shown in Table 1(b) of the article.

## Verification of Experimental Results on TRN
Run the following command in the terminal:
```
python main_trn.py --dataset a --checkpoint ./checkpoints/TRN_2016a_all.pth
```
This verifies the results of TRN in the "All" scenario on RML2016.10A as shown in Table 1(a) of the article.

Run the following command in the terminal:
```
python main_trn.py --dataset b --checkpoint ./checkpoints/TRN_2016b_all.pth
```
This verifies the results of TRN in the "All" scenario on RML2016.10B as shown in Table 1(b) of the article.


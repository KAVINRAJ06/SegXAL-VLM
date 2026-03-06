<div align="center">
  <div>
  <h1>SegAL-VLM: Explainable Active Learning with Vision–Language Models for Data-Efficient Remote Sensing Segmentation</h1>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=kavinraj.SegAL-VLM&left_color=green&right_color=red)

  </div>
</div>

<div align="center">

<div>
    <a href='https://www.srmist.edu.in/faculty/dr-athira-m-nambiar/' target='_blank'>Dr. Athira Nambiar</a><sup>1</sup>&emsp;
    <a href='https://github.com/KAVINRAJ06' target='_blank'>Kavinraj M</a><sup>1</sup>&emsp;
    <a href='https://github.com/SamrithaaPrabakar' target='_blank'>Samrithaa P</a><sup>1</sup>
</div>

<div>
<sup>1</sup>SRM Institute of Science and Technology, Kattankulathur, India
</div>

</div>

<br>

This repository contains the implementation of **SegAL-VLM**, a **Explainable Active Learning with Vision–Language Models for
Data-Efficient Remote Sensing Segmentation**

The model integrates **Vision Transformers and CLIP-based language guidance** to improve segmentation performance while reducing labeling cost through **uncertainty-aware active learning**.

---

## Abstract
Semantic segmentation of Earth observation imagery is a critical real-world problem, that entails fine-grained pixel-level understanding of urban structures, land-cover regions, and natural environments. Despite recent advances in deep learning based segmentation, traditional pipelines rely on large annotated datasets and uncertainty centric active learning strategies that fail to capture class-specific semantic structure and interpretability during learning. To address this limitation, we propose **SegXAL-VLM**, a vision-language model-driven, annotation-efficient Explainable Active Learning framework for remote sensing semantic segmentation. In particular, our approach combines vision language semantic prompt-conditioning with cross modal attention alignment to ensure interpretable and reliable dense scene understanding. Semantic relevance and prediction uncertainty, along with region-level ambiguities, are jointly modeled using a Multimodal Explainability Module for informed sample selection under limited annotation budgets. Interpretable decisions are facilitated through text-guided attention and entropy based uncertainty estimation, leveraging semantic alignment and contextual cues, while visual explainability maps highlight informative regions. Extensive evaluations on the LoveDA dataset demonstrate that SegXAL-VLM achieves competitive segmentation performance (65.98% against fully supervised 67.4% mIoU) with substantially fewer labeled samples (40% of data) and outperforms conventional uncertainty based active learning approaches.


## Our Network

![Network Architecture](https://raw.githubusercontent.com/KAVINRAJ06/SegXAL-VLM/main/assets/SegXAL_VLM.png)

## Dataset

We evaluate our method on the **LoveDA dataset**, a high-resolution remote sensing benchmark for land-cover segmentation.

The dataset contains **5,987 aerial images** collected from **three Chinese cities (Nanjing, Changzhou, and Wuhan)** with **0.3m spatial resolution**.

Download dataset:  
https://github.com/Junjue-Wang/LoveDA


---

## Class-wise IoU and mIoU on LoveDA

The table below shows **class-wise IoU (%) and overall mIoU** under different **prompt-conditioning strategies**.

- **Bold** indicates the best result in each column  
- _italic_ indicates the IoU for the class targeted by the prompt

<div align="center">

| Method | Background | Building | Road | Water | Barren | Forest | Agriculture | mIoU |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Fully Supervised (100%)** | **74.21** | **64.81** | **71.01** | **63.27** | **48.69** | **70.80** | **79.04** | **67.40** |
| SegXAL-VLM (All Classes) | 72.79 | 63.39 | 69.59 | 61.85 | 47.27 | 69.38 | 77.62 | 65.98 |
| Focus on Background | _73.58_ | 60.44 | 70.01 | 60.60 | 45.93 | 67.36 | 78.33 | 65.18 |
| Focus on Building | 69.47 | _64.76_ | 69.18 | 59.60 | 45.38 | 67.66 | 79.04 | 65.01 |
| Focus on Road | 71.94 | 62.63 | _71.72_ | 61.68 | 45.67 | 68.67 | 78.21 | 65.79 |
| Focus on Water | 70.92 | 62.86 | 69.14 | _63.29_ | 45.64 | 67.93 | 78.89 | 65.52 |
| Focus on Barren | 70.22 | 63.51 | 69.38 | 60.60 | _46.92_ | 67.42 | 76.69 | 64.96 |
| Focus on Forest | 73.21 | 59.03 | 70.41 | 59.05 | 44.27 | _71.03_ | 75.69 | 64.67 |
| Focus on Agriculture | 72.30 | 62.43 | 68.62 | 60.86 | 45.04 | 65.98 | _78.65_ | 64.84 |

</div>


## Comparison with State-of-the-Art Methods

Comparison of **mIoU (%) on the LoveDA dataset**.

<div align="center">

| Method | mIoU (%) |
|:--:|:--:|
| **Fully Supervised (100%)** | **67.40** |
| Text2Seg (Zero-shot) | 31.10 |
| MetaSegNet (Text-based) | 52.00 |
| MRF-STI | 63.15 |
| **SegXAL-VLM (Ours)** | **65.98** |

</div>

## Software Requirements

- Python 3.9+, PyTorch, timm, transformers, albumentations, numpy, opencv-python, matplotlib, tqdm
- To download the dependencies: **!pip install -r requirements.txt**


## For further details, contact
athiram@srmist.edu.in, km4531@srmist.edu.in, sp8573@srmist.edu.in

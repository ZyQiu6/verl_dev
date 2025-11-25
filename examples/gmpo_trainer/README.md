<div align=center>
  
# Geometric-Mean Policy Optimization
</div>

This is the official implementaion of paper [***Geometric-Mean Policy Optimization***](https://arxiv.org/abs/2507.20673).

<div align=center>
<<<<<<< HEAD
<img width="3092" height="864" alt="image" src="https://github.com/user-attachments/assets/af4c7e0f-923a-45ef-9bcf-57109b8ee61e" />
</div>


=======
<img width="3092" height="864" alt="image" src="https://github.com/user-attachments/assets/20b04c4e-7ee8-4775-9af8-33c0158336e2" />
</div>

>>>>>>> origin/npu
## 1. Contents
- Geometric-Mean Policy Optimization
  - [1. Contents](#1-contents)
  - [2. Introduction](#2-introduction)
<<<<<<< HEAD
  - [3. Code Usage](#4-code-usage)
  - [4. Contacts](#5-contacts)
  - [5. Citation](#7-citation)

## 2. Introduction

Recent advancements, such as Group Relative Policy Optimization (GRPO), have enhanced the reasoning capabilities of large language models by optimizing the arithmetic mean of token-level rewards. However, GRPO suffers from unstable policy updates when processing tokens with outlier importance-weighted rewards, which manifests as extreme importance sampling ratios during training, i.e., the ratio between the sampling probabilities assigned to a token by the current and old policies. In this work, we propose Geometric-Mean Policy Optimization (GMPO), a stabilized variant of GRPO. Instead of optimizing the arithmetic mean, GMPO maximizes the geometric mean of token-level rewards, which is inherently less sensitive to outliers and maintains a more stable range of importance sampling ratio. In addition, we provide comprehensive theoretical and experimental analysis to justify the design and stability benefits of GMPO. Beyond improved stability, GMPO-7B outperforms GRPO by an average of 4.1% on multiple mathematical benchmarks and 1.4% on multimodal reasoning benchmark, including AIME24, AMC, MATH500, OlympiadBench, Minerva, and Geometry3K.
=======
  - [3. Code Usage](#3-code-usage)
  - [4. Contacts](#4-contacts)
  - [5. Citation](#5-citation)

## 2. Introduction

Group Relative Policy Optimization (GRPO) has significantly enhanced the reasoning capability of large language models by optimizing the arithmetic mean of token-level rewards. Unfortunately, GRPO is observed to suffer from unstable policy updates when facing tokens with outlier importance-weighted rewards, which manifest as extreme importance sampling ratios during training. In this study, we propose Geometric-Mean Policy Optimization (GMPO), with the aim to improve the stability of GRPO through suppressing token reward outliers. Instead of optimizing the arithmetic mean, GMPO maximizes the geometric mean of token-level rewards, which is inherently less sensitive to outliers and maintains a more stable range of importance sampling ratio. GMPO is plug-and-play—simply replacing GRPO's arithmetic mean with the geometric mean of token-level rewards, as the latter is inherently less sensitive to outliers. GMPO is theoretically plausible—analysis reveals that both GMPO and GRPO are weighted forms of the policy gradient while the former enjoys more stable weights, which consequently benefits policy optimization and performance. Experiments on multiple mathematical reasoning benchmarks show that GMPO-7B improves the average Pass@1 of GRPO by up to 4.1%, outperforming many state-of-the-art approaches.
>>>>>>> origin/npu

## 3. Code Usage

The key configurations are:
```
clip_ratio_low=0.4
clip_ratio_high=0.4
loss_mode=geo_mean
```
<<<<<<< HEAD

=======
We observed that using a large clip ratio during Mixture-of-Experts (MoE) model training often leads to optimization instability. When training MoE models, consider lowering the clip ratio to achieve more stable convergence.
>>>>>>> origin/npu
To get started quickly, run:
```
bash examples/gmpo_trainer/run_qwen2_5-7b_math.sh
```

GMPO can be combined with other methods such as DAPO (experimental - not fully tested):
```
bash examples/gmpo_trainer/test_dapo_7b_math.sh 
bash examples/gmpo_trainer/test_dapo_qwen3_30b_math.sh
```

## 4. Contacts
If you have any question about our work or this repository, please don't hesitate to contact us by emails or open an issue under this project.
- [zhaoyuzhong20@mails.ucas.ac.cn](zhaoyuzhong20@mails.ucas.ac.cn)
- [liuyue171@mails.ucas.ac.cn](liuyue171@mails.ucas.ac.cn)
- [lecu@microsoft.com](lecu@microsoft.com)
- [wanfang@ucas.ac.cn](wanfang@ucas.ac.cn)

## 5. Citation
```
<<<<<<< HEAD
@misc{zhao2025geometricmeanpolicyoptimization,
      title={Geometric-Mean Policy Optimization}, 
      author={Yuzhong Zhao and Yue Liu and Junpeng Liu and Jingye Chen and Xun Wu and Yaru Hao and Tengchao Lv and Shaohan Huang and Lei Cui and Qixiang Ye and Fang Wan and Furu Wei},
      year={2025},
      eprint={2507.20673},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.20673}, 
=======
@article{zhao2025geometric,
  title={Geometric-mean policy optimization},
  author={Zhao, Yuzhong and Liu, Yue and Liu, Junpeng and Chen, Jingye and Wu, Xun and Hao, Yaru and Lv, Tengchao and Huang, Shaohan and Cui, Lei and Ye, Qixiang and others},
  journal={arXiv preprint arXiv:2507.20673},
  year={2025}
>>>>>>> origin/npu
}
```

# PIM-GAN: Kinetics-Constrained Generative Framework for Microstructure Evolution

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Traceable-success.svg)](docs/release_checklist.md)

本仓库包含论文 **《Learning statistically representative microstructure evolution from ex-situ micrographs using a kinetics-constrained generative framework》** 的完整官方实现代码、数据预处理流水线、训练配置文件及图表复现脚本。

---

## 目录
- [1. 框架概述与核心方法](#1-框架概述与核心方法)
- [2. 代码仓库结构](#2-代码仓库结构)
- [3. 环境依赖与安装](#3-环境依赖与安装)
- [4. 数据集组织与划分规范](#4-数据集组织与划分规范)
- [5. 快速上手：完整复现流程](#5-快速上手完整复现流程)
- [6. 模型架构与物理损失细节](#6-模型架构与物理损失细节)
- [7. 实验评估与基线对比](#7-实验评估与基线对比)
- [8. 领域迁移测试 (GH4738 & A356)](#8-领域迁移测试-gh4738--a356)
- [9. 引用与致谢](#9-引用与致谢)

---

## 1. 框架概述与核心方法

常规的原位（in-situ）微观组织演化表征受限于成像视场与极端温场环境；而常规扫描电镜（ex-situ SEM）由于**缺乏初-终态视场的空间配准（Spatial Registration）**，无法进行确定性的像素级演化追踪。

**PIM-GAN (Process-Informed Microstructure GAN)** 将微观组织演化学习从传统的“空间轨迹预测”重塑为**“统计代表性演化建模”**：
- **生成器 (Generator)**：基于 6 阶段条件 U-Net。在未配准设定下，跳跃连接（Skip Connections）作为先验机制传递初始晶相的介观拓扑与分布特征，而非像素坐标强行复制；瓶颈层引入**工艺先验注意力模块 (PIAM)**，结合 Arrhenius 扩散项与 Larson-Miller 累积热暴露参量自适应调制相边界与曲率敏感特征。
- **判别器 (Discriminator)**：混合 CNN-Transformer PatchGAN 架构。浅层 CNN 提取局部相边缘，Transformer 自注意力建模长程拓扑连续性，最终输出 $30\times 30$ 矩阵的无 Sigmoid 概率对数（Raw Logits）。
- **复合物理损失**：摒弃导致人工模糊的像素级 $L_1$ 损失，采用 VGG-19 Gram 纹理损失；同时结合可微连续高斯拉普拉斯（LoG）晶粒计数、JMAK 溶解损失与 LSW 粗化损失，通过基于 CALPHAD 标称平衡相分数的自适应软切换函数 $\omega(A_\delta)$ 进行全阶段平滑正则化。

---

## 2. 代码仓库结构

```text
PIM-GAN-reproducibility/
├── configs/                            # 实验超参数与物理参数配置 (YAML)
│   ├── 01_data_protocol.yaml           # 数据集划分、原始尺寸、切片分辨率规范
│   ├── 02_model_architecture.yaml      # U-Net、PIAM、Transformer-D 结构定义
│   ├── 03_losses.yaml                  # 损失权重系数 (lambda_adv, lambda_tex, lambda_phys)
│   ├── 04_training.yaml                # 学习率、优化器与训练轮数 (100 Epochs)
│   ├── 05_evaluation.yaml              # 统计评估指标与视场采样数量
│   ├── 06_transfer_GH4738.yaml         # GH4738 纳米级 gamma' 相零样本外推配置
│   ├── 07_transfer_A356.yaml           # A356 微米级共晶 Si 微调配置
│   └── 08_sensitivity_sweeps.yaml      # 分辨率、裁剪尺寸与数据量敏感性参数
├── data/
│   ├── raw/                            # 原始 SEM 实验图像存放路径
│   │   ├── in718_initial/              # 6 个批次的初始态母图 (对应 GHdata_chu)
│   │   ├── in718_final/                # 306 个工艺态最终母图 (对应 GHdata)
│   │   ├── external_885/               # 885 °C 盲测外推实验母图 (30-240 min)
│   │   ├── transfer_GH4738/            # GH4738 迁移验证图像
│   │   └── transfer_A356/              # A356 迁移验证图像
│   ├── templates/                      # 元数据、切片清单与记录表模板
│   │   ├── parent_pairs.csv            # 306 组母图对关联与划分清单
│   │   ├── heat_treatment_schedule.csv # 51 种热处理完整工艺参数表
│   │   └── seed_run_metrics.csv        # 5 次独立随机种子运行原始记录
│   └── examples/                       # 文档与表格提取的基础统计参考数据
├── src/pim_gan/                        # 核心算法 Python 包
│   ├── data/                           # 切片切取、数据增强、Otsu 阈值与数据集类
│   │   ├── dataset.py                  # 双通道配准无关配对 Dataset
│   │   ├── crop_tiles.py               # 425x425 原图规则提取与 512x512 重采样
│   │   ├── augment.py                  # 水平翻转与微角度旋转双倍增强
│   │   └── phase_masks.py              # Otsu 阈值分割与训练集固定 T_ref 计算
│   ├── models/                         # 深度神经网络架构
│   │   ├── generator.py                # 6 阶段 U-Net 生成器顶层封装
│   │   ├── encoder.py / decoder.py     # 卷积下采样与带 Skip 的转置卷积上采样
│   │   ├── piam.py                     # Process-Informed Attention Module
│   │   ├── process_mlp.py              # X_phys 嵌入与可学习活化能 Q、JMAK (n, k) 预测
│   │   ├── discriminator.py            # CNN-Stem + Transformer PatchGAN (30x30)
│   │   └── transformer_blocks.py       # 自注意力与多层感知机块
│   ├── physics/                        # 物理动力学先验与可微算子
│   │   ├── soft_descriptors.py         # 连续相分数与可微 LoG 粒子计数
│   │   ├── jmak.py / lsw.py            # JMAK 溶解与 LSW 粗化动力学残差
│   │   ├── kinetic_switch.py           # CALPHAD 平衡相比例软切换机制 omega(A_delta)
│   │   └── process_priors.py           # Arrhenius 与 Larson-Miller 解析方程计算
│   ├── losses/                         # 复合损失函数
│   │   ├── total.py                    # L_total = L_adv + L_tex + L_kinetic 顶层计算
│   │   ├── adversarial.py              # BCEWithLogits 对抗损失
│   │   └── vgg19_gram.py               # VGG-19 多层 Gram 矩阵统计特征匹配
│   ├── training/                       # 训练循环与跨种子实验管理器
│   │   ├── trainer.py                  # 完整的 GAN 交替优化训练循环
│   │   ├── checkpoints.py              # 模型权重持久化与加载
│   │   └── seed.py                     # 严格的随机种子冻结 (Python, NumPy, PyTorch)
│   ├── evaluation/                     # 免配准统计与拓扑指标度量
│   │   ├── area_fraction.py            # 2D delta 相面积百分比计算
│   │   ├── characteristic_width.py     # 2A / P_peri 真实扩散特征宽度 (微米)
│   │   ├── topology.py                 # 欧拉示性数 chi 与拓扑骨架长度 L_skel
│   │   ├── s2_correlation.py           # 两点自相关函数 S2(r) 径向计算与 RMSE
│   │   ├── field_uncertainty.py        # Table S7: 视场级差异 Welch's t-test
│   │   └── run_uncertainty.py          # Fig. S9: 5 次完整训练 95% 置信区间评估
│   └── plots/                          # 论文正文及附录所有图表自动重绘模块
├── scripts/                            # 用户命令行执行入口
│   ├── 00_check_parent_manifest.py     # 检查 parent_pairs.csv 划分严密性
│   ├── 01_validate_manifest.py         # 校验工艺文本与原始图像的存在性
│   ├── 02_preprocess_sem.py            # 批处理裁剪原图、数据增强并计算 T_ref
│   ├── 03_train_pim_gan.py             # 启动主模型训练
│   ├── 06_evaluate_980_blind.py        # 980 °C 盲测时间序列测试 (Level 2)
│   ├── 07_evaluate_885_external.py     # 885 °C 未见外推温度验证 (Level 3)
│   ├── 08_run_ablations.py             # 执行 PIAM、判别器、动力学损失消融实验
│   ├── 10_run_five_seeds.py            # 执行 5 独立种子全流程复现并导出不确定度
│   ├── 11_analyze_885_fields.py        # 复现 Table S7 的视场统计与 t 检验结果
│   ├── 15_transfer_GH4738_A356.py      # 执行跨合金跨尺度迁移测试
│   └── 17_make_figures.py              # 一键生成论文高清矢量图表
├── data.csv                            # 51 种热处理工艺参数表 (用户数据库)
├── pyproject.toml                      # 工程依赖声明文件
└── README.md
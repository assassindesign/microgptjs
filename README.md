# MicroGPT.js

一个极简的、无依赖的纯 JavaScript (Node.js) GPT 实现。

这是对 [Andrej Karpathy](https://github.com/karpathy) “原子级” Python（https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95 ） 实现的直接移植。仅使用 Node.js 标准库 + ES5语法，演示 Transformer 的核心算法——自动求导、注意力机制和优化器，代码中没有复杂的语法糖，你可以理解每一步代码做了什么。

microgpt-tutorial.html（https://htmlpreview.github.io/?https://github.com/assassindesign/microgptjs/blob/main/microgpt-tutorial.html）是整个代码的教学
增加了一组中文古诗词数据，也可以同样训练。

<img width="183" height="212" alt="f0ec2a" src="https://github.com/user-attachments/assets/a4924486-8360-420f-92a5-7960f6aa7149" />


## 🌟 特性

- **纯 JavaScript 实现**：无需安装 TensorFlow、PyTorch 或任何 npm 包。
- **完整的自动求导引擎**：通过 `Value` 类实现反向传播。
- **完整 GPT 架构**：
  - Token 与位置嵌入
  - 多头自注意力机制
  - 层归一化
  - 前馈神经网络 (MLP) 与残差连接
- **Adam 优化器**：完整实现了 Adam 优化算法。
- **模型保存与断点续训**：训练过程中会保存 checkpoint，可以中断后继续训练。

## 🚀 快速开始

### 环境要求

- 已安装 [Node.js](https://nodejs.org/) 。
- 系统包含 `curl` 命令（用于自动下载训练数据）。

### 运行

1. 克隆仓库
2. 运行脚本

```bash
node microgpt.js
```

## 💾 New：模型保存与断点续训

`microgpt.js` 会自动使用 `gpt_checkpoint.json` 保存和读取模型。

如果当前目录没有 `gpt_checkpoint.json`，程序会从第 0 步开始训练，并在训练过程中自动保存 checkpoint。

如果当前目录已经有 `gpt_checkpoint.json`，程序会自动读取它，并从 checkpoint 记录的 `_step` 继续训练。

checkpoint 使用 JSON 格式保存，内容包括：

- 所有模型参数和权重：`wte`、`wpe`、`lm_head`、`attn_wq`、`attn_wk`、`attn_wv`、`attn_wo`、`mlp_fc1`、`mlp_fc2`
- Adam 优化器状态：`_m`、`_v`
- 训练进度：`_step`
- 随机数状态：`_seed`
- 词表和模型配置：`_uchars`、`_BOS`、`_vocabSize`、`_nLayer`、`_nEmb`、`_blockSize`、`_nHead`、`_headDim`

默认每 20 步保存一次 checkpoint，训练结束时也会保存一次。

如果想继续训练更久，只需要把 `microgpt.js` 中的总训练步数调大：

```js
var steps = 2000;
```

然后重新运行：

```bash
node microgpt.js
```

例如 checkpoint 已经训练到 `_step = 1000`，把 `steps` 改成 `2000` 后再次运行，程序会从第 1000 步继续训练到第 2000 步。

如果只想读取已有模型并直接推理，保持 `steps` 小于或等于 checkpoint 里的 `_step` 即可。

注意：`gpt_checkpoint.json` 和训练数据的词表必须匹配。如果更换了 `input.txt`，导致字符表变化，程序会拒绝加载旧 checkpoint，避免把不兼容的权重读进模型。

# MicroGPT.js

一个极简的、无依赖的纯 JavaScript (Node.js) GPT 实现。

这是对 [Andrej Karpathy](https://github.com/karpathy) “原子级” Python（https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95） 实现的直接移植。旨在仅使用 Node.js 标准库 + ES5语法，演示 Transformer 的核心算法——自动求导、注意力机制和优化器。

## 🌟 特性

- **纯 JavaScript 实现**：无需安装 TensorFlow、PyTorch 或任何 npm 包。
- **完整的自动求导引擎**：通过 `Value` 类实现反向传播。
- **完整 GPT 架构**：
  - Token 与位置嵌入
  - 多头自注意力机制
  - 层归一化
  - 前馈神经网络 (MLP) 与残差连接
- **Adam 优化器**：完整实现了 Adam 优化算法。

## 🚀 快速开始

### 环境要求

- 已安装 [Node.js](https://nodejs.org/) 。
- 系统包含 `curl` 命令（用于自动下载训练数据）。

### 运行

1. 克隆仓库
2. 运行脚本

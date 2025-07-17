# Multi-Head Attention Implementation

一个完整的多头注意力机制实现，基于PyTorch构建，包含完整的Transformer模块。

## 🚀 特性

- **完整的多头注意力实现** - 包含Query、Key、Value线性变换和注意力权重计算
- **支持交叉注意力** - 可用于编码器-解码器架构
- **Transformer块** - 包含多头注意力、前馈网络、层归一化和残差连接
- **掩码支持** - 支持填充掩码和前瞻掩码
- **灵活的配置** - 可调整模型维度、注意力头数等参数
- **详细文档和示例** - 包含使用示例和可视化代码

## 📁 文件结构

```
.
├── multi_head_attention.py  # 主要实现文件
├── demo.py                  # 演示脚本
├── requirements.txt         # 依赖包
└── README.md               # 文档
```

## 🛠️ 安装

1. 创建虚拟环境：
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 💡 核心组件

### MultiHeadAttention

多头注意力机制的核心实现。

**参数:**
- `d_model`: 模型维度（嵌入维度）
- `num_heads`: 注意力头数
- `dropout`: Dropout概率（默认0.1）

**主要方法:**
- `forward(query, key, value, mask=None)`: 前向传播

### TransformerBlock

完整的Transformer块，包含多头注意力和前馈网络。

**参数:**
- `d_model`: 模型维度
- `num_heads`: 注意力头数
- `d_ff`: 前馈网络维度（默认为4*d_model）
- `dropout`: Dropout概率（默认0.1）

## 🔧 使用示例

### 基本自注意力

```python
import torch
from multi_head_attention import MultiHeadAttention

# 创建模型
attention = MultiHeadAttention(d_model=512, num_heads=8)

# 输入数据 (batch_size, seq_len, d_model)
x = torch.randn(32, 100, 512)

# 前向传播
output, attention_weights = attention(x, x, x)
print(f"输出形状: {output.shape}")  # torch.Size([32, 100, 512])
```

### 交叉注意力（编码器-解码器）

```python
# 不同序列长度的交叉注意力
encoder_output = torch.randn(32, 50, 512)  # 编码器输出
decoder_input = torch.randn(32, 30, 512)   # 解码器输入

# 解码器查询注意编码器的键值
output, attention_weights = attention(
    query=decoder_input,    # Q来自解码器
    key=encoder_output,     # K来自编码器
    value=encoder_output    # V来自编码器
)
```

### 使用掩码

```python
from multi_head_attention import create_padding_mask, create_look_ahead_mask

# 填充掩码
seq_tokens = torch.randint(1, 1000, (32, 100))
seq_tokens[:, -10:] = 0  # 最后10个token为填充
padding_mask = create_padding_mask(seq_tokens, pad_token=0)

# 前瞻掩码（用于解码器）
look_ahead_mask = create_look_ahead_mask(100)

# 使用掩码的注意力
output, _ = attention(x, x, x, mask=padding_mask)
```

### 完整的Transformer块

```python
from multi_head_attention import TransformerBlock

# 创建Transformer块
transformer_block = TransformerBlock(d_model=512, num_heads=8)

# 前向传播
output = transformer_block(x)
```

## 🎯 运行演示

运行完整的演示脚本：

```bash
python demo.py
```

演示包括：
- 基本多头注意力
- 交叉注意力（编码器-解码器）
- 完整的Transformer块
- 注意力权重可视化

运行测试：

```bash
python multi_head_attention.py
```

## 📊 模型参数量

对于不同配置的参数量：

| d_model | num_heads | MultiHeadAttention | TransformerBlock |
|---------|-----------|-------------------|------------------|
| 256     | 8         | 263,168           | 789,760          |
| 512     | 8         | 1,050,624         | 3,152,384        |
| 768     | 12        | 2,363,136         | 7,088,640        |

## 🧠 技术原理

### 多头注意力机制

1. **线性变换**: 输入通过三个线性层得到Q、K、V
2. **多头分割**: 将Q、K、V分割成多个注意力头
3. **缩放点积注意力**: 计算注意力权重和输出
4. **头合并**: 连接所有头的输出
5. **输出投影**: 通过最终线性层得到输出

数学公式：
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
```

### Transformer块结构

```
输入 → 多头注意力 → 残差连接 → 层归一化 → 
前馈网络 → 残差连接 → 层归一化 → 输出
```

## 🔍 注意力可视化

注意力权重矩阵展示了序列中每个位置对其他位置的关注程度：

```python
# 获取注意力权重
_, attention_weights = attention(x, x, x)

# 形状: (batch_size, num_heads, seq_len, seq_len)
print(f"注意力权重形状: {attention_weights.shape}")

# 可视化第一个头的注意力模式
import matplotlib.pyplot as plt
plt.imshow(attention_weights[0, 0].detach().numpy())
plt.title("注意力权重热力图")
plt.show()
```

## ⚙️ 配置建议

### 常见配置

- **BERT-base**: d_model=768, num_heads=12
- **GPT-2**: d_model=768, num_heads=12
- **T5-base**: d_model=512, num_heads=8
- **轻量级**: d_model=256, num_heads=8

### 注意事项

1. `d_model` 必须能被 `num_heads` 整除
2. 更多的头数可能带来更好的表达能力，但也增加计算成本
3. 使用适当的dropout防止过拟合
4. 考虑序列长度对内存的影响（注意力复杂度为O(n²)）

## 🚀 性能优化

1. **梯度检查点**: 对于长序列使用梯度检查点节省内存
2. **混合精度**: 使用float16加速训练
3. **注意力优化**: 对于超长序列考虑稀疏注意力机制

## 📚 扩展阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化解释
- [PyTorch官方教程](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个实现！

## 📄 许可证

MIT License

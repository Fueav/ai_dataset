# Merlin Chain Function Calling 数据集生成工具

🚀 基于DeepSeek API和MCP工具的智能数据集生成系统，支持生成高质量的function calling训练数据。

## 📋 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用指南](#使用指南)
- [进阶功能](#进阶功能)
- [故障排除](#故障排除)
- [文件说明](#文件说明)

## ✨ 功能特性

### 🧠 智能问题生成 (`generate_question_dataset_smart.py`)
- **智能去重**: 基于问题模式去重，避免重复生成
- **工具平衡**: 确保19个MCP工具的问题分布均衡
- **进度跟踪**: 持久化状态管理，支持断点续传
- **安全退出**: 类似Go defer的机制，Ctrl+C时自动保存数据
- **配置化**: 支持配置文件、命令行、环境变量多种配置方式

### 🔧 完整数据补全 (`generate_complete_dataset.py`)
- **真实Function Calling**: 集成MCP客户端，生成真实的工具调用和响应
- **智能工具推断**: 根据用户问题自动选择合适的MCP工具
- **批量处理**: 支持批量处理，避免API限流
- **详细日志**: 完整的日志记录，便于调试和监控

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    数据集生成流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                           │
│  Step 1: 问题生成阶段                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  generate_question_dataset_smart.py                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ 配置管理器    │  │ 去重管理器    │  │ DeepSeek API│ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  │         │               │               │         │   │
│  │         ▼               ▼               ▼         │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │    智能prompt生成 + 工具分配 + 批量生成            │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          function_calling_dataset_smart.json        │   │
│  │               (问题 + 用户对话)                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│  Step 2: 完整补全阶段                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  generate_complete_dataset.py                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ MCP客户端    │  │ 工具推断器    │  │ DeepSeek API│ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  │         │               │               │         │   │
│  │         ▼               ▼               ▼         │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │ 真实工具调用 + 结果获取 + GPT回复生成              │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │       function_calling_dataset_completed.json       │   │
│  │    (完整的function calling训练数据)                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd ai_dataset

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

复制配置模板并设置API Key：

```bash
# 复制配置模板
cp config.template.json config.json

# 编辑配置文件，设置你的API Key
vim config.json
```

或者设置环境变量：

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key-here"
```

### 3. 生成问题数据集

```bash
# 使用默认配置生成问题
python generate_question_dataset_smart.py

# 或者自定义参数
python generate_question_dataset_smart.py \
  --total_conversations 1000 \
  --batch_size 20
```

### 4. 生成完整数据集

```bash
# 基于问题数据集生成完整的function calling数据
python generate_complete_dataset.py

# 或者指定输入文件
python generate_complete_dataset.py \
  --question_file function_calling_dataset_smart.json \
  --batch_size 5
```

## ⚙️ 配置说明

### 配置文件结构 (`config.json`)

```json
{
  "api": {
    "deepseek_api_key": "your-api-key-here",
    "base_url": "https://api.deepseek.com",
    "timeout": 60
  },
  "generation": {
    "default_total_conversations": 6000,
    "default_batch_size": 50,
    "default_prompt_file": "prompt.txt",
    "default_output_file": "function_calling_dataset_smart.json",
    "system_prompt_file": "prompt-2.txt"
  },
  "completion": {
    "default_question_file": "function_calling_dataset_smart.json",
    "default_output_file": "function_calling_dataset_completed.json",
    "default_batch_size": 1,
    "enable_mcp_connection": true,
    "api_retry_count": 3,
    "batch_delay_seconds": 3
  },
  "system": {
    "enable_debug": false,
    "auto_cleanup_temp_files": true,
    "max_retries": 3
  }
}
```

### 配置项说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `api.deepseek_api_key` | DeepSeek API密钥 | 必填 |
| `generation.default_total_conversations` | 目标生成对话数量 | 6000 |
| `generation.default_batch_size` | 每批生成数量 | 50 |
| `completion.default_batch_size` | 完整化批次大小 | 1 |
| `system.auto_cleanup_temp_files` | 自动清理临时文件 | true |

## 📖 使用指南

### 问题生成阶段

#### 基础用法

```bash
# 默认生成6000个对话，每批50个
python generate_question_dataset_smart.py

# 小规模测试
python generate_question_dataset_smart.py --total_conversations 100 --batch_size 10

# 重置状态重新开始
python generate_question_dataset_smart.py --reset
```

#### 输出文件

- **主文件**: `function_calling_dataset_smart.json` - 最终的问题数据集
- **临时文件**: `temp_smart_question_batch_*_*.json` - 批次临时文件（会自动清理）
- **状态文件**: `generation_state.json` - 进度和去重状态

#### 生成特性

1. **智能工具分配**: 自动为19个MCP工具分配问题数量
2. **去重检查**: 基于问题模式自动去重
3. **断点续传**: 支持中断后继续生成
4. **安全退出**: Ctrl+C时自动保存已生成数据

### 完整数据补全阶段

#### 基础用法

```bash
# 使用默认输入文件
python generate_complete_dataset.py

# 指定输入文件和批次大小
python generate_complete_dataset.py \
  --question_file my_questions.json \
  --batch_size 3
```

#### 输出文件

- **主文件**: `function_calling_dataset_completed.json` - 完整的训练数据
- **临时文件**: `temp_complete_batch_*.json` - 批次临时文件
- **日志文件**: `complete_dataset.log` - 详细执行日志

#### 补全特性

1. **真实工具调用**: 连接MCP服务器执行真实的工具调用
2. **智能工具推断**: 根据问题内容自动选择合适工具
3. **完整对话生成**: 包含function_call、observation、gpt三个阶段

## 🔧 进阶功能

### 多种配置方式

```bash
# 1. 配置文件方式（推荐）
python generate_question_dataset_smart.py

# 2. 环境变量方式
export DEEPSEEK_API_KEY="your-key"
python generate_question_dataset_smart.py

# 3. 命令行方式
python generate_question_dataset_smart.py --api_key "your-key"

# 4. 自定义配置文件
python generate_question_dataset_smart.py --config my_config.json
```

### 批量处理策略

```bash
# 大批量快速生成（问题阶段）
python generate_question_dataset_smart.py \
  --total_conversations 5000 \
  --batch_size 100

# 小批量精细处理（完整化阶段）
python generate_complete_dataset.py \
  --batch_size 1  # 避免API限流
```

### 状态管理

```bash
# 查看当前生成状态
cat generation_state.json

# 重置状态重新开始
python generate_question_dataset_smart.py --reset

# 查看详细日志
tail -f complete_dataset.log
```

## 🗂️ 文件说明

### 核心脚本

| 文件 | 说明 | 作用 |
|------|------|------|
| `generate_question_dataset_smart.py` | 智能问题生成器 | 生成用户问题和初始对话 |
| `generate_complete_dataset.py` | 完整数据补全器 | 补全function calling响应 |
| `config_manager.py` | 配置管理器 | 统一配置文件管理 |

### 工具模块

| 文件 | 说明 |
|------|------|
| `deepseek_api_client.py` | DeepSeek API客户端 |
| `dataset_utils.py` | 数据集处理工具 |
| `dedup_manager.py` | 去重管理器 |
| `merlin_mcp_client.py` | MCP客户端 |

### 配置文件

| 文件 | 说明 |
|------|------|
| `config.json` | 主配置文件（包含API Key） |
| `config.template.json` | 配置模板（供参考） |
| `prompt.txt` | 问题生成提示词 |
| `prompt-2.txt` | 完整化提示词 |

### 状态文件

| 文件 | 说明 |
|------|------|
| `generation_state.json` | 生成状态和去重数据 |
| `complete_dataset.log` | 完整化过程日志 |

## 🛠️ 故障排除

### 常见问题

#### 1. API Key 配置问题

```bash
# 错误：❌ 配置错误: API Key未配置
# 解决：设置API Key
echo '{"api": {"deepseek_api_key": "your-key"}}' > config.json
```

#### 2. 临时文件清理问题

```bash
# 手动清理临时文件
rm temp_smart_question_batch_*.json
rm temp_complete_batch_*.json
```

#### 3. 状态重置

```bash
# 完全重置状态
rm generation_state.json
python generate_question_dataset_smart.py --reset
```

#### 4. MCP连接问题

```bash
# 检查MCP服务是否运行
# 查看日志获取详细错误信息
tail -f complete_dataset.log
```

### 性能优化

#### 1. 批次大小调优

```bash
# 问题生成阶段：大批次提高效率
--batch_size 100

# 完整化阶段：小批次避免限流
--batch_size 1
```

#### 2. 并发控制

```bash
# 避免API限流，增加延迟
# 在config.json中设置：
"batch_delay_seconds": 5
```

## 📊 生成统计

### 工具分布目标

系统会自动为19个MCP工具分配目标数量：

- `get_address_details_by_address`: 600个
- `get_token_info_by_address`: 480个
- `list_address_latest_txs`: 480个
- `get_tx_by_hash`: 420个
- `search_chain_data`: 420个
- 其他工具: 180-300个不等

### 用户角色分布

- 区块链小白: 30%
- 区块链开发者: 30%
- 区块链投资者: 25%
- 区块链专家: 15%

### 语言风格分布

- 口语化: 35%
- 技术用语: 25%
- 中英混合: 25%
- 错误表达: 15%

## 🔒 安全注意事项

1. **API Key保护**: `config.json` 已添加到 `.gitignore`，不会提交到版本控制
2. **环境变量**: 推荐在生产环境使用环境变量设置API Key
3. **配置模板**: 使用 `config.template.json` 作为配置参考

## 📝 更新日志

### v2.0.0 (Current)
- ✅ 重构为配置文件管理
- ✅ 智能去重和工具平衡
- ✅ 安全退出机制
- ✅ 真实MCP工具调用
- ✅ 完整的日志系统

### v1.0.0
- ✅ 基础问题生成功能
- ✅ 简单的function calling补全

## 📞 支持

如有问题，请检查：
1. 配置文件是否正确设置
2. API Key是否有效
3. 网络连接是否正常
4. 查看日志文件获取详细错误信息

---

🎉 现在你可以开始生成高质量的Merlin Chain function calling数据集了！

- 🔥 使用 Deepseek API 作为数据生成引擎
- 🔗 集成 Merlin Chain MCP 客户端，支持真实工具调用
- 📊 支持批量生成，避免API限流
- 💾 自动保存中间结果，防止数据丢失
- 🎯 针对 Merlin Chain (BTC Layer2) 的20个MCP工具优化
- 📝 两阶段生成：问题生成 + 完整对话生成
- ⚡ 模块化设计，组件可复用

## 项目结构

```
ai_dataset/
├── prompt.txt                      # 问题生成任务的提示词
├── prompt-2.txt                    # 完整对话生成任务的提示词
├── generate_question_dataset.py    # 生成问题数据集（阶段1）
├── generate_complete_dataset.py    # 生成完整对话数据集（阶段2）
├── merlin_mcp_client.py            # Merlin Chain MCP客户端封装
├── deepseek_api_client.py          # DeepSeek API客户端封装
├── dataset_utils.py                # 数据集处理工具类
└── requirements.txt                # Python依赖列表
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 获取 Deepseek API Key

访问 [Deepseek官网](https://platform.deepseek.com/) 获取API Key

### 3. 两阶段数据生成

#### 阶段1：生成问题数据集

```bash
# 使用默认参数 (6000个对话，批次大小50)
python generate_question_dataset.py

# 自定义参数
python generate_question_dataset.py --total_conversations 1000 --batch_size 25 --output_file my_questions.json

# 完整参数示例
python generate_question_dataset.py \
    --total_conversations 2000 \
    --batch_size 30 \
    --output_file custom_dataset.json \
    --prompt_file prompt.txt \
    --api_key YOUR_API_KEY
```

生成用户问题和assistant引导对话，默认输出到 `function_calling_dataset.json`

#### 阶段2：生成完整对话数据集

```bash
# 使用默认参数
python generate_complete_dataset.py

# 自定义参数
python generate_complete_dataset.py --input_file my_questions.json --output_file my_complete.json --batch_size 2

# 断点续传 (从第100个对话开始)
python generate_complete_dataset.py --start_from 100

# 完整参数示例
python generate_complete_dataset.py \
    --input_file custom_dataset.json \
    --output_file custom_complete.json \
    --batch_size 1 \
    --start_from 0 \
    --api_key YOUR_API_KEY
```

基于问题数据集，调用MCP工具生成完整对话，默认输出到 `function_calling_dataset-2.json`

## 核心组件

### MCP客户端 (`merlin_mcp_client.py`)
- 封装与Merlin Chain MCP服务器的连接
- 支持SSE协议和工具调用
- 自动获取工具schema

### API客户端 (`deepseek_api_client.py`) 
- 封装DeepSeek API调用
- 支持function calling
- 包含重试机制和错误处理

### 工具类 (`dataset_utils.py`)
- JSON数据解析和格式转换
- 文件读写操作
- ShareGPT格式支持

## 命令行参数

### 问题数据集生成脚本参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `--total_conversations` | int | 6000 | 目标对话数量 |
| `--batch_size` | int | 50 | 每批生成的对话数量 |
| `--output_file` | str | function_calling_dataset.json | 输出文件名 |
| `--prompt_file` | str | prompt.txt | 提示词文件路径 |
| `--api_key` | str | - | DeepSeek API Key (可选) |

### 完整数据集生成脚本参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `--input_file` | str | function_calling_dataset.json | 输入问题数据集文件 |
| `--output_file` | str | function_calling_dataset-2.json | 输出完整数据集文件 |
| `--batch_size` | int | 1 | 每批处理的对话数量 |
| `--start_from` | int | 0 | 从第几个对话开始处理 |
| `--api_key` | str | - | DeepSeek API Key (可选) |

### 查看帮助信息

```bash
# 查看问题生成脚本帮助
python generate_question_dataset.py --help

# 查看完整数据集生成脚本帮助
python generate_complete_dataset.py --help
```

## 输出格式

```json
{
  "metadata": {
    "total_conversations": 1000,
    "generated_at": "2024-01-01 12:00:00",
    "target_conversations": 6000,
    "batch_size": 50
  },
  "conversations": [
    {
      "conversations": [
        {
          "from": "user",
          "value": "我想查看这个交易的详情"
        },
        {
          "from": "assistant", 
          "value": "请提供交易哈希..."
        }
      ]
    }
  ]
}
```

## 使用示例

### 快速开始 (使用默认参数)
```bash
# 第一步：生成1000个问题对话 (小规模测试)
python3 generate_question_dataset.py --total_conversations 1000 --batch_size 20

# 第二步：生成完整数据集
python3 generate_complete_dataset.py --batch_size 1
```

### 生产环境使用
```bash
# 生成6000个问题对话 (默认目标)
python3 generate_question_dataset.py

# 生成完整数据集，批次大小为2，提高效率
python3 generate_complete_dataset.py --batch_size 2
```

### 断点续传
```bash
# 如果处理中断，从第500个对话继续
python3 generate_complete_dataset.py --start_from 500
```

### 自定义文件名
```bash
# 使用自定义文件名
python3 generate_question_dataset.py --output_file my_questions.json
python3 generate_complete_dataset.py --input_file my_questions.json --output_file my_complete.json
```

## 注意事项

1. **API限流**: 脚本会在批次间自动延迟，避免触发限流
2. **中间结果**: 每批生成的数据都会保存为临时文件
3. **错误处理**: 单批失败不会影响整体进程
4. **数据质量**: 建议生成后人工审核部分数据
5. **批次大小**: 问题生成建议50个/批，完整对话生成建议1-2个/批
6. **进度跟踪**: 支持断点续传，可随时中断和恢复

## 故障排除

### API调用失败
- 检查API Key是否正确
- 确认网络连接正常
- 验证Deepseek服务状态

### JSON解析错误
- 检查prompt.txt格式是否正确
- 确认API返回内容格式

### 生成数据不符合预期
- 调整prompt.txt中的指令
- 修改temperature参数控制随机性

## 贡献

欢迎提交Issue和Pull Request来改进此项目。

## 许可证

MIT License
# V8 配置文件说明

本目录包含 V8 量化交易系统的所有配置文件。

## 文件列表

### `v8.yaml`
**用途**: 系统运行时配置  
**内容**: 交易对、K 线周期、门控参数、日志级别等  
**注意**: 不包含任何密钥，密钥从环境变量读取

### `api_keys.yaml`
**用途**: API 密钥环境变量名映射  
**内容**: OKX、Telegram、钉钉、Triton 等服务的配置  
**注意**: 此文件只定义环境变量名，实际密钥需设置在系统环境变量中

## 环境变量设置

在运行 V8 系统前，需要设置以下环境变量：

### OKX 交易所（必选）
```bash
# 实盘
export OKX_API_KEY="your_api_key"
export OKX_SECRET_KEY="your_secret_key"
export OKX_PASSPHRASE="your_passphrase"

# 模拟盘（可选）
export OKX_DEMO_API_KEY="your_demo_api_key"
export OKX_DEMO_SECRET_KEY="your_demo_secret_key"
export OKX_DEMO_PASSPHRASE="your_demo_passphrase"
```

### 告警通知（可选）
```bash
# Telegram
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# 钉钉
export DINGTALK_WEBHOOK="your_webhook_url"
export DINGTALK_SECRET="your_secret"  # 可选
```

## 配置加载顺序

1. `common/config.py` 中的 `load_config()` 函数加载 `config/v8.yaml`
2. `common/config.py` 中的 `load_api_keys()` 函数加载 `config/api_keys.yaml`
3. API 密钥从环境变量读取，如果环境变量未设置则抛出异常

## 安全提醒

- ❌ **绝对不要**将真实密钥提交到配置文件
- ❌ **绝对不要**将配置文件提交到 Git 仓库
- ✅ 使用 `.gitignore` 忽略包含敏感信息的文件
- ✅ 使用环境变量或密钥管理服务存储真实密钥

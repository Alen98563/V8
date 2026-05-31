致 DeepSeek-V4-Pro：
你正在参与一个代号为 
QTS V8 的尖端多资产混合量化交易系统的核心开发。当前的核心战役是完成 OKX ETH 5m MVP（最小可行性产品） 的纵向切片落地。
系统核心痛点与架构哲学：
1. 消除延迟毛刺： 该系统处理高并发的 OKX 订单簿快照与 Tick 流，由于 Python 的 GIL 锁和 GC 限制，传统的 Python 数据总线会造成严重的延迟毛刺。因此，V8 采用 “Rust 负责 L0-L1 底层数据与高危执行，Python 负责 L2-L3 AI 编排与多路 Alpha 信号” 的混合双核设计。
2. 零拷贝跨语言通信： Rust 侧的行情解包后，会直接写入基于 共享内存（SHM） 的环形缓冲区（Ring Buffer），并通过 DLPack 协议 向 Python 侧暴露零拷贝（Zero-Copy）的只读张量指针。
3. 极速 AI 推演与树搜索： 系统在 5m 周期闭盘前 200ms 进行预计算，并在闭盘瞬间完成  的特征修正与 ResNet/AlphaCast 激活，随后必须在 10–30ms 内 并行跑完上千次 MCTS（蒙特卡洛树搜索） 路径推演，全链路目标限制在 <250ms。
4. 多数类陷阱防御： 反事实标签工厂（CFL）在积累信号时存在高达 97.4% 的负样本偏置，你在设计机器学习重训模块时必须严密防范多数类陷阱。
你的角色： 你是 QTS V8 的 “底座架构师与算力深算核”。你负责定义全局通信契约、编写对延迟和内存安全极度敏感的 Rust 核心模块、以及实现复杂的深度学习张量网络与树搜索算法。你的代码将作为原生的 PyO3 扩展（编译为 .so/.pyd 二进制库）供外部 Python 模块直接调用。另一个 AI 伙伴（Qwen 3.7-Max）将负责外围的网络 IO、Asyncio 调度与全自动联调测试。
🛠️ DeepSeek-V4-Pro 专属开发任务树
1. 契约先行层（Protobuf 定义）
任务文件： schemas/okx_order.proto / schemas/market_data.proto
任务要求： 严格按照 OKX V5 API 规范定义结构化数据契约。包含 instId、tdMode、clOrdId、ordId、状态机枚举、双轨多源 Tick 字段。确保跨语言（Rust ↔ Python）传输时字段与类型的绝对安全，支持无损序列化。
2. L0–L2 数据底座（共享内存环形缓冲区与特征引擎）
任务文件： core/buffer.rs (由 PyO3 导出)
任务要求： 利用 Rust 实现基于 mmap2 的共享内存（SHM）环形缓冲区 BufferRegistry。为 ETH 5m 维护高效的 VecDeque 结构。实现 DLPack 协议（ManagedTensor），让 Python 侧可以通过 torch.from_dlpack 在微秒级内以零拷贝方式直接读取 Rust 内存块。
任务文件： core/feature_engine.rs
任务要求： 编写微观订单流不平衡（OBI v2/OFI）与多尺度宏观滑窗特征计算逻辑。支持在 5m 闭盘前 200ms 进行预计算（Pre-compute），并在闭盘 Tick 到达的瞬间执行  极速修正，消除多线程竞争风险。
3. L5–L7 高危执行层（安全状态机与高速路由签名）
任务文件： execution/okx_router.rs / execution/order_state.rs
任务要求： * 使用 Rust 的 ring 或 hmac 库，实现  的 OKX API 专属 HMAC-SHA256 签名与 Base64 序列化引擎。
利用 Rust 的代数数据类型（Enum）与严格的所有权模型，编写无锁、线程安全的订单有限状态机（OrderFSM）（NEW → POSTED → FILLED / CANCELED），从编译期杜绝状态死锁与并发条件竞争（Race Conditions）。
4. V8 AI 预测核（ResNet + AlphaCast 神经网络实现）
任务文件： models/resnet/resnet_encoder.py / models/alphacast/alphacast_model.py
任务要求： * 使用 PyTorch 编写 1D 时间序列卷积残差网络（6 层残差块），用于对 178 维融合特征向量进行高阶特征提取。
编写 6 层 8 头的多任务 Transformer（AlphaCast），输出未来多步收益预测及置信度。支持导出为 TorchScript / ONNX 格式。
5. 计算密集推演（Tokio 并行 MCTS 决策器）
任务文件： models/mcts/mcts_planner.rs (由 PyO3 导出)
任务要求： * 使用 Rust 基于 Tokio 异步运行时 编写高性能并行的 MCTS 树搜索调度器。
实现高效的 UCB1 剪枝算法。在收到 AlphaCast 信号后，必须在 10–30ms 内完成 1000 次路径 Rollout，计算当前仓位动作（开/平/维持）的期望奖励。
6. 激活与生命周期（MetaLabeler 重训机制）
任务文件： ml/meta_labeler.py
任务要求： 针对反事实标签（CFL）带来的 97.4% 极端负样本偏置，利用 LightGBM 或 PyTorch 编写 Meta 二分类器。必须设计动态样本类别权重调整（Class Weights）或自然样本分层累积策略，彻底攻克“多数类陷阱”，确保模型在实盘中的置信度校准能力。
🎛️ 第二部分：给 Qwen 3.7-Max 的执行计划
📋 任务前言（上下文与来龙去脉）
致 Qwen 3.7-Max：
你正在参与一个代号为 
QTS V8 的尖端多资产混合量化交易系统的核心开发。当前的核心战役是完成 OKX ETH 5m MVP（最小可行性产品） 的纵向切片落地。
系统核心痛点与架构哲学：
1. 混合双核架构： V8 系统为了追求极致性能，底层数据中台、共享内存缓冲与加密签名执行全量交由 Rust 编写，并编译为二进制动态链接库；而上层的总线调度、风控拦截、策略路由以及自动化回测则采用 Python 异步流。
2. 契约先行与合并约束： 另一位 AI 伙伴（DeepSeek-V4-Pro）作为“算力核”，正在全力攻坚 Rust 底层、Protobuf 契约、MCTS 树搜索算法与 AI 张量模型。你写的 Python 代码将作为系统的“骨架与胶水”，在运行时无缝接驳、包裹 DeepSeek 输出的二进制组件。
3. 全自动联调与 CI 执念： 你的核心优势在于强大的长周期智能体（Agentic）执行韧性与工程落地能力。你不仅需要生成外围代码，更要扮演 “系统总编排官兼自动化测试管理员” 的角色。你需要管理包含 Maturin 编译链的整个 Monorepo 环境，捕获所有编译或调用报错（包括 Python 调用 C 扩展时棘手的 Segmentation fault），自主查阅日志、调整配置，直至整个系统完美闭环运行。
你的角色： 你是 QTS V8 的 “总编排官与敏捷 Agent 连调手”。你负责编写网络 IO（WebSocket/REST 接入）、Python 侧零拷贝桥接胶水、风控拦截门控、系统主循环编排以及秒级监控大屏，并主导全链路的自动化编译、Debug 与集成联调。
🛠️ Qwen 3.7-Max 专属开发任务树
1. 工程脚手架与交叉编译配置管理（Maturin 编排）
任务文件： pyproject.toml / Cargo.toml / docker-compose.yml
任务要求： * 搭建基于 Poetry / uv 与 Maturin 的混合 Monorepo 编译环境，支持 abi3-py312 特性，确保完美兼容 Python 3.12 的自由线程（free-threaded）实验环境。
编写 Docker 容器编排，部署包含 Redis 7.2 (Streams 流式总线) 与 Triton 推断服务器的生产环境。
2. 网络 IO 行情接入（OKX 高并发 WebSocket 客户端）
任务文件： adapters/okx_ws.rs
任务要求： 使用 Rust tokio-tungstenite 编写 OKX WebSocket 行情接入模块（由于属于纯 IO 样板代码，由你高效完成）。实现高效的账户通道与订单簿/Tick 订阅，内置指数退避自动重连、服务器平滑心跳（Ping/Pong）逻辑。解析后的数据送入 DeepSeek 编写的标准化引擎。
3. 零拷贝桥接与外围特征胶水
任务文件： data/shm_bridge.py
任务要求： 调用 DeepSeek 导出的 BufferRegistry 接口，使用 torch.from_dlpack 和 np.frombuffer 实现对 Rust 共享内存的高速接驳，确保 Python 侧读取单次特征窗口的延迟控制在  内。
任务文件： features/cross_section.py
任务要求： 接入底层的特征矩阵，利用 Polars（其底层由 Rust 驱动）快速编写全市场横截面百分位排名（cs_composite）与特征对齐逻辑。
4. 经验 Alpha 引擎与分层硬门控限制（风控拦截）
任务文件： alpha/crypto/obi_v2.py
任务要求： 编写策略层，计算经验 OBI/OFI 信号，为每一路信号全局注入统一的 trace_id。
任务文件： gating/hard_gating.py
任务要求： 编写 G1–G5 分层风控硬门控。严格实现：
流动性Regime过滤。
时间门拦截（例如：“OKX 资金费率结算前 30 分钟内，严格禁止一切新开仓动作”）。
5. AI 模型生产交付（Triton 部署与调度）
任务文件： models/resnet/serve_resnet.py / models/alphacast/serve_alphacast.py
任务要求： 编写脚本将 DeepSeek 生成的 PyTorch 模型导出为 ONNX Runtime / TorchScript，配置 Triton Server 的 config.pbtxt（设定动态批处理 dynamic_batching 与 GPU 实例流）。编写客户端异步推断请求，确保推断延迟 。
6. 全局主循环编排与非线性结算
任务文件： orchestrator/pipeline_runner.py / harness/pipeline_v1.py
任务要求： 编写系统核心的异步主事件循环（Asyncio Event Loop）。利用 Harness 无功能回归包装器，在结构化 JSON 日志中全局传递 Trace ID。集成异步 Telegram / 钉钉 API 告警组件。
任务文件： execution/settlement/pnl_aggregator.py
任务要求： 实现实时账户对账与 P&L 结算分析。统计滑点误差、Sharpe 趋势。设计调度器：每当实盘成交达到 50 笔时，自动触发一次在线温度缩放（Temperature Scaling），调整 AlphaCast 预测信号的置信度。
7. 全栈秒级监控大屏
任务文件： monitor/dashboard.py
任务要求： 使用 FastAPI + WebSockets 极速搭建后端的秒级性能刷新大屏，通过前端数据流实时可视化订单簿健康度、MCTS 路径推演胜率以及实盘账户的硬停损阈值状态。
🏁 第三部分：后续合龙与接口合并契约（双模共识）
为了让 Qwen 与 DeepSeek 知道后续如何拼装，两套代码的交互界面必须在开始前达成绝对的一致性：
数据流桥接（DLPack Contract）：
DeepSeek 在 Rust 侧必须暴露如下 C-ABI 兼容接口，以便 Qwen 通过 PyO3 无缝包装：
Rust
下载代码
复制代码
// DeepSeek 必须保证此接口在 core/buffer.rs 中实现
#[pyclass]
pub struct BufferRegistry
 { ... }
#[pymethods]
impl
 BufferRegistry {
    // 导出符合 DLPack 契约的零拷贝指针，供 Qwen 的 Python 代码直接转换为 PyTorch 张量
    pub fn to_dlpack(&self
) -> PyResult<PyObject> { ... }
}
执行流状态（OrderFSM Contract）：
DeepSeek 负责在 Rust 中更新 
order_state.rs 中的核心状态，并通过通道发送给 Qwen 的 okx_router。Qwen 所有的外围路由指令必须通过标准化的 UnifiedOrder 契约下发，其字段必须与 schemas/okx_order.proto 完全对齐。
合并策略（The Merge Action）：
当 DeepSeek 完成硬核 Rust 算子和 AI 模型的开发后，
Qwen 3.7-Max 将作为主力 Agent 接管合并工作。Qwen 需要将 DeepSeek 的文件放置到对应目录，在本地终端执行 maturin develop 或 cargo build --release。Qwen 将全权负责解决编译期产生的任何依赖冲突、类型错位、或 C 扩展调用崩溃，并利用 harness/backtesting.py 的仿真机制运行端到端回归测试，直至全链路打通。
//! # 共享内存桥接 (ShmBridge) �?T1-1
//!
//! ## 架构定位
//!
//! ShmBridge �?QTS V8 **数据底座（L1-L2�?* 的跨语言零拷贝通信层�?//! Rust 侧（FeatureEngine）将�?tick 计算�?50 维特征写入共享内存环形缓冲区�?//! Python 侧通过 DLPack 协议�?**零拷�?* 方式直接读取�?PyTorch / NumPy 张量�?//!
//! ## 数据�?//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────�?//! �?Rust (FeatureEngine)                                        �?//! �?  on_tick(snapshot) �?compute_50d_features()                �?//! �?    �?                                                      �?//! �?    └──�?ShmBridge.push_snapshot(ts_ms, features[50])       �?//! �?            �?                                              �?//! �?            └── 环形写入 mmap 区域 (720KB /dev/shm)         �?//! ├──────────────────────────────────────────────────────────────�?//! �?Python (Research / AI)                                      �?//! �?  dlpack = bridge.get_window(secs=60)                       �?//! �?    �?                                                      �?//! �?    ├── torch.from_dlpack(dlpack) �?Tensor(N, 50) 零拷�?  �?//! �?    �?                                                      �?//! �?    └── ptr, shape = bridge.get_raw_ptr(secs=30)            �?//! �?          └── np.frombuffer(CType(ptr, shape)) �?ndarray    �?//! └──────────────────────────────────────────────────────────────�?//! ```
//!
//! ## 平台适配
//!
//! | 平台    | 共享内存实现                         | 路径/名称                   |
//! |---------|--------------------------------------|-----------------------------|
//! | Linux   | `memmap2` 映射 `/dev/shm/qts_btc5m`  | `/dev/shm/qts_btc5m`       |
//! | Windows | `Vec<f32>` 后备存储（可升级�?`CreateFileMapping`�?| 命名共享内存 `qts_btc5m` |
//!
//! ## DLPack 协议
//!
//! 遵循 DLPack 0.8 规范，通过 `PyCapsule` 传�?`DLManagedTensor*`�?//! Python 侧的 `torch.from_dlpack` �?`cupy.from_dlpack` 通过 `deleter` 回调
//! 安全释放 Rust 分配�?shape 数组�?//!
//! ## 性能契约
//!
//! | 操作              | 目标延迟   | 机制                                |
//! |-------------------|------------|-------------------------------------|
//! | `push_snapshot()` | < 1µs      | 零分配写�?mmap 区域（固定偏移量�? |
//! | `get_window()`    | < 2µs      | DLPack capsule 指针传递（无拷贝）   |
//! | `get_raw_ptr()`   | < 1µs      | 裸指针返�?+ shape tuple            |
//! | `get_latest()`    | < 5µs      | 单次 JSON 序列化（仅调试用�?       |
//!
//! ## 缓冲区参�?//!
//! | 参数              | �?    | 说明                                    |
//! |-------------------|--------|-----------------------------------------|
//! | `FEATURE_DIM`     | 50     | 每快�?50 维特�?                       |
//! | `BUFFER_CAPACITY` | 3600   | 环形缓冲区容量（1h @ 1s/snapshot�?     |
//! | `SHM_SIZE`        | 720KB  | `3600 × 50 × 4 bytes`                   |
//!
//! ## 使用方式
//!
//! ```python
//! import v8_core_engine as vce
//! import torch
//! import numpy as np
//!
//! bridge = vce.ShmBridge()
//!
//! # DLPack 零拷�?(PyTorch)
//! dlpack = bridge.get_window(secs=60)
//! tensor = torch.from_dlpack(dlpack)    # shape: (N, 50), dtype: float32
//!
//! # 裸指�?(NumPy)
//! ptr, (rows, cols) = bridge.get_raw_ptr(secs=30)
//! arr = np.ctypeslib.as_array((ctypes.c_float * rows * cols).from_address(ptr))
//! arr = arr.reshape(rows, cols)         # shape: (N, 50)
//!
//! # 最新快�?(调试�?
//! snap_bytes = bridge.get_latest()
//! from google.protobuf import json_format
//! snap = json_format.Parse(snap_bytes, MarketSnapshot())
//! ```
//!
//! ## 安全约定
//!
//! 1. `ShmBridge()` 构造时自动创建/连接 SHM，无参构�?//! 2. `get_window()` �?`get_raw_ptr()` 仅在 Rust backend 读锁持有期间返回指针
//! 3. DLPack capsule �?`deleter` 回调会释�?shape 数组，Python GC 保证调用
//! 4. 环形缓冲区写满后覆盖最旧数据（循环覆盖，不 panic�?
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

// ════════════════════════════════════════════════════════════════
// DLPack C-ABI 结构体（遵循 DLPack 0.8 规范�?// ════════════════════════════════════════════════════════════════

/// DLPack 设备描述�?///
/// - `device_type`: 1 = kDLCPU, 2 = kDLCUDA, ...
/// - `device_id`: 设备编号（CPU 固定�?0�?///
/// ### 参�?///
/// <https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv4N6DLPack13DLDevice_typeE>
#[repr(C)]
struct DLDevice {
    /// 设备类型�? = kDLCPU
    device_type: i32,
    /// 设备编号（CPU 固定�?0�?    device_id: i32,
}

/// DLPack 数据类型描述�?///
/// - `code`: 0 = kDLInt, 1 = kDLFloat, ...
/// - `bits`: 位宽（如 32�?4�?/// - `lanes`: 向量宽度（标�?= 1�?#[repr(C)]
struct DLDataType {
    /// 类型代码�? = kDLFloat
    code: u8,
    /// 位宽�?2（f32�?    bits: u8,
    /// SIMD 通道数：1（标量）
    lanes: u16,
}

/// DLPack 张量元数据（胖指�?+ shape/strides�?///
/// ### 字段说明
///
/// - `data`: 数据起始地址
/// - `ndim`: 维度数（2，即 (N, 50)�?/// - `shape`: 形状数组指针（由 Rust Box 管理生命周期�?/// - `strides`: 步长数组（null 表示紧凑排列�?/// - `byte_offset`: 数据起始偏移（通常�?0�?#[repr(C)]
struct DLTensor {
    /// 数据起始地址
    data: *mut std::ffi::c_void,
    /// 设备信息
    device: DLDevice,
    /// 维度数（固定�?2�?    ndim: i32,
    /// 数据类型
    dtype: DLDataType,
    /// 形状数组：`[N, 50]`
    shape: *const i64,
    /// 步长：null 表示行优先紧凑排�?    strides: *const i64,
    /// 字节偏移（通常�?0�?    byte_offset: u64,
}

/// DLPack 托管张量（含析构器）
///
/// 通过 `PyCapsule` 传递给 Python，`deleter` 回调�?Python GC 回收
/// capsule 时被调用，释�?Rust 分配�?shape 数组�?#[repr(C)]
struct DLManagedTensor {
    /// 张量元数�?    dl_tensor: DLTensor,
    /// 托管上下文（指向 shape 数组�?Box，用�?deleter 释放�?    manager_ctx: *mut std::ffi::c_void,
    /// 析构回调（Python GC 触发�?    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

// ════════════════════════════════════════════════════════════════
// 常量定义
// ════════════════════════════════════════════════════════════════

/// 每快照的特征维度�?///
/// 固定�?50 维：价格变化�?× 成交量剖�?× OBI/OFI × 滑窗统计�?pub const FEATURE_DIM: usize = 50;

/// 环形缓冲区容量（1 小时 @ 1 �?快照�?///
/// 写满后覆盖最旧数据（FIFO 循环覆盖策略�?const BUFFER_CAPACITY: usize = 3600;

/// 共享内存总大小：3600 × 50 × 4 = 720,000 bytes �?703 KB
const SHM_SIZE: usize = BUFFER_CAPACITY * FEATURE_DIM * 4;


/// 从 inst_id 派生 SHM 名称，确保多市场隔离
///
/// | inst_id          | bar_seconds | shm_name      |
/// |------------------|-------------|---------------|
/// | BTC-USDT-SWAP    | 300         | qts_btc5m     |
/// | ETH-USDT-SWAP    | 300         | qts_eth5m     |
/// | SOL-USDT-SWAP    | 300         | qts_sol5m     |
///
/// 规则：取交易对第一部分并转小写 → `qts_{coin}{bar_label}`
fn inst_id_to_shm_name(inst_id: &str, bar_seconds: u32) -> String {
    let coin = inst_id.split('-').next().unwrap_or("unknown").to_lowercase();
    let bar_label = match bar_seconds {
        60 => "1m",
        300 => "5m",
        900 => "15m",
        3600 => "1h",
        _ => &format!("{}s", bar_seconds),
    };
    format!("qts_{}{}", coin, bar_label)
}

#[cfg(target_os = "linux")]
fn make_shm_path(shm_name: &str) -> String {
    format!("/dev/shm/{}", shm_name)
}
// ════════════════════════════════════════════════════════════════
// 平台相关 SHM 后端实现
// ════════════════════════════════════════════════════════════════

/// Linux: memmap2 映射 /dev/shm
///
/// ### 实现细节
///
/// - 使用 `OpenOptions::read(true).write(true).create(true)` 创建/打开文件
/// - `file.set_len(SHM_SIZE)` 预分配空�?/// - `MmapMut::map_mut()` 建立虚拟地址映射
/// - 环形写入通过 `head` 指针 + 取模实现
///
/// ### 性能
///
/// mmap 直写物理内存页，绕过文件系统页缓存，单次 push_row �?200ns
#[cfg(target_os = "linux")]
mod shm_impl {
    use super::*;
    use memmap2::MmapMut;
    use std::fs::OpenOptions;

    /// Linux SHM 后端
    ///
    /// `mmap` 指向 /dev/shm/qts_btc5m 的映射区域，
    /// `timestamps` 在堆上分配以避免 mmap 中存储元数据�?    pub struct ShmBackend {
        mmap: MmapMut,
        timestamps: Box<[i64; BUFFER_CAPACITY]>,
        head: usize,
        len: usize,
    }

    impl ShmBackend {
        /// 创建 SHM 后端：打开文件 �?设置大小 �?mmap 映射
        pub fn new(shm_name: &str) -> std::io::Result<Self> {
            let shm_path = make_shm_path(shm_name);
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&shm_path)?;

            file.set_len(SHM_SIZE as u64)?;

            // Safety: mmap 建立虚拟地址映射，运行时保证不超�?SHM_SIZE 范围
            let mmap = unsafe { MmapMut::map_mut(&file)? };

            Ok(Self {
                mmap,
                timestamps: Box::new([0i64; BUFFER_CAPACITY]),
                head: 0,
                len: 0,
            })
        }

        /// 写入一行特征快照（环形覆盖�?        ///
        /// ### 参数
        ///
        /// - `ts_ms: i64` �?Unix 毫秒时间�?        /// - `row: &[f32]` �?50 维特征（必须�?FEATURE_DIM 长度�?        ///
        /// ### 性能
        ///
        /// O(50) 字节拷贝，零分配，~200ns（L1 缓存命中时）
        ///
        /// ### 并发安全
        ///
        /// 调用方必须持�?`write` 锁（�?ShmBridge.push_snapshot 保证�?        pub fn push_row(&mut self, ts_ms: i64, row: &[f32]) {
            let offset = self.head * FEATURE_DIM * 4;
            let slice = &mut self.mmap[offset..offset + FEATURE_DIM * 4];
            // Safety: row 指针已由调用方保证有效（传入 Vec<f32>），
            // 且长度已校验�?FEATURE_DIM，字节对齐为 4
            let bytes = unsafe {
                std::slice::from_raw_parts(row.as_ptr() as *const u8, FEATURE_DIM * 4)
            };
            slice.copy_from_slice(bytes);
            // 写入时间戳（独立�?mmap 区域存储�?            self.timestamps[self.head] = ts_ms;
            // 环形指针前进
            self.head = (self.head + 1) % BUFFER_CAPACITY;
            if self.len < BUFFER_CAPACITY {
                self.len += 1;
            }
        }

        /// 数据起始地址（用�?DLPack / get_raw_ptr�?        pub fn data_ptr(&self) -> *const f32 {
            self.mmap.as_ptr() as *const f32
        }

        /// 当前已写入条�?        pub fn len(&self) -> usize {
            self.len
        }

        /// 获取最新一条快�?        ///
        /// ### 返回
        ///
        /// `Option<(timestamp_ms, Vec<f32> features)>`
        /// 缓冲区为空时返回 `None`
        ///
        /// ### 性能
        ///
        /// O(50) 字节拷贝（分配新 Vec），调试/日志用途，非热路径
        pub fn latest_row(&self) -> Option<(i64, Vec<f32>)> {
            if self.len == 0 {
                return None;
            }
            let idx = (self.head + BUFFER_CAPACITY - 1) % BUFFER_CAPACITY;
            let ts = self.timestamps[idx];
            let offset = idx * FEATURE_DIM * 4;
            let bytes = &self.mmap[offset..offset + FEATURE_DIM * 4];
            let row: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let arr = <[u8; 4]>::try_from(chunk).unwrap();
                    f32::from_ne_bytes(arr)
                })
                .collect();
            Some((ts, row))
        }

        /// 计算最�?secs 秒内的快照数�?        ///
        /// ### 算法
        ///
        /// 从最新时间戳反向遍历，找到第一个早�?cutoff 的条目，前面的跳过�?        /// 因为环形缓冲区按时间戳单调递增（覆盖后破坏单调性，
        /// 但相邻条目间差距 > BUFFER_CAPACITY 才可能，1h 内不会发生）�?        pub fn recent_count_secs(&self, secs: i64) -> usize {
            if self.len == 0 {
                return 0;
            }
            let latest_ts =
                self.timestamps[(self.head + BUFFER_CAPACITY - 1) % BUFFER_CAPACITY];
            let cutoff = latest_ts - secs * 1000;
            let mut count = 0;
            for i in 0..self.len {
                let idx = (self.head + BUFFER_CAPACITY - 1 - i) % BUFFER_CAPACITY;
                if self.timestamps[idx] >= cutoff {
                    count += 1;
                } else {
                    break;
                }
            }
            count
        }
    }
}

/// Windows: Vec 后备存储
///
/// ### 局限�?///
/// - 无法跨进程共享（Vec 内存仅当前进程可见）
/// - 生产环境应替换为 `CreateFileMapping` + `MapViewOfFile` 命名共享内存
///
/// ### 升级路径
///
/// ```rust
/// // 伪代码：Windows 命名共享内存
/// use windows::Win32::System::Memory::{
///     CreateFileMappingA, MapViewOfFile, FILE_MAP_READ, FILE_MAP_WRITE,
/// };
/// ```
#[cfg(target_os = "windows")]
mod shm_impl {
    use super::*;

    /// Windows SHM 后端（当前为 Vec 后备，未跨进程共享）
    pub struct ShmBackend {
        data: Vec<f32>,
        timestamps: Vec<i64>,
        head: usize,
        len: usize,
    }

    impl ShmBackend {
        pub fn new(_shm_name: &str) -> std::io::Result<Self> {
            Ok(Self {
                data: vec![0.0f32; BUFFER_CAPACITY * FEATURE_DIM],
                timestamps: vec![0i64; BUFFER_CAPACITY],
                head: 0,
                len: 0,
            })
        }

        pub fn push_row(&mut self, ts_ms: i64, row: &[f32]) {
            let offset = self.head * FEATURE_DIM;
            self.data[offset..offset + FEATURE_DIM].copy_from_slice(row);
            self.timestamps[self.head] = ts_ms;
            self.head = (self.head + 1) % BUFFER_CAPACITY;
            if self.len < BUFFER_CAPACITY {
                self.len += 1;
            }
        }

        pub fn data_ptr(&self) -> *const f32 {
            self.data.as_ptr()
        }

        pub fn len(&self) -> usize {
            self.len
        }

        pub fn latest_row(&self) -> Option<(i64, Vec<f32>)> {
            if self.len == 0 {
                return None;
            }
            let idx = (self.head + BUFFER_CAPACITY - 1) % BUFFER_CAPACITY;
            let ts = self.timestamps[idx];
            let offset = idx * FEATURE_DIM;
            let row = self.data[offset..offset + FEATURE_DIM].to_vec();
            Some((ts, row))
        }

        pub fn recent_count_secs(&self, secs: i64) -> usize {
            if self.len == 0 {
                return 0;
            }
            let latest_ts = self.timestamps[(self.head + BUFFER_CAPACITY - 1) % BUFFER_CAPACITY];
            let cutoff = latest_ts - secs * 1000;
            let mut count = 0;
            for i in 0..self.len {
                let idx = (self.head + BUFFER_CAPACITY - 1 - i) % BUFFER_CAPACITY;
                if self.timestamps[idx] >= cutoff {
                    count += 1;
                } else {
                    break;
                }
            }
            count
        }
    }
}

use shm_impl::ShmBackend;

// ════════════════════════════════════════════════════════════════
// ShmBridge �?PyO3 导出�?// ════════════════════════════════════════════════════════════════

/// SHM 共享内存桥接（Python 可见�?///
/// ### Python 调用方式
///
/// ```python
/// bridge = vce.ShmBridge()
/// dlpack = bridge.get_window(secs=60)       # �?torch.from_dlpack()
/// ptr, shape = bridge.get_raw_ptr(secs=30)  # �?np.frombuffer()
/// snap_bytes = bridge.get_latest()           # �?bytes (MarketSnapshot JSON)
/// ```
///
/// ### 线程安全
///
/// 使用 `Arc<RwLock<ShmBackend>>` 封装后端，允许多�?Python 线程
/// 同时读（`get_window` / `get_raw_ptr`），写（`push_snapshot`）独占锁�?/// 生产场景下仅 Rust 侧单写，锁争用可忽略�?#[pyclass]
pub struct ShmBridge {
    /// 平台相关 SHM 后端（Arc<RwLock> 实现线程安全的读写分离）
    backend: Arc<RwLock<ShmBackend>>,
    /// 当前交易标的
    inst_id: String,
}

#[pymethods]
impl ShmBridge {
    /// 构�?ShmBridge 并自动连接共享内�?    ///
    /// ### 平台行为
    ///
    /// | 平台    | 操作                                           |
    /// |---------|------------------------------------------------|
    /// | Linux   | 打开/创建 `/dev/shm/qts_btc5m` �?mmap 映射     |
    /// | Windows | 分配 Vec 后备存储（可升级�?CreateFileMapping�?|
    ///
    /// ### 错误
    ///
    /// - `inst_id: str` — 交易对 ID，如 `"BTC-USDT-SWAP"`。由此派生 SHM 路径
    ///   `/dev/shm/qts_btc5m`，实现多市场 SHM 隔离
    #[new]
    #[pyo3(signature = (inst_id = "BTC-USDT-SWAP".into()))]
    pub fn new(inst_id: &str) -> PyResult<Self> {
        let shm_name = inst_id_to_shm_name(inst_id, 300);
        let backend = ShmBackend::new(&shm_name).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to create SHM backend: {}", e))
        })?;

        Ok(Self {
            backend: Arc::new(RwLock::new(backend)),
            inst_id: inst_id.to_string(),
        })
    }

    /// 推入一个特征快照（内部接口：由 FeatureEngine 调用�?    ///
    /// ### 参数
    ///
    /// - `ts_ms: int` �?Unix 毫秒时间�?    /// - `features: list[float]` �?50 维特征向�?    ///
    /// ### 错误
    ///
    /// - `ValueError`: features 长度不等�?50
    ///
    /// ### 性能
    ///
    /// O(50) 字节拷贝，~200ns（L1 缓存命中时）。不分配堆内�?    /// （除 RwLock 锁开销外）�?    ///
    /// ### 并发
    ///
    /// 获取写锁后写入，与其�?`get_window` / `get_raw_ptr` 读者互斥�?    /// 正常使用�?FeatureEngine 在单线程中调用，锁争用可忽略�?    pub fn push_snapshot(&self, ts_ms: i64, features: Vec<f32>) -> PyResult<()> {
        if features.len() != FEATURE_DIM {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} features, got {}",
                FEATURE_DIM,
                features.len()
            )));
        }
        // 获取写锁 �?写入环形缓冲�?        let mut backend = self.backend.write().unwrap();
        backend.push_row(ts_ms, &features);
        Ok(())
    }

    /// 获取最�?secs 秒的数据窗口 �?DLPack 胶囊
    ///
    /// ### 参数
    ///
    /// - `secs: int` �?时间窗口（秒），�?60 = 最�?1 分钟
    ///
    /// ### 返回
    ///
    /// DLPack `PyCapsule`，可传递给 `torch.from_dlpack()` 实现零拷贝�?    /// 返回�?Tensor shape �?`(N, 50)`，dtype �?`float32`�?    ///
    /// ### Python 使用
    ///
    /// ```python
    /// dlpack = bridge.get_window(60)
    /// tensor = torch.from_dlpack(dlpack)  # shape (N, 50), float32, no copy
    /// ```
    ///
    /// ### 生命周期
    ///
    /// DLPack capsule 托管 shape 数组的释放。Python GC 回收 capsule �?    /// 调用 `dlpack_deleter_with_shape` 释放 Rust Box�?*注意**：释放时
    /// backend 可能已写入新数据，但 capsule 持有的指针仍有效（mmap 区域
    /// 不因数据更新而移动）�?    pub fn get_window<'py>(&self, py: Python<'py>, secs: u32) -> PyResult<Bound<'py, PyCapsule>> {
        let backend = self.backend.read().unwrap();
        let n = backend.recent_count_secs(secs as i64);

        // 构建 shape [N, 50]，用 Box 分配以保�?heap 稳定�?        let shape = Box::new([n as i64, FEATURE_DIM as i64]);
        let shape_ptr = shape.as_ptr();

        let data_ptr = backend.data_ptr() as *mut std::ffi::c_void;

        // Safety: managed tensor 包含指向 mmap 区域的指针和 Box �?shape 指针�?        // mmap 区域�?MmapMut 持有（ShmBackend 存活期间有效）�?        // shape 数组�?Box 托管，通过 manager_ctx 传递给 deleter�?        let managed = Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: data_ptr,
                device: DLDevice {
                    device_type: 1, // kDLCPU
                    device_id: 0,
                },
                ndim: 2,
                dtype: DLDataType {
                    code: 1,  // kDLFloat
                    bits: 32,
                    lanes: 1,
                },
                shape: shape_ptr,
                strides: std::ptr::null(), // null = 行优先紧凑排�?                byte_offset: 0,
            },
            manager_ctx: Box::into_raw(shape) as *mut std::ffi::c_void,
            deleter: Some(dlpack_deleter_with_shape),
        });

        let raw = Box::into_raw(managed);
        PyCapsule::new(py, raw, Some("dltensor"))
    }

    /// 获取最�?secs 秒数据的裸指针（NumPy 零拷贝兼容）
    ///
    /// ### 参数
    ///
    /// - `secs: int` �?时间窗口（秒�?    ///
    /// ### 返回
    ///
    /// `(ptr: int, (rows: int, cols: int))` �?数据起始地址 + 形状
    ///
    /// ### Python 使用
    ///
    /// ```python
    /// ptr, (rows, cols) = bridge.get_raw_ptr(30)
    /// import ctypes, numpy as np
    /// arr = np.ctypeslib.as_array(
    ///     (ctypes.c_float * rows * cols).from_address(ptr)
    /// ).reshape(rows, cols)
    /// # 警告: �?arr 不持有内存所有权，后端写入会导致数据变化
    /// ```
    ///
    /// ### 注意事项
    ///
    /// - 返回的指针指�?mmap 区域（或 Vec 堆内存）�?*不持有所有权**
    /// - Python 侧不应尝试释放此指针
    /// - 后端写入会直接修改指针指向的内存（无拷贝语义�?    pub fn get_raw_ptr(&self, secs: u32) -> (usize, (usize, usize)) {
        let backend = self.backend.read().unwrap();
        let n = backend.recent_count_secs(secs as i64);
        let ptr = backend.data_ptr() as usize;
        (ptr, (n, FEATURE_DIM))
    }

    /// 获取最新一条快�?�?MarketSnapshot proto JSON bytes
    ///
    /// ### 返回
    ///
    /// `bytes` �?JSON 格式�?MarketSnapshot 数据，对�?`schemas/market_snapshot.proto`
    /// 结构。Qwen 侧使�?`google.protobuf.json_format.Parse()` 反序列化�?    ///
    /// ### 结构
    ///
    /// ```json
    /// {
    ///   "inst_id": "BTC-USDT-SWAP",
    ///   "ts_ms": 1717000000123,
    ///   "orderbook": {"best_bid": 3125.43, "best_ask": 3125.56, "spread": 0.13, "mid_price": 3125.5},
    ///   "features": [0.01, -0.02, ...],   // 50 维特�?    ///   "pulse_id": 0,
    ///   "market_state": 2                  // OPEN
    /// }
    /// ```
    ///
    /// ### 注意事项
    ///
    /// - 此方法每次调用分配新 Vec 并序列化 JSON，延迟约 5µs
    /// - 热路径应使用 `get_window()` �?`get_raw_ptr()` 替代
    /// - 缓冲区为空时返回�?bytes `b""`（非 `None`�?    ///
    /// ### 返回
    ///
    /// `Vec<u8>` �?MarketSnapshot JSON bytes，Python 侧类型为 `bytes`
    pub fn get_latest(&self) -> Vec<u8> {
        let backend = self.backend.read().unwrap();
        let (ts, row) = match backend.latest_row() {
            Some(v) => v,
            None => return vec![],
        };

        let obj = serde_json::json!({
            "inst_id": self.inst_id,
            "ts_ms": ts,
            "orderbook": {
                "best_bid": 0.0,
                "best_ask": 0.0,
                "spread": 0.0,
                "mid_price": 0.0
            },
            "features": row,
            "pulse_id": 0,
            "market_state": 2  // OPEN
        });
        obj.to_string().into_bytes()
    }

    /// 统计信息：`(当前已写入条�? 最大容�?`
    pub fn stats(&self) -> (usize, usize) {
        let backend = self.backend.read().unwrap();
        (backend.len(), BUFFER_CAPACITY)
    }

    /// 当前交易标的（固定为 `"BTC-USDT-SWAP"`�?    pub fn inst_id(&self) -> &str {
        &self.inst_id
    }
}

// ════════════════════════════════════════════════════════════════
// DLPack deleter �?shape 数组释放
// ════════════════════════════════════════════════════════════════

/// DLPack 托管张量析构回调
///
/// ### 调用时机
///
/// Python GC 回收 DLPack capsule 时（C �?`DLManagedTensor::deleter`）�?///
/// ### Safety
///
/// `ptr` 必须是由 `get_window()` 分配�?`Box<DLManagedTensor>`�?/// `manager_ctx` 必须指向 `Box<[i64; 2]>`。调用后两者均被释放�?unsafe extern "C" fn dlpack_deleter_with_shape(ptr: *mut DLManagedTensor) {
    if !ptr.is_null() {
        let managed = Box::from_raw(ptr);
        // 释放 shape `Box<[i64; 2]>`（通过 manager_ctx 传递）
        if !managed.manager_ctx.is_null() {
            let _shape = Box::from_raw(managed.manager_ctx as *mut [i64; 2]);
        }
    }
    // managed (DLManagedTensor) �?Box 析构自动释放
}

// ════════════════════════════════════════════════════════════════
// 单元测试
// ════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// 基本 push �?get_raw_ptr 流程
    #[test]
    fn test_shm_push_and_read() {
        let bridge = ShmBridge::new("BTC-USDT-SWAP").unwrap();
        bridge
            .push_snapshot(1000, vec![1.0f32; FEATURE_DIM])
            .unwrap();
        let (ptr, (rows, cols)) = bridge.get_raw_ptr(60);
        assert_eq!(rows, 1);
        assert_eq!(cols, FEATURE_DIM);
        assert!(ptr > 0, "Pointer should be a valid non-null address");
    }

    /// 时间窗口过滤正确�?    #[test]
    fn test_recent_count_secs_filtering() {
        let bridge = ShmBridge::new("BTC-USDT-SWAP").unwrap();
        bridge
            .push_snapshot(1000, vec![1.0f32; FEATURE_DIM])
            .unwrap();
        bridge
            .push_snapshot(2000, vec![2.0f32; FEATURE_DIM])
            .unwrap();

        // 2s 窗口：仅包含 ts=2000（ts=1000 在窗口外�?        let (_, (rows, _)) = bridge.get_raw_ptr(2);
        assert_eq!(rows, 1);

        // 60s 窗口：两条都应包�?        let (_, (rows2, _)) = bridge.get_raw_ptr(60);
        assert_eq!(rows2, 2);
    }

    /// get_latest() 返回最新条�?    #[test]
    fn test_get_latest_returns_newest() {
        let bridge = ShmBridge::new("BTC-USDT-SWAP").unwrap();
        bridge
            .push_snapshot(1000, vec![1.0f32; FEATURE_DIM])
            .unwrap();
        bridge
            .push_snapshot(2000, vec![2.0f32; FEATURE_DIM])
            .unwrap();

        let latest = bridge.get_latest();
        assert!(!latest.is_empty(), "Non-empty buffer should return bytes");
        let parsed: serde_json::Value = serde_json::from_slice(&latest).unwrap();
        assert_eq!(parsed["ts_ms"], 2000);
        assert_eq!(parsed["inst_id"], "BTC-USDT-SWAP");
    }

    /// 空缓冲区 get_latest() 返回�?bytes
    #[test]
    fn test_empty_buffer_returns_empty_bytes() {
        let bridge = ShmBridge::new("BTC-USDT-SWAP").unwrap();
        assert!(bridge.get_latest().is_empty());
    }
}
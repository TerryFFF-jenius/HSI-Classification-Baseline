# HongHu 单 seed 诊断资产

本目录只准备新的、可追溯的诊断划分、一次性 PCA 产物、固定 validation batch 和四组运行清单。它不声称复现历史实验，也不读取历史 validation 索引、历史 PCA 或历史 checkpoint。

当前预注册预算为四次独立训练、每次 200 epochs，总计 800 epochs：

| 模型 | 路径 |
| --- | --- |
| `gscvit_tssr` | TSSR |
| `gscvit_dssr` | TSSR -> 光谱状态路由 |
| `gscvit_spatial` | TSSR -> 空间状态路由 |
| `gscvit_dssr_spatial` | TSSR -> 光谱状态路由 -> 空间状态路由 |

`gscvit_dssr` 明确依赖 TSSR 的 `alpha/beta`，不标记为独立于 TSSR 的路径。

## 当前状态

本目录的准备脚本可以生成新 split、PCA 和固定 validation batch；训练和 AutoDL 尚未执行。生成后的资产必须只供四组运行共享：同一 PCA、同一 patch 顺序、同一 split manifest 和同一 validation batch。

PCA 忠实保持当前 `data_loader.py` 语义：整幅 HongHu 场景先用 `PCA(n_components=15, whiten=True, random_state=20261017).fit_transform`，再按标签非零像素的 row-major 顺序建立 patch 索引并切分。没有静默改为 train-only PCA。

## 本地准备命令（仅生成资产）

```cmd
cd /d D:\HSI_Baseline
set KMP_DUPLICATE_LIB_OK=TRUE
C:\Users\PC\Anaconda3\python.exe diagnostics\honghu_single_seed_20261018\prepare_assets.py --project-root D:\HSI_Baseline --output-dir D:\HSI_Baseline\diagnostics\honghu_single_seed_20261018\assets
```

该命令会读取当前 HongHu 数据源并生成 `assets`；不会创建 checkpoint，不会调用模型，不会访问测试标签用于模型选择。

## 未来训练命令（建议，需主线程审核后执行）

训练脚本为诊断专用入口，使用已生成的固定资产，不修改默认训练入口。每个模型单独执行一次，四条命令中的资产目录必须完全相同：

```cmd
cd /d D:\HSI_Baseline
set KMP_DUPLICATE_LIB_OK=TRUE
C:\Users\PC\Anaconda3\python.exe diagnostics\honghu_single_seed_20261018\diagnostic_train.py --project-root D:\HSI_Baseline --assets-dir D:\HSI_Baseline\diagnostics\honghu_single_seed_20261018\assets --model-name gscvit_tssr --output-dir D:\HSI_Baseline\diagnostics\honghu_single_seed_20261018\runs\gscvit_tssr
```

其余三组只替换 `--model-name` 和输出目录为 `gscvit_dssr`、`gscvit_spatial`、`gscvit_dssr_spatial`。这些命令本轮不执行。

## 未来特征诊断

训练完成并由主线程审核 checkpoint 后，使用 `feature_diagnosis.py` 对四个 checkpoint 读取同一个 `validation_batch.npz`。脚本只做固定 batch 的前向和统计，不使用 test，不训练，不调用 `optimizer.step()`。它记录 route feature、alpha、beta、状态输出、permutation/inverse，以及 Rs/Rp 的 RMS 比和余弦相似度；无法取得的中间量必须写为缺失原因，不允许伪造。

## 决策边界

本目录只支持单 seed 筛查。只有四组 split、PCA、配置和 checkpoint 规则一致，且空间单路增量为正、串联空间条件增量为零或负、选定 checkpoint 附近至少三个连续 validation 记录保持该方向，并且 residual 诊断支持相应机制时，才可提出一次 `spatial_state_strength=0.25` 的后续筛选。不得在本目录增加 seed、扫强度、改损失/学习率、改归一化或加入 State Fusion。

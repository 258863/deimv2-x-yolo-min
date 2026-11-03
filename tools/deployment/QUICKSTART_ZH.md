# DEIMv2 转换与推理简明流程

适用范围：本仓库 DEIMv2-X，固定输入 640×640，batch=1。

## 给模型起名称

模型名称与类别名称分开保存。例如，模型叫“电容检测模型_v1”，其中检测的类别叫 `capacitor`。

在训练配置中设置：

```yaml
model_name: 电容检测模型_v1
```

也可在原训练命令后添加 `--model-name "电容检测模型_v1"`，命令行优先。名称不改变网络结构、类别或权重文件名。

checkpoint 顶层保存 `model_name`。导出会将它写入 ONNX metadata；布局修正会保留；构建会继续写入 `model.json`。恢复训练未指定新名称时，沿用 checkpoint 名称。旧权重没有名称时，不自动猜测。

这是模型的显示信息，不会自动注册到业务软件，也不是 SPVS 文件封装。

## 1. 转换顺序

```text
FP32 .pth + 模型代码
  → export：导出 FP32 ONNX
  → patch：等价修正三尺度特征布局，仍为 FP32
  → build：生成 TRT 混合精度 engine
  → 验证预测结果
  → 接入 EXE
```

不需要先生成 FP16 `.pth`。在 `build` 阶段启用 FP16，并让敏感层保留 FP32。当前 engine 的输入、输出仍是 FP32。此流程不使用 INT8。

## 2. 执行转换

先安装训练依赖、部署依赖及与目标环境匹配的 TensorRT，见 [README](README.md#环境)。以下命令在仓库根目录执行。替换数据和权重路径。输出文件或目录必须不存在。

### 导出 ONNX

```powershell
python tools/deployment/deploy.py export --config configs/deimv2/deimv2_dinov3_x_yolo.yml --checkpoint D:/run/best.pth --device cuda:0 --output artifacts/model_raw.onnx
```

默认使用新版训练代码保存的 checkpoint，自动读取其中的 `names` 和类别数量。无需填写 `--names` 或 `--data`，也不需要训练图片。工具优先使用 checkpoint 中的 EMA 权重。

导出会将类别信息写入 ONNX metadata。`patch` 保留这些信息，`build` 自动读取并写入 `model.json`。

仅兼容旧模型时：如果旧 checkpoint 没有 `names`，可补填 `--names` 或 `--data`，两者互斥；都不填则沿用配置中的有效 `yolo_data`。旧 ONNX 没有名称时，构建阶段需补填 `--names`。

### 训练时保存类别信息

YOLO 数据集的 `data.yaml` 中设置 `names`。训练会按类别编号顺序读取名称，并在所有通过 solver 保存的 checkpoint 中记录：

```python
{"model": ..., "ema": ..., "names": ["capacitor"], "num_classes": 1}
```

名称保存在 checkpoint 顶层，不在参数张量中。最佳权重和含 EMA 的 checkpoint 使用同一保存逻辑。旧权重不会自动补写。

新 checkpoint 已有名称时，`export` 可以同时省略 `--names` 和 `--data`。手动参数与已有名称或顺序冲突时会报错。恢复训练时也检查数据集名称顺序；微调到新数据集使用新数据集的名称。

```text
训练 data.yaml → checkpoint names → ONNX metadata → model.json
```

engine 本身不提供这些名称，EXE 应读取配套 `model.json`。

### 修正布局

```powershell
python tools/deployment/deploy.py patch --onnx artifacts/model_raw.onnx --output artifacts/model_fixed.onnx
```

这一步把“整体 reshape、transpose 后切片”改为“先按尺度切片，再分别 reshape、transpose”。权重不变。工具会检查结构，并对比 ONNX Runtime 的改写前后输出。

默认使用随机输入检查。可添加 `--inputs artifacts/normalized_inputs.npz`，使用真实图片预处理后的输入检查。

### 构建混合精度 TRT

如果已有经过检查的 `model_fixed.onnx`，直接执行这一步：

```powershell
python tools/deployment/deploy.py build --onnx artifacts/model_fixed.onnx --output artifacts/trt_fp16mix --precision fp16mix
```

新 ONNX 会自动提供类别名称。旧 ONNX 可添加 `--names capacitor`（单类电容）或 `--names hd,lx,sxbcj`（三类 DIP）。类别数量和顺序必须与模型一致；改名称不会改变模型能力。

输出文件：

```text
artifacts/trt_fp16mix/
├── model.engine                  推理引擎
├── model.json                    类别、输入尺寸、环境及哈希
└── fp32_protected_layers.json     FP32保护层名单
```

应在部署机器上构建。更换 GPU 或 TensorRT 环境后，应重新构建并验证。

### engine 与 model.json 分别保存什么

当前工具不会把模型显示名称和类别名称列表写入 TensorRT engine。ONNX 中的自定义元信息由构建工具读取，再保存到配套 `model.json`。

| 信息 | model.engine | model.json |
|---|---|---|
| 模型显示名称，如“电容检测模型_v1” | 不保存 | 上游有名称时保存为 `model_name` |
| 类别名称，如 `capacitor` | 不保存 | 保存为有序列表 `names` |
| 类别数量 | 可从分类输出 `[1,300,C]` 的 C 得到 | 可从 `names` 长度得到 |
| 张量名称，如 `images`、`pred_logits` | 保存，用于绑定输入输出 | 不等于模型名称或类别名称 |
| 构建环境与校验信息 | 具有运行环境兼容要求 | 记录 GPU、TensorRT 版本及文件哈希 |

EXE 使用 `model.engine` 执行推理，读取同一输出目录的 `model.json` 获取名称及类别映射。例如，`names[0]` 就是类别0的名称。类别顺序不能随意调整。单独传 engine 不会带上这些显示名称。

如需只传一个文件，可以另行将 engine 和 JSON 封装成模型包。当前构建工具输出的是分开的文件，不会自动生成 SPVS。

## 3. 验证 engine

```powershell
python tools/deployment/verify_engine.py --onnx artifacts/model_fixed.onnx --engine artifacts/trt_fp16mix/model.engine --inputs artifacts/normalized_inputs.npz --output artifacts/trt_fp16mix/check.json
```

NPZ 的每个条目是一张图片预处理后的 FP32 张量，形状为 `[1,3,640,640]`。该脚本比较 ONNX 与 TRT 的输出，并记录耗时和类别集合差异。它不读取原始图片，也不计算有人工真值的检测精度。

还需用固定的标注测试集检查框位置、漏检和误检。构建成功不等于精度验收通过。已知三类模型重建差异见 [远端验证记录](REMOTE_VALIDATION.md)。

## 4. EXE 推理接口

EXE 可以通过标准 TensorRT 接口加载 engine，但必须实现 DEIMv2 的前后处理。

| 阶段 | 操作 |
|---|---|
| 前处理 | BGR转RGB；按比例缩放，居中填充到640×640，填充值114；像素除255；减均值 `[0.485,0.456,0.406]`，除标准差 `[0.229,0.224,0.225]`；转换为NCHW |
| 输入 | `images`，FP32，`[1,3,640,640]` |
| 输出 | `pred_logits`，`[1,300,C]`；`pred_boxes`，`[1,300,4]`；C为类别数 |
| 分类解码 | 对 logits 做 sigmoid，将候选框×类别分数展开，选 top300，再按置信度过滤 |
| 框解码 | 将归一化中心点x、中心点y、宽、高转为角点坐标，乘640；减去padding，按实际缩放后宽高与原图的比例还原；裁剪越界框并删除无效框 |
| NMS | 当前验证流程不额外使用NMS |

参考预处理使用 PIL bilinear 插值，缩放宽高按 `round` 取整，左上padding取剩余尺寸的一半向下取整。EXE应对照样例验证插值、取整和坐标还原，避免直接照搬YOLO处理代码。

加载一次 engine，复用执行上下文、CUDA stream 和显存缓冲区。将推理放在后台线程，避免阻塞界面。并发调用不能同时复用同一个执行上下文和缓冲区。

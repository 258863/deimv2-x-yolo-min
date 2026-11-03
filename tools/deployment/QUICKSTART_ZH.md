# DEIMv2 转换与推理简明流程

适用范围：本仓库 DEIMv2-X，固定输入 640×640，batch=1。

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
python tools/deployment/deploy.py export --config configs/deimv2/deimv2_dinov3_x_yolo.yml --data D:/dataset/data.yaml --checkpoint D:/run/best.pth --device cuda:0 --output artifacts/model_raw.onnx
```

配置与数据文件必须对应这份权重。工具优先使用 checkpoint 中的 EMA 权重。

### 修正布局

```powershell
python tools/deployment/deploy.py patch --onnx artifacts/model_raw.onnx --output artifacts/model_fixed.onnx
```

这一步把“整体 reshape、transpose 后切片”改为“先按尺度切片，再分别 reshape、transpose”。权重不变。工具会检查结构，并对比 ONNX Runtime 的改写前后输出。

默认使用随机输入检查。可添加 `--inputs artifacts/normalized_inputs.npz`，使用真实图片预处理后的输入检查。

### 构建混合精度 TRT

如果已有经过检查的 `model_fixed.onnx`，直接执行这一步：

```powershell
python tools/deployment/deploy.py build --onnx artifacts/model_fixed.onnx --output artifacts/trt_fp16mix --precision fp16mix --names capacitor
```

`capacitor` 表示类别0的名称，仅适用于单类电容模型。三类 DIP 模型应填写 `--names hd,lx,sxbcj`。类别数量和顺序必须与模型一致；改名称不会改变模型能力。

输出文件：

```text
artifacts/trt_fp16mix/
├── model.engine                  推理引擎
├── model.json                    类别、输入尺寸、环境及哈希
└── fp32_protected_layers.json     FP32保护层名单
```

应在部署机器上构建。更换 GPU 或 TensorRT 环境后，应重新构建并验证。

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

## 5. 交付内容

交给 EXE 开发人员：

- `model.engine` 和 `model.json`。
- 本文及推理参考代码 [verify_engine.py](verify_engine.py)。该脚本是张量推理验证示例，不是完整的图片推理接口。
- 样例图片及参考框、分数、类别。
- 如需换机重建，再提供 `model_fixed.onnx` 和构建工具。

`type3` 是业务软件的调用入口，需要由该软件接入 DEIMv2 处理逻辑。当前工具不会注册 `type3`，也不会生成 `.spvs`。若软件只接受 SPVS，需按其格式和推理接口另行封装，不能直接改 engine 的文件后缀。

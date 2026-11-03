# DEIMv2 TensorRT 部署：布局修正与混合精度

快速上手：[转换与推理简明流程](QUICKSTART_ZH.md)。包含转换命令、精度设置、验证步骤及 EXE 交付要求。

这是已验证实验方法的可移植工具版：FP16混合精度（敏感层FP32），以及严格FP32对照；不是INT8量化，不需要校准集。默认仅支持X模型、batch=1、640×640、8个注意力头、每头32通道、三尺度6400/1600/400位置。创新模块配置未验证。

## 环境

在仓库根目录运行。Python建议3.11；安装训练依赖及匹配驱动的CUDA PyTorch/torchvision，然后：

```powershell
python -m pip install -r tools/deployment/requirements.txt
python -m pip install tensorrt-cu12==10.13.3.9
```

第二条只需在构建/验证engine的GPU环境执行，Python导入仍为`import tensorrt`。不要为了部署在已有共享环境中盲目降级torch。已验证实验环境为Windows、RTX3090 Ti、torch2.7.0+cu126、TensorRT10.13.3.9。engine与硬件/运行时绑定，换机重建并验收。工具不会自动安装依赖、训练或发布。

## 1. 导出原始ONNX

```powershell
python tools/deployment/deploy.py export --config configs/deimv2/deimv2_dinov3_x_yolo.yml --checkpoint D:/run/final_epoch.pth --device cuda:0 --output artifacts/model_raw.onnx
```

自动选择checkpoint的`ema.module`（存在时）或`model`。默认从新版训练checkpoint自动读取`names`/`num_classes`，无需填写`--names`或`--data`。只有旧checkpoint缺少名称时，才需用`--names`或`--data`补充（两者互斥），或使用配置中的有效数据路径。导出会将名称和类别数写入ONNX metadata，`patch`会保留，`build`会自动读取。显式名称与已有元信息冲突时拒绝。严格匹配模型参数；仅接受可信checkpoint（torch.load使用weights_only=False）。不调用deploy()结构转换，导出opset17、固定shape、FP32原始pred_logits/pred_boxes。输出已存在时拒绝覆盖。

## 2. 等价修正特征布局

```powershell
python tools/deployment/deploy.py patch --onnx artifacts/model_raw.onnx --output artifacts/model_fixed.onnx
```

原路径：memory[1,8400,256] → reshape[1,8400,8,32] → transpose[1,8,32,8400] → 三尺度slice。
新路径：先沿memory轴1拆成6400/1600/400，再分别reshape和transpose。权重、数学含义、原有两个输出均不改变。

工具会核对节点连接、固定输入、实际中间形状、切片范围与轴。模型不匹配或已修正会拒绝；不会只凭文件名猜结构。当前ONNX导出存在动态shape算术，因此检查会在CPU运行一次中间输出，需要足够内存。

随后ONNX Runtime对比原图与修正图输出，默认使用固定seed42随机张量；建议提供真实图像预处理后的NPZ：

```powershell
python tools/deployment/deploy.py patch --onnx artifacts/model_raw.onnx --output artifacts/model_fixed.onnx --inputs artifacts/normalized_inputs.npz
```

NPZ每个key是一张图的float32 `[1,3,640,640]` 输入。RGB、PIL bilinear resize、居中LetterBox填充114、除255、ImageNet均值[.485,.456,.406]/标准差[.229,.224,.225]。只改变图布局，原始query顺序可直接比较；结果保存为`model_fixed.verification.json`。随机/少量图等价不代替完整业务集验收。

## 3. 构建engine

```powershell
python tools/deployment/deploy.py build --onnx artifacts/model_fixed.onnx --output artifacts/trt_fp16mix --precision fp16mix --device 0
python tools/deployment/deploy.py build --onnx artifacts/model_fixed.onnx --output artifacts/trt_fp32 --precision fp32 --device 0
```

类别名称必须与当前模型类别数量、顺序一致；六类模型请填写六类名称。输出目录必须不存在。每次产生`model.engine`、`model.json`、`fp32_protected_layers.json`，记录输入shape、类别、GPU、TensorRT版本、ONNX/engine SHA256、构建时间及未验收状态。

FP16mix：workspace4GiB、FP16、TF32、OBEY_PRECISION_CONSTRAINTS；Reduce/Normalization/Softmax和名称含`/Pow`的ElementWise使用FP32计算与浮点输出。原三类模型的保护名单共121项，不对其它模型硬编码此数量。FP32：关闭FP16/INT8/TF32。两种方式都保持FP32输入输出，不设置动态profile。

## 4. 验收

```powershell
python tools/deployment/verify_engine.py --onnx artifacts/model_fixed.onnx --engine artifacts/trt_fp16mix/model.engine --inputs artifacts/normalized_inputs.npz --output artifacts/trt_fp16mix/check.json
```

使用相同输入、ORT CPU参考、TRT同步执行。按sigmoid后query/class展平top300，记录conf≥0.5类别集合、top10分数差、纯前向耗时。query排序可能不同，因此不把原始张量逐元素差直接当作模型错误。此检查没有GT，不计算召回、框定位质量或端到端FPS。差异如实记录，不自动声称验收通过或修改model.json状态。

还需用冻结的人工标签数据集按既定业务口径计算逐类召回、检查框位置和误检，并测完整流水线时延。空预测不等于正常；与FP32类别集合一致不意味着所有框和分数完全一致。框还原需按实际round后的宽高缩放率去除居中padding。

## 实验依据（2026-10-29）

同一3090 Ti，640/batch1/conf0.5，图像—类别命中、不要求IoU：

| 实现 | 完整集368项 | 整理集332项 | 同轮流水线ms | FPS |
|---|---:|---:|---:|---:|
|旧FP16mix|287/368 77.99%|280/332 84.34%|16.92|59.11|
|布局修正FP32|289/368 78.53%|282/332 84.94%|29.05|34.43|
|布局修正FP16mix|287/368 77.99%|280/332 84.34%|17.16|58.28|

新FP16与FP32预测类别集合不同的图片从17张减到2张；总召回未提高（追回3项、失去3项）。剩余两项分数约0.4986/0.4851；没有事后降低阈值。原始FP32 TRT直接构建仅52.17%，布局等价改写后恢复；此为该模型/版本的优化路径一致性问题，不是FP32正常精度损失，也未定位厂商内部具体指令。

流水线含读图、预处理、H2D、推理、top300、D2H，不含绘制/保存/NMS/Web。332图是事后整理子集，包含3个未专家认证的框草案，不是新独立测试集。上述是原实验脚本结果；本仓库工具的每次新导出/新构建都需要重新验收。

原始三类ONNX SHA256：db8272dbd1550542b0956678ab18d266bdcaeed4ec85ba55947c1af018fd7301。
已验证布局修正ONNX SHA256：bc85972a6aa7a64f4804c4759eabc9b3951188b375210f63e0fd62b5faea1ea5。
新FP16 engine SHA256：6085ee1a0cacf3499d07e936d2b557999ab99e281f489ff7a8432430c3f0bf1a。

## 本次仓库工具验证

- 15个unittest测试通过（原12项与部署输入/结构拒绝检查3项）。
- 使用原始三类ONNX及4张已保存的真实预处理输入运行`patch`成功；CPU ONNX Runtime改写前后等价检查通过。
- 工具生成的ONNX SHA256为`bc85972a6aa7a64f4804c4759eabc9b3951188b375210f63e0fd62b5faea1ea5`，与原3090 Ti实验的修正ONNX逐字节一致。
- CLI帮助、Python编译、git diff --check通过。
- 后续远端已使用本CLI构建FP16 engine并完成387张评估：286/368，较原实验287/368少1项；仅0243的lx分数从0.502075变为0.499390。用户接受该结果并授权发布工具，详见[远端验收记录](REMOTE_VALIDATION.md)。本次未重新验证checkpoint导出及严格FP32 CLI构建。

仓库不携带ONNX、engine、私有图片或checkpoint；`.gitignore`排除这些部署产物。此工具不自动替换DIP Web默认模型。

# 检测 + 分类多任务训练（共用一个骨干）

```
图片 640×640 → DINOv3STAs（共用）→ 3 层特征 ─┬→ HybridEncoder → DEIMTransformer → 检测框
                                            └→ 平均池化 + LayerNorm + Linear  → 整图分类
```

- 检测部分与 `deimv2_dinov3_x_yolo.yml` 完全一样；COCO 的 `model.pth` 照常用 `-t` 加载，只有分类头 `image_head.*` 从头训练。
- 总损失 = 检测损失 + `image_cls_weight` × 交叉熵（默认 1.0，标签平滑 0.05，与 rcclz 分类实验一致）。
- 分类头与 `classification/model.py` 相同：3 层特征各自平均池化后拼接。

## 数据

在 YOLO 的 `data.yaml` 里加两项：

```yaml
path: .
train: images/train
val: images/val
names: [缺陷]                                   # 检测类别（框）
image_names: [OK, 少锡膏偏移, 立碑浮高侧立, 缺件, 翻帖]   # 整图分类类别
image_labels: image_labels.csv                  # 或 folder
```

`image_labels` 两种写法：

| 写法 | 含义 |
|---|---|
| `image_labels.csv` | 两列 `image,label`。`image` 可写相对 `path` 的路径、文件名或不带后缀的文件名；`label` 写类别名或序号。也可按划分分开：`{train: train.csv, val: val.csv}` |
| `folder` | 图片的上一级文件夹名就是类别：`images/train/缺件/0001.jpg`（标注在 `labels/train/缺件/0001.txt`） |

- CSV 的列名也可以是 `file` / `file_name` / `path`（图片）和 `class_name` / `class_id`（类别），rcclz 的 `manifest.csv` 可以直接用。
- 没有分类标签的图片只参与检测（训练开始时会提示有多少张）。
- 没有框标注的图片默认按“没有目标”参与检测。**故意不画框的类别**写进 `det_ignore_classes`：

```yaml
det_ignore_classes: [3]     # 或写类别名：[少锡膏偏移]
```

  这些图只算分类损失：检测损失里把它们整张去掉（不当背景），验证时也不计入 mAP。拼图增强（Mosaic / MixUp）只要拼进了这种图，那张拼图的检测损失同样跳过。

## rcclz_multitask

数据集文件夹里已经放好 `deimv2_multitask.yaml`（`path: .`，整个文件夹拷到哪都不用改），并对过一遍：

| 划分 | OK | 翻帖 | 缺件 | 少锡膏偏移（只分类） | 立碑浮高侧立 |
|---|---|---|---|---|---|
| train | 499 | 95 | 244 | 262 | 37 |
| val | 129 | 24 | 61 | 66 | 9 |

每张检测图正好 1 个框、框类别 = 整图类别；少锡膏偏移全部无框；所有图片都有分类标签。专用配置 `deimv2_dinov3_x_yolo_multitask_rcclz.yml` 带上了类别权重 `[0.456, 2.394, 0.932, 0.868, 6.146]`：

```bash
python train.py -c configs/deimv2/deimv2_dinov3_x_yolo_multitask_rcclz.yml \
  -t ckpts/deimv2_coco/DEIMv2_DINOv3_X_COCO/model.pth --use-amp --seed 42 -d cuda \
  -u yolo_data=/abs/path/rcclz_multitask/deimv2_multitask.yaml
```

## 数据增强与分类标签

检测用的 Mosaic / MixUp / CopyBlend 会把几张图拼在一起，整图类别可能变得不确定：

- 拼进来的图类别都相同：保留这个类别；
- 类别不同：这张增强后的图不算分类损失，只算检测损失。

另外，随机裁剪（RandomIoUCrop）如果把这张图上所有标注框都裁掉了，缺陷可能已经看不到，这张也不算分类损失。

Mosaic 只在第 4–29 轮、MixUp 第 4–29 轮、CopyBlend 第 4–50 轮开启，之后分类损失作用于全部有标签的图。

## 训练

```bash
python train.py -c configs/deimv2/deimv2_dinov3_x_yolo_multitask.yml \
  -t ckpts/deimv2_coco/DEIMv2_DINOv3_X_COCO/model.pth \
  --use-amp --seed 42 -d cuda \
  -u yolo_data=/abs/path/rcclz_multitask/data.yaml
```

可以用 `-u` 调的参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `MultiTaskCriterion.image_cls_weight` | 1.0 | 分类损失权重；检测掉点时调小（0.3–0.5），分类不够时调大 |
| `MultiTaskCriterion.class_weights` | 无 | 每个分类类别一个权重，缓解类别不平衡 |
| `MultiTaskCriterion.label_smoothing` | 0.05 | |
| `DEIMMultiTask.cls_source` | backbone | `encoder`：用编码器（FPN）输出做分类 |
| `DEIMMultiTask.cls_hidden_dim` | 0 | >0 时分类头多一层 MLP |
| `DINOv3STAs.finetune` | True | False：冻结 DINOv3 |

## 验证输出

每轮验证除了检测 mAP，还会打印：

```
Image classification: accuracy 0.9312  macro-F1 0.9105  (1520 labelled images)
  per-class recall: [0.97, 0.88, 0.91, 0.95, 0.86]
```

`log.txt` 里每轮多一个 `test_image_cls`（准确率、macro-F1、各类召回、混淆矩阵）；TensorBoard 里是 `Test/image_cls_accuracy`、`Test/image_cls_macro_f1`。

**最佳模型（best_stg1/2.pth）仍按检测 mAP 选**，分类指标只记录不参与。

## 导出

`tools/deployment/deploy.py export` 可以直接导出多任务模型的**检测部分**（输出仍是 `pred_logits`、`pred_boxes`），后面的 patch / build / verify 不变。分类输出进 ONNX / TensorRT 还没做。

## 测试

```bash
python -m unittest tests/test_multitask.py -v
```

覆盖：CSV / 文件夹两种标签读取、manifest 式列名、`det_ignore_classes`（检测损失去掉这些图、整批都是时损失为 0）、增强时标签合并规则、损失（忽略无标签图、整批无标签）、准确率与混淆矩阵、用真实 X 配置建模型并跑一次前向 + 反向、普通检测权重能加载进多任务模型。

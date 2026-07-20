# 角膜炎图文联合分类首轮实验

这版代码优先解决两件事：

1. 把 `picture` 中 6 类、`1121` 例患者图像与 `text/角膜炎数据完整版信息.xlsx` 中对应病历稳定对齐。
2. 先跑一组本地可验证、服务器也容易复现的多模态基线实验，确认“病历文本是否带来独立增益”。

## 当前实验定义

- 主任务：6 类角膜炎/正常分类
- 样本：仅使用同时具备图像与病历文本的 `1121` 例
- 类别：
  - `A_HSK_EM`：单纯疱疹病毒性角膜炎-上皮/混合型
  - `B_OVK`：其他疱疹病毒性角膜炎
  - `C_BK`：细菌性角膜炎
  - `D_FK`：真菌性角膜炎
  - `E_Normal`：正常
  - `F_NIK`：非感染性角膜炎
- 暂不纳入：`Z棘阿米巴性角膜炎8例`
  - 原因：当前只有 Excel 文本，没有 `picture` 中对应图像，不能直接参与图文主实验

## 四组首轮实验

`run_experiments.py` 默认会跑以下四组：

- `image`：只用三模态图像
- `text`：只用病历文本
- `late`：图像特征和文本特征直接拼接
- `prior`：把文本分支预测概率作为“临床先验”，和图像分支预测概率做对数融合

说明：

- 这版是“先把问题跑通、把文本价值验证清楚”的首轮基线。
- `prior` 虽然不是深度门控网络，但很适合作为“临床先验引导”第一版验证。
- 等你把首轮结果跑回来后，下一步我们再把它升级成真正的深度特征级 `text-guided gating / FiLM / attention` 融合。

## 运行前依赖

至少需要这些 Python 包：

```txt
numpy
pandas
scikit-learn
scipy
Pillow
openpyxl
```

## 推荐运行步骤

先在项目根目录执行清单构建：

```bash
python shiyan/build_manifest.py
```

默认会生成：

- `shiyan/outputs/manifest_6class.csv`
- `shiyan/outputs/manifest_6class_summary.json`

然后跑 5 折交叉验证基线：

```bash
python shiyan/run_experiments.py \
  --manifest shiyan/outputs/manifest_6class.csv \
  --output-dir shiyan/outputs/baseline_results \
  --models image text late prior \
  --cv-folds 5 \
  --prior-weight 0.75 \
  --cache-image-features
```

## 快速联调命令

如果你只想先确认环境和流程没有问题，可以先跑一个小样本版本：

```bash
python shiyan/run_experiments.py \
  --manifest shiyan/outputs/manifest_6class.csv \
  --output-dir shiyan/outputs/debug_results \
  --models image text late prior \
  --cv-folds 2 \
  --max-samples-per-class 6
```

## 输出结果

结果会保存在 `--output-dir` 指定目录，核心文件包括：

- `summary.json`：总结果汇总
- `summary_table.csv`：便于直接看均值/std
- `predictions.csv`：逐样本预测结果
- `image_fold_metrics.json`
- `text_fold_metrics.json`
- `late_fold_metrics.json`
- `prior_fold_metrics.json`

## 你把哪些结果发给我

首轮建议把下面这些结果贴回来：

- `summary_table.csv`
- `summary.json`
- 你关注的 `confusion_matrix`

我会基于这些结果继续帮你：

1. 判断病历文本到底有没有增益
2. 看哪些类别最容易混淆
3. 决定下一步是做深度特征级融合，还是先补做 5 类/层级任务/类别重平衡

## GPU 正式实验

如果服务器装有 `torch` 且可用 CUDA，推荐直接运行：

```bash
python shiyan/run_experiments_torch.py \
  --manifest shiyan/outputs/manifest_6class.csv \
  --output-dir shiyan/outputs/torch_results \
  --models image text late prior \
  --cv-folds 5 \
  --epochs 15 \
  --batch-size 16 \
  --num-workers 4 \
  --image-size 224 \
  --device cuda
```

这版会实际调用 GPU，并为每个模型每一折输出：

- `learning_curve.png`
- `confusion_matrix.png`
- `per_class_f1.png`
- `metrics.json`
- `best_model.pt`

如果想先快速联调 GPU 环境：

```bash
python shiyan/run_experiments_torch.py \
  --manifest shiyan/outputs/manifest_6class.csv \
  --output-dir shiyan/outputs/torch_debug \
  --models late prior \
  --cv-folds 2 \
  --epochs 2 \
  --batch-size 8 \
  --max-samples-per-class 8 \
  --image-size 160 \
  --device cuda
```

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from text_utils import normalize_clinical_text


MODALITIES = ("DLI", "FSI", "SBI")
RESAMPLING_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


@dataclass
class FoldResult:
    model_name: str
    fold_id: int
    accuracy: float
    macro_f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行角膜炎图文多模态分类实验。")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path.cwd() / "shiyan" / "outputs" / "manifest_6class.csv",
        help="build_manifest.py 生成的清单文件。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "shiyan" / "outputs" / "baseline_results",
        help="结果输出目录。",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["image", "text", "late", "prior"],
        choices=["image", "text", "late", "prior"],
        help="要运行的模型列表。",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="分层交叉验证折数。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="仅用于快速验证，限制每类最多保留多少样本。",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=24,
        help="图像缩略图边长，默认 24。",
    )
    parser.add_argument(
        "--text-max-features",
        type=int,
        default=5000,
        help="文本 TF-IDF 最大特征数。",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="raw_text_clean",
        help="Manifest column used for text modeling.",
    )
    parser.add_argument("--text-ngram-min", type=int, default=1, help="Minimum TF-IDF ngram length.")
    parser.add_argument("--text-ngram-max", type=int, default=2, help="Maximum TF-IDF ngram length.")
    parser.add_argument(
        "--prior-weight",
        type=float,
        default=0.75,
        help="文本先验融合权重，prior 模型使用。",
    )
    parser.add_argument(
        "--cache-image-features",
        action="store_true",
        help="将图像特征缓存到 output-dir 中，便于重复实验。",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到 manifest 文件: {path}")
    df = pd.read_csv(path)
    required_columns = {
        "sample_id",
        "label_id",
        "label_name",
        "raw_text",
        "dli_paths",
        "fsi_paths",
        "sbi_paths",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"manifest 缺少必要字段: {sorted(missing)}")
    return df


def subset_manifest(df: pd.DataFrame, max_samples_per_class: int | None, seed: int) -> pd.DataFrame:
    if max_samples_per_class is None:
        return df.reset_index(drop=True)
    sampled = []
    for _, group in df.groupby("label_id", sort=True):
        sampled.append(group.sample(n=min(len(group), max_samples_per_class), random_state=seed))
    subset = pd.concat(sampled, axis=0).sort_values("sample_id").reset_index(drop=True)
    return subset


def load_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def extract_single_image_features(image: Image.Image, image_size: int) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    rgb_mean = rgb.mean(axis=(0, 1))
    rgb_std = rgb.std(axis=(0, 1))

    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    hsv_mean = hsv.mean(axis=(0, 1))
    hsv_std = hsv.std(axis=(0, 1))

    gray = image.convert("L")
    gray_small = gray.resize((image_size, image_size), RESAMPLING_BILINEAR)
    gray_array = np.asarray(gray_small, dtype=np.float32).reshape(-1) / 255.0

    gray_hist, _ = np.histogram(gray_array, bins=16, range=(0.0, 1.0), density=True)

    return np.concatenate([rgb_mean, rgb_std, hsv_mean, hsv_std, gray_hist.astype(np.float32), gray_array], axis=0)


def aggregate_modality_features(path_string: str, image_size: int) -> np.ndarray:
    image_paths = [path for path in str(path_string).split(";") if path]
    features = [extract_single_image_features(load_image(path), image_size=image_size) for path in image_paths]
    return np.mean(features, axis=0).astype(np.float32)


def build_image_features(df: pd.DataFrame, image_size: int, cache_path: Path | None = None) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if list(cached["sample_id"]) == list(df["sample_id"].astype(str).values):
            return cached["features"]

    all_features = []
    for _, row in df.iterrows():
        parts = [
            aggregate_modality_features(row["dli_paths"], image_size=image_size),
            aggregate_modality_features(row["fsi_paths"], image_size=image_size),
            aggregate_modality_features(row["sbi_paths"], image_size=image_size),
        ]
        all_features.append(np.concatenate(parts, axis=0))
    feature_array = np.stack(all_features, axis=0).astype(np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, sample_id=df["sample_id"].astype(str).values, features=feature_array)
    return feature_array


def ensure_text_column(df: pd.DataFrame, text_column: str) -> tuple[pd.DataFrame, str]:
    if text_column in df.columns:
        return df, text_column
    if text_column == "raw_text_clean" and "raw_text" in df.columns:
        resolved = df.copy()
        resolved["raw_text_clean"] = resolved["raw_text"].fillna("").astype(str).map(normalize_clinical_text)
        return resolved, "raw_text_clean"
    raise ValueError(f"manifest ???????? {text_column}")


def build_text_vectorizer(max_features: int, ngram_min: int, ngram_max: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
    )


def make_dense_classifier() -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs",
        max_iter=3000,
        class_weight="balanced",
    )


def make_sparse_classifier() -> LogisticRegression:
    return LogisticRegression(
        solver="saga",
        max_iter=3000,
        class_weight="balanced",
    )


def fit_image_model(x_train: np.ndarray, y_train: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    clf = make_dense_classifier()
    clf.fit(x_train_scaled, y_train)
    return scaler, clf


def predict_image_model(
    scaler: StandardScaler, clf: LogisticRegression, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x_test_scaled = scaler.transform(x_test)
    probs = clf.predict_proba(x_test_scaled)
    preds = probs.argmax(axis=1)
    return preds, probs


def fit_text_model(
    text_train: Sequence[str],
    y_train: np.ndarray,
    text_max_features: int,
    text_ngram_min: int,
    text_ngram_max: int,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = build_text_vectorizer(
        max_features=text_max_features,
        ngram_min=text_ngram_min,
        ngram_max=text_ngram_max,
    )
    x_train_text = vectorizer.fit_transform(text_train)
    clf = make_sparse_classifier()
    clf.fit(x_train_text, y_train)
    return vectorizer, clf


def predict_text_model(
    vectorizer: TfidfVectorizer, clf: LogisticRegression, text_test: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    x_test_text = vectorizer.transform(text_test)
    probs = clf.predict_proba(x_test_text)
    preds = probs.argmax(axis=1)
    return preds, probs


def fit_late_fusion_model(
    x_img_train: np.ndarray,
    text_train: Sequence[str],
    y_train: np.ndarray,
    text_max_features: int,
    text_ngram_min: int,
    text_ngram_max: int,
) -> tuple[StandardScaler, TfidfVectorizer, LogisticRegression]:
    scaler = StandardScaler()
    x_img_scaled = scaler.fit_transform(x_img_train)
    vectorizer = build_text_vectorizer(
        max_features=text_max_features,
        ngram_min=text_ngram_min,
        ngram_max=text_ngram_max,
    )
    x_text = vectorizer.fit_transform(text_train)
    x_joint = sp.hstack([sp.csr_matrix(x_img_scaled), x_text], format="csr")
    clf = make_sparse_classifier()
    clf.fit(x_joint, y_train)
    return scaler, vectorizer, clf


def predict_late_fusion_model(
    scaler: StandardScaler,
    vectorizer: TfidfVectorizer,
    clf: LogisticRegression,
    x_img_test: np.ndarray,
    text_test: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_img_scaled = scaler.transform(x_img_test)
    x_text = vectorizer.transform(text_test)
    x_joint = sp.hstack([sp.csr_matrix(x_img_scaled), x_text], format="csr")
    probs = clf.predict_proba(x_joint)
    preds = probs.argmax(axis=1)
    return preds, probs


def predict_prior_guided(
    image_probs: np.ndarray,
    text_probs: np.ndarray,
    prior_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-8
    fused_log_probs = np.log(image_probs + eps) + prior_weight * np.log(text_probs + eps)
    fused_probs = np.exp(fused_log_probs - fused_log_probs.max(axis=1, keepdims=True))
    fused_probs = fused_probs / fused_probs.sum(axis=1, keepdims=True)
    preds = fused_probs.argmax(axis=1)
    return preds, fused_probs


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int],
    label_names: Sequence[str],
) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fold_predictions_to_frame(
    sample_ids: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    label_names: Sequence[str],
    model_name: str,
    fold_id: int,
) -> pd.DataFrame:
    data = {
        "sample_id": list(sample_ids),
        "model_name": [model_name] * len(sample_ids),
        "fold_id": [fold_id] * len(sample_ids),
        "y_true": y_true,
        "y_pred": y_pred,
    }
    for class_id, label_name in enumerate(label_names):
        data[f"prob_{class_id}_{label_name}"] = probs[:, class_id]
    return pd.DataFrame(data)


def summarize_results(
    per_model_metrics: Dict[str, List[Dict[str, object]]],
    label_names: Sequence[str],
) -> dict:
    summary = {}
    for model_name, fold_metrics in per_model_metrics.items():
        accuracies = [item["accuracy"] for item in fold_metrics]
        macro_f1s = [item["macro_f1"] for item in fold_metrics]
        summary[model_name] = {
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_macro_f1": float(np.mean(macro_f1s)),
            "std_macro_f1": float(np.std(macro_f1s)),
            "num_folds": len(fold_metrics),
            "labels": list(label_names),
        }
    return summary


def main() -> None:
    args = parse_args()
    df = read_manifest(args.manifest.resolve())
    df, text_column = ensure_text_column(df, args.text_column)
    df = subset_manifest(df, max_samples_per_class=args.max_samples_per_class, seed=args.seed)

    label_map = (
        df[["label_id", "label_name"]].drop_duplicates().sort_values("label_id").reset_index(drop=True)
    )
    labels = label_map["label_id"].astype(int).tolist()
    label_names = label_map["label_name"].astype(str).tolist()

    cache_path = None
    if args.cache_image_features:
        cache_name = f"image_features_n{len(df)}_s{args.image_size}.npz"
        cache_path = args.output_dir.resolve() / "cache" / cache_name
    x_img = build_image_features(df, image_size=args.image_size, cache_path=cache_path)
    texts = df[text_column].fillna("").astype(str).tolist()
    y = df["label_id"].astype(int).to_numpy()
    sample_ids = df["sample_id"].astype(str).tolist()

    splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    per_model_metrics: Dict[str, List[Dict[str, object]]] = {name: [] for name in args.models}
    prediction_frames: List[pd.DataFrame] = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(x_img, y), start=1):
        x_img_train, x_img_test = x_img[train_idx], x_img[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        text_train = [texts[i] for i in train_idx]
        text_test = [texts[i] for i in test_idx]
        sample_ids_test = [sample_ids[i] for i in test_idx]

        image_scaler = image_clf = None
        text_vectorizer = text_clf = None

        if any(model in args.models for model in ("image", "prior")):
            image_scaler, image_clf = fit_image_model(x_img_train, y_train)
            image_preds, image_probs = predict_image_model(image_scaler, image_clf, x_img_test)
            if "image" in args.models:
                metrics = compute_metrics(y_test, image_preds, labels=labels, label_names=label_names)
                metrics["fold_id"] = fold_id
                per_model_metrics["image"].append(metrics)
                prediction_frames.append(
                    fold_predictions_to_frame(
                        sample_ids=sample_ids_test,
                        y_true=y_test,
                        y_pred=image_preds,
                        probs=image_probs,
                        label_names=label_names,
                        model_name="image",
                        fold_id=fold_id,
                    )
                )

        if any(model in args.models for model in ("text", "prior")):
            text_vectorizer, text_clf = fit_text_model(
                text_train,
                y_train,
                text_max_features=args.text_max_features,
                text_ngram_min=args.text_ngram_min,
                text_ngram_max=args.text_ngram_max,
            )
            text_preds, text_probs = predict_text_model(text_vectorizer, text_clf, text_test)
            if "text" in args.models:
                metrics = compute_metrics(y_test, text_preds, labels=labels, label_names=label_names)
                metrics["fold_id"] = fold_id
                per_model_metrics["text"].append(metrics)
                prediction_frames.append(
                    fold_predictions_to_frame(
                        sample_ids=sample_ids_test,
                        y_true=y_test,
                        y_pred=text_preds,
                        probs=text_probs,
                        label_names=label_names,
                        model_name="text",
                        fold_id=fold_id,
                    )
                )

        if "late" in args.models:
            late_scaler, late_vectorizer, late_clf = fit_late_fusion_model(
                x_img_train=x_img_train,
                text_train=text_train,
                y_train=y_train,
                text_max_features=args.text_max_features,
                text_ngram_min=args.text_ngram_min,
                text_ngram_max=args.text_ngram_max,
            )
            late_preds, late_probs = predict_late_fusion_model(
                scaler=late_scaler,
                vectorizer=late_vectorizer,
                clf=late_clf,
                x_img_test=x_img_test,
                text_test=text_test,
            )
            metrics = compute_metrics(y_test, late_preds, labels=labels, label_names=label_names)
            metrics["fold_id"] = fold_id
            per_model_metrics["late"].append(metrics)
            prediction_frames.append(
                fold_predictions_to_frame(
                    sample_ids=sample_ids_test,
                    y_true=y_test,
                    y_pred=late_preds,
                    probs=late_probs,
                    label_names=label_names,
                    model_name="late",
                    fold_id=fold_id,
                )
            )

        if "prior" in args.models:
            if image_scaler is None or image_clf is None or text_vectorizer is None or text_clf is None:
                raise RuntimeError("prior 模型需要先完成 image 与 text 分支训练。")
            _, image_probs = predict_image_model(image_scaler, image_clf, x_img_test)
            _, text_probs = predict_text_model(text_vectorizer, text_clf, text_test)
            prior_preds, prior_probs = predict_prior_guided(
                image_probs=image_probs,
                text_probs=text_probs,
                prior_weight=args.prior_weight,
            )
            metrics = compute_metrics(y_test, prior_preds, labels=labels, label_names=label_names)
            metrics["fold_id"] = fold_id
            metrics["prior_weight"] = args.prior_weight
            per_model_metrics["prior"].append(metrics)
            prediction_frames.append(
                fold_predictions_to_frame(
                    sample_ids=sample_ids_test,
                    y_true=y_test,
                    y_pred=prior_preds,
                    probs=prior_probs,
                    label_names=label_names,
                    model_name="prior",
                    fold_id=fold_id,
                )
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_results(per_model_metrics=per_model_metrics, label_names=label_names)
    summary["config"] = {
        "manifest": str(args.manifest.resolve()),
        "num_samples": int(len(df)),
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "models": args.models,
        "image_size": args.image_size,
        "text_max_features": args.text_max_features,
        "text_column": text_column,
        "text_ngram_min": args.text_ngram_min,
        "text_ngram_max": args.text_ngram_max,
        "prior_weight": args.prior_weight,
        "max_samples_per_class": args.max_samples_per_class,
    }

    save_json(output_dir / "summary.json", summary)

    for model_name, fold_metrics in per_model_metrics.items():
        save_json(output_dir / f"{model_name}_fold_metrics.json", {"folds": fold_metrics})

    if prediction_frames:
        predictions_df = pd.concat(prediction_frames, axis=0).reset_index(drop=True)
        predictions_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for model_name, model_summary in summary.items():
        if model_name == "config":
            continue
        summary_rows.append(
            {
                "model_name": model_name,
                "mean_accuracy": model_summary["mean_accuracy"],
                "std_accuracy": model_summary["std_accuracy"],
                "mean_macro_f1": model_summary["mean_macro_f1"],
                "std_macro_f1": model_summary["std_macro_f1"],
            }
        )
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(output_dir / "summary_table.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

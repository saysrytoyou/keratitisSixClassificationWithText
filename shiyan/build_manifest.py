from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from text_utils import normalize_clinical_text


TEXT_COLUMN = "原始病历（主诉+现病史+既往史）"


@dataclass(frozen=True)
class ClassSpec:
    image_dir: str
    sheet_name: str
    label_id: int
    label_name: str


CLASS_SPECS: List[ClassSpec] = [
    ClassSpec("A_HSK_EM", "A上皮型合并混合型单纯疱疹病毒性角膜炎169", 0, "单纯疱疹病毒性角膜炎-上皮/混合型"),
    ClassSpec("B_OVK", "B 其他疱疹病毒性角膜炎254", 1, "其他疱疹病毒性角膜炎"),
    ClassSpec("C_BK", "C细菌性角膜炎200", 2, "细菌性角膜炎"),
    ClassSpec("D_FK", "D真菌性角膜炎130", 3, "真菌性角膜炎"),
    ClassSpec("E_Normal", "K正常组193", 4, "正常"),
    ClassSpec("F_NIK", "E非感染性角膜炎175", 5, "非感染性角膜炎"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建角膜炎 6 类图文配对样本清单。")
    parser.add_argument(
        "--picture-dir",
        type=Path,
        default=Path.cwd() / "picture",
        help="图像根目录，默认读取当前工作目录下的 picture。",
    )
    parser.add_argument(
        "--text-xlsx",
        type=Path,
        default=None,
        help="病历 Excel 路径，默认自动读取当前工作目录下 text 中唯一的 xlsx。",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path.cwd() / "shiyan" / "outputs" / "manifest_6class.csv",
        help="输出 CSV 路径。",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path.cwd() / "shiyan" / "outputs" / "manifest_6class_summary.json",
        help="输出统计摘要 JSON 路径。",
    )
    return parser.parse_args()


def resolve_text_xlsx(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"未找到 Excel 文件: {explicit_path}")
        return explicit_path

    text_dir = Path.cwd() / "text"
    files = sorted(text_dir.glob("*.xlsx"))
    if len(files) != 1:
        raise FileNotFoundError(f"text 目录下应当恰好存在 1 个 xlsx，当前找到 {len(files)} 个。")
    return files[0]


def collect_modality_paths(patient_dir: Path) -> Dict[str, List[str]]:
    files = sorted(patient_dir.glob("*.jpg"))
    grouped = {"DLI": [], "FSI": [], "SBI": []}
    for file_path in files:
        name = file_path.name.upper()
        for modality in grouped:
            # Allow noisy filenames such as "_ DLI_" or "__DLI__" without requiring manual renaming.
            if re.search(rf"[_\s]+{modality}[_\s]+", name):
                grouped[modality].append(str(file_path.resolve()))
                break
    for modality, paths in grouped.items():
        if not paths:
            raise ValueError(f"{patient_dir} 缺少 {modality} 模态图像。")
    return grouped


def build_records(picture_dir: Path, text_xlsx: Path) -> List[dict]:
    xls = pd.ExcelFile(text_xlsx)
    records: List[dict] = []

    for spec in CLASS_SPECS:
        class_dir = picture_dir / spec.image_dir
        if not class_dir.exists():
            raise FileNotFoundError(f"未找到图像类别目录: {class_dir}")

        image_patients = sorted([path for path in class_dir.iterdir() if path.is_dir()])
        sheet_df = pd.read_excel(text_xlsx, sheet_name=spec.sheet_name).fillna("")

        if len(image_patients) != len(sheet_df):
            raise ValueError(
                f"{spec.image_dir} 图像患者数 {len(image_patients)} 与 sheet {spec.sheet_name} 文本数 {len(sheet_df)} 不一致。"
            )

        for patient_dir, (_, row) in zip(image_patients, sheet_df.iterrows()):
            grouped = collect_modality_paths(patient_dir)
            patient_index = int(patient_dir.name.split("_")[-1])
            text_value = str(row.get(TEXT_COLUMN, "")).strip()
            normalized_text = normalize_clinical_text(text_value)

            records.append(
                {
                    "sample_id": patient_dir.name,
                    "patient_index": patient_index,
                    "class_dir": spec.image_dir,
                    "label_id": spec.label_id,
                    "label_name": spec.label_name,
                    "sheet_name": spec.sheet_name,
                    "excel_code": str(row.get("最新编号", "")).strip(),
                    "original_code": str(row.get("原始编号", "")).strip(),
                    "pid": str(row.get("PID", "")).strip(),
                    "patient_name": str(row.get("姓名", "")).strip(),
                    "eye": str(row.get("眼别", "")).strip(),
                    "diagnosis": str(row.get("诊断", "")).strip(),
                    "raw_text": text_value,
                    "raw_text_clean": normalized_text,
                    "text_char_count": len(normalized_text),
                    "dli_paths": ";".join(grouped["DLI"]),
                    "fsi_paths": ";".join(grouped["FSI"]),
                    "sbi_paths": ";".join(grouped["SBI"]),
                    "num_dli": len(grouped["DLI"]),
                    "num_fsi": len(grouped["FSI"]),
                    "num_sbi": len(grouped["SBI"]),
                }
            )
    records.sort(key=lambda item: item["patient_index"])
    return records


def write_outputs(records: List[dict], output_csv: Path, output_summary: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "num_samples": int(len(df)),
        "num_classes": int(df["label_id"].nunique()),
        "class_distribution": {
            name: int(count) for name, count in df["label_name"].value_counts(sort=False).items()
        },
        "modality_image_count_range": {
            "DLI": [int(df["num_dli"].min()), int(df["num_dli"].max())],
            "FSI": [int(df["num_fsi"].min()), int(df["num_fsi"].max())],
            "SBI": [int(df["num_sbi"].min()), int(df["num_sbi"].max())],
        },
        "text_char_count": {
            "min": int(df["text_char_count"].min()),
            "median": float(df["text_char_count"].median()),
            "max": int(df["text_char_count"].max()),
        },
        "patient_index_range": [int(df["patient_index"].min()), int(df["patient_index"].max())],
        "output_csv": str(output_csv.resolve()),
    }
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    picture_dir = args.picture_dir.resolve()
    text_xlsx = resolve_text_xlsx(args.text_xlsx).resolve()
    records = build_records(picture_dir=picture_dir, text_xlsx=text_xlsx)
    write_outputs(records=records, output_csv=args.output_csv.resolve(), output_summary=args.output_summary.resolve())


if __name__ == "__main__":
    main()

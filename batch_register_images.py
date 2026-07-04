"""
バッチ画像位置合わせスクリプト (df データセット汎用版)

フォルダ内の全画像を「条件-セット-連番」規則で pre(連番0)/post(連番1以降)に
自動グルーピングし、デザインナイフ傷(スクラッチ)をランドマークとして
phaseCorrelate初期値 -> findTransformECC (MOTION_EUCLIDEAN) で位置合わせを行う。

傷検出には PC-alignment-anti/registration.py の detect_grooves
(プロファイル最小値検出、移動平均窓で平滑化)の考え方を流用する。

位置合わせ後、pre/post間で共通の有効領域のみをクロップして保存し、
傷付近およびピラー配列領域(任意指定)の NCC・一致率を算出してCSVに記録する。
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import tifffile

logger = logging.getLogger("batch_register")

IMAGE_SUFFIXES = {".tif", ".tiff"}

# 傷(スクラッチ)ランドマーク検出のデフォルト探索帯域。
# registration.py の detect_grooves を参考にした値であり、CLI引数で上書き可能。
DEFAULT_V_COL_RANGE = (0, 300)
DEFAULT_V_ROW_BAND = (500, 1500)
DEFAULT_H_ROW_RANGE = (0, 300)
DEFAULT_H_COL_BAND = (1000, 1900)


# ===================== ファイルグルーピング =====================

@dataclass
class GroupKey:
    condition: str
    set_id: str

    @property
    def label(self) -> str:
        return f"{self.condition}-{self.set_id}"


def parse_filename(stem: str) -> Optional[tuple]:
    """'条件-セット-連番' 形式のファイル名を分解する。連番は末尾の整数。"""
    parts = stem.rsplit("-", 2)
    if len(parts) != 3:
        return None
    condition, set_id, seq_str = parts
    if not re.fullmatch(r"\d+", seq_str):
        return None
    return condition, set_id, int(seq_str)


def group_files(input_dir: Path) -> dict:
    """入力フォルダ内のファイルを (条件, セット) -> {連番: Path} にグルーピング。"""
    groups: dict = {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parsed = parse_filename(path.stem)
        if parsed is None:
            logger.warning("ファイル名規則に一致しないためスキップ: %s", path.name)
            continue
        condition, set_id, seq = parsed
        key = GroupKey(condition, set_id)
        groups.setdefault(key.label, {"key": key, "files": {}})
        groups[key.label]["files"][seq] = path
    return groups


# ===================== 傷(スクラッチ)ランドマーク検出 =====================

def _subpixel_min(profile: np.ndarray, idx: int) -> float:
    if 0 < idx < len(profile) - 1:
        y0, y1, y2 = profile[idx - 1], profile[idx], profile[idx + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            return idx + 0.5 * (y0 - y2) / denom
    return float(idx)


def _prominence(sub_profile: np.ndarray, local_idx: int) -> float:
    """
    プロファイル最小値の「深さ」を、同じ探索範囲内の頑健なばらつき指標(MAD)に対する
    比で評価する。argmin は常にその範囲内で最も低い値を選ぶため、通常の標準偏差や
    生画素の標準偏差を分母にすると、周期的なピラー構造由来の分散混入や、多数サンプル
    からの極値選択(順序統計量)の影響で、真の傷が無くても高いスコアが出やすい。
    中央値絶対偏差(MAD)は少数の極値(=最小点そのもの)に影響されにくく、
    「大多数の点から見て、その谷がどれだけ際立っているか」を頑健に測れる。
    """
    eps = 1e-6
    median = np.median(sub_profile)
    mad = np.median(np.abs(sub_profile - median)) * 1.4826  # 正規分布換算の標準偏差相当
    depth = median - sub_profile[local_idx]
    return float(depth / (mad + eps))


def _find_profile_min(band: np.ndarray, axis: int, smooth_window: int, edge_margin: int) -> tuple:
    """
    band を axis 方向に平均化してプロファイル化し、移動平均で平滑化した後、
    最小値の位置(サブピクセル)とプロミネンスを求める。
    np.convolve の mode='same' は端で暗黙的にゼロ相当のパディングを行うため、
    プロファイル端に平滑化アーティファクト(見かけ上の谷)が生じる。
    detect_grooves (registration.py) 同様、両端 edge_margin 分を探索範囲から除外する。
    """
    profile = np.mean(band, axis=axis)
    smooth = np.convolve(profile, np.ones(smooth_window) / smooth_window, mode="same")

    margin = min(edge_margin, (len(smooth) - 1) // 2)
    lo, hi = margin, len(smooth) - margin
    sub = smooth[lo:hi]
    local_idx = int(np.argmin(sub))
    idx = lo + local_idx

    position = _subpixel_min(smooth, idx)
    prom = _prominence(sub, local_idx)
    return position, prom


def detect_scratch_landmark(
    img: np.ndarray,
    v_col_range=DEFAULT_V_COL_RANGE,
    v_row_band=DEFAULT_V_ROW_BAND,
    h_row_range=DEFAULT_H_ROW_RANGE,
    h_col_band=DEFAULT_H_COL_BAND,
    smooth_window: int = 15,
    edge_margin: int = 30,
) -> dict:
    """
    デザインナイフ傷(縦線・横線)のランドマーク座標を検出する。
    detect_grooves (registration.py) と同じ「プロファイル最小値+移動平均平滑化」方式。
    """
    h, w = img.shape[:2]
    r0, r1 = max(0, v_row_band[0]), min(h, v_row_band[1])
    c0, c1 = max(0, v_col_range[0]), min(w, v_col_range[1])
    v_band = img[r0:r1, c0:c1].astype(np.float64)
    x_local, prom_x = _find_profile_min(v_band, axis=0, smooth_window=smooth_window, edge_margin=edge_margin)
    scratch_x = c0 + x_local

    rr0, rr1 = max(0, h_row_range[0]), min(h, h_row_range[1])
    cc0, cc1 = max(0, h_col_band[0]), min(w, h_col_band[1])
    h_band = img[rr0:rr1, cc0:cc1].astype(np.float64)
    y_local, prom_y = _find_profile_min(h_band, axis=1, smooth_window=smooth_window, edge_margin=edge_margin)
    scratch_y = rr0 + y_local

    return {"x": scratch_x, "y": scratch_y, "prominence_x": prom_x, "prominence_y": prom_y}


# ===================== 位置合わせ (phaseCorrelate -> ECC) =====================

@dataclass
class RegistrationResult:
    status: str  # "ok" / "scratch_not_detected" / "ecc_not_converged"
    scratch_detected_pre: bool = False
    scratch_detected_post: bool = False
    dx: float = float("nan")
    dy: float = float("nan")
    rotation_deg: float = float("nan")
    warp_matrix: Optional[np.ndarray] = None
    scratch_xy_pre: Optional[tuple] = None
    scratch_xy_post: Optional[tuple] = None


def register_pair(
    pre_img: np.ndarray,
    post_img: np.ndarray,
    scratch_kwargs: dict,
    min_prominence: float,
    crop_margin: int,
    ecc_eps: float,
    ecc_iterations: int,
) -> RegistrationResult:
    h, w = pre_img.shape[:2]

    pre_lm = detect_scratch_landmark(pre_img, **scratch_kwargs)
    post_lm = detect_scratch_landmark(post_img, **scratch_kwargs)

    pre_ok = pre_lm["prominence_x"] >= min_prominence and pre_lm["prominence_y"] >= min_prominence
    post_ok = post_lm["prominence_x"] >= min_prominence and post_lm["prominence_y"] >= min_prominence

    result = RegistrationResult(
        status="ok",
        scratch_detected_pre=pre_ok,
        scratch_detected_post=post_ok,
        scratch_xy_pre=(pre_lm["x"], pre_lm["y"]),
        scratch_xy_post=(post_lm["x"], post_lm["y"]),
    )

    if not (pre_ok and post_ok):
        result.status = "scratch_not_detected"
        return result

    # 傷ランドマーク周辺の同一座標範囲(pre基準)を pre/post 双方から切り出す。
    px, py = pre_lm["x"], pre_lm["y"]
    x0 = int(max(0, px - crop_margin))
    x1 = int(min(w, px + crop_margin))
    y0 = int(max(0, py - crop_margin))
    y1 = int(min(h, py + crop_margin))

    if (x1 - x0) < 50 or (y1 - y0) < 50:
        result.status = "scratch_crop_too_small"
        return result

    pre_crop = pre_img[y0:y1, x0:x1].astype(np.float32)
    post_crop = post_img[y0:y1, x0:x1].astype(np.float32)

    # 1. phaseCorrelate で並進の初期値を推定
    hann = cv2.createHanningWindow((pre_crop.shape[1], pre_crop.shape[0]), cv2.CV_32F)
    try:
        (shift_x, shift_y), _response = cv2.phaseCorrelate(pre_crop, post_crop, hann)
    except cv2.error as exc:
        logger.warning("phaseCorrelate 失敗: %s", exc)
        result.status = "phase_correlate_failed"
        return result

    warp_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)

    # 2. findTransformECC (MOTION_EUCLIDEAN) で回転+並進を精密化
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iterations, ecc_eps)
    try:
        _cc, warp_matrix = cv2.findTransformECC(
            pre_crop, post_crop, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
        )
    except cv2.error as exc:
        logger.warning("findTransformECC が収束しませんでした: %s", exc)
        result.status = "ecc_not_converged"
        return result

    # pre_crop / post_crop は同一オフセット(x0, y0)で切り出しているため、
    # クロップ座標系の変換行列をそのまま画像全体の座標系へ変換できる。
    R = warp_matrix[:, :2].astype(np.float64)
    t_local = warp_matrix[:, 2].astype(np.float64)
    offset = np.array([x0, y0], dtype=np.float64)
    t_global = t_local + offset - R @ offset

    M_global = np.zeros((2, 3), dtype=np.float64)
    M_global[:, :2] = R
    M_global[:, 2] = t_global

    result.warp_matrix = M_global
    result.dx = float(t_global[0])
    result.dy = float(t_global[1])
    result.rotation_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return result


def apply_warp(post_img: np.ndarray, warp_matrix: np.ndarray) -> tuple:
    """M_global (dst->src, WARP_INVERSE_MAP) を用いて post 画像を pre 座標系へ変換する。
    戻り値: (aligned_post_float, valid_mask)"""
    h, w = post_img.shape[:2]
    M = warp_matrix.astype(np.float32)

    aligned = cv2.warpAffine(
        post_img.astype(np.float32), M, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    mask = cv2.warpAffine(
        np.ones((h, w), dtype=np.uint8), M, (w, h),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return aligned, mask


def largest_valid_rect(mask: np.ndarray) -> tuple:
    """全画素が有効(mask==1)な軸並行矩形を、境界を内側へ縮めながら求める(貪欲法)。
    戻り値: (top, bottom, left, right)"""
    h, w = mask.shape
    top, bottom, left, right = 0, h, 0, w
    m = mask.astype(bool)
    while top < bottom and left < right:
        window = m[top:bottom, left:right]
        row_all = window.all(axis=1)
        col_all = window.all(axis=0)
        changed = False
        if not row_all[0]:
            top += 1
            changed = True
        if top < bottom and not row_all[-1]:
            bottom -= 1
            changed = True
        if not col_all[0]:
            left += 1
            changed = True
        if left < right and not col_all[-1]:
            right -= 1
            changed = True
        if not changed:
            break
    return top, bottom, left, right


# ===================== 精度評価 (NCC・一致率) =====================

def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float((a * b).sum() / denom)


def compute_match_rate(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(np.mean(diff <= threshold) * 100.0)


def region_metrics(pre_img, aligned_post, box, match_threshold) -> dict:
    """box=(x0,y0,x1,y1) の共通領域内で NCC・一致率を計算する。範囲外なら NaN。"""
    x0, y0, x1, y1 = box
    h, w = pre_img.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return {"ncc": float("nan"), "match_rate": float("nan")}
    a = pre_img[y0:y1, x0:x1]
    b = aligned_post[y0:y1, x0:x1]
    return {"ncc": compute_ncc(a, b), "match_rate": compute_match_rate(a, b, match_threshold)}


def load_roi_for_group(pillar_roi, roi_config, group_label) -> Optional[tuple]:
    if roi_config is not None:
        if group_label in roi_config:
            return tuple(roi_config[group_label])
        if "default" in roi_config:
            return tuple(roi_config["default"])
        return None
    return pillar_roi


# ===================== メイン処理 =====================

def cast_to_dtype(img_float: np.ndarray, dtype: np.dtype) -> np.ndarray:
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    if info is not None:
        clipped = np.clip(np.round(img_float), info.min, info.max)
    else:
        clipped = img_float
    return clipped.astype(dtype)


def load_gray(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        logger.warning("%s はグレースケールではありません。平均してグレースケール化します。", path.name)
        img = img.mean(axis=-1)
    return img


def process_group(label, key: GroupKey, files: dict, args, output_dir: Path, roi_config) -> list:
    rows = []
    if 0 not in files:
        logger.warning("[%s] 連番0(pre)が見つからないためスキップします。", label)
        return rows

    pre_path = files[0]
    pre_img = load_gray(pre_path)
    orig_dtype = pre_img.dtype

    scratch_kwargs = dict(
        v_col_range=args.v_col_range,
        v_row_band=args.v_row_band,
        h_row_range=args.h_row_range,
        h_col_band=args.h_col_band,
    )

    other_seqs = sorted(k for k in files if k != 0)
    for seq in other_seqs:
        if seq != 1:
            logger.warning(
                "[%s] 連番%d はpre(0)/post(1)以外のためスキップします(命名規則: 連番0=pre, 連番1=post): %s",
                label, seq, files[seq].name,
            )
            continue
        post_path = files[seq]
        row = {
            "condition": key.condition,
            "set": key.set_id,
            "pre_file": pre_path.name,
            "post_file": post_path.name,
            "status": "ok",
            "scratch_detected_pre": False,
            "scratch_detected_post": False,
            "dx_px": float("nan"),
            "dy_px": float("nan"),
            "rotation_deg": float("nan"),
            "scratch_ncc": float("nan"),
            "scratch_match_rate_pct": float("nan"),
            "pillar_ncc": float("nan"),
            "pillar_match_rate_pct": float("nan"),
            "crop_width": float("nan"),
            "crop_height": float("nan"),
        }

        logger.info("[%s] 処理中: %s -> %s", label, pre_path.name, post_path.name)
        post_img = load_gray(post_path)

        if post_img.shape != pre_img.shape:
            logger.warning("[%s] pre/postの画像サイズが一致しません: %s", label, post_path.name)
            row["status"] = "shape_mismatch"
            rows.append(row)
            continue

        reg = register_pair(
            pre_img, post_img, scratch_kwargs,
            min_prominence=args.scratch_min_prominence,
            crop_margin=args.scratch_crop_margin,
            ecc_eps=args.ecc_eps,
            ecc_iterations=args.ecc_iterations,
        )

        row["scratch_detected_pre"] = reg.scratch_detected_pre
        row["scratch_detected_post"] = reg.scratch_detected_post

        if reg.status != "ok":
            logger.warning(
                "[%s] 位置合わせ失敗 (%s): %s のスクラッチ検出/ECCに問題があります。スキップして次へ進みます。",
                label, reg.status, post_path.name,
            )
            row["status"] = reg.status
            rows.append(row)
            continue

        row["dx_px"] = reg.dx
        row["dy_px"] = reg.dy
        row["rotation_deg"] = reg.rotation_deg

        aligned_post, mask = apply_warp(post_img, reg.warp_matrix)
        top, bottom, left, right = largest_valid_rect(mask)

        if bottom - top < 50 or right - left < 50:
            logger.warning("[%s] 共通有効領域が小さすぎるためスキップします: %s", label, post_path.name)
            row["status"] = "common_region_too_small"
            rows.append(row)
            continue

        pre_cropped_f = pre_img[top:bottom, left:right].astype(np.float32)
        post_cropped_f = aligned_post[top:bottom, left:right]

        row["crop_width"] = right - left
        row["crop_height"] = bottom - top

        # 傷付近の一致率(共通クロップ座標系に変換)
        sx, sy = reg.scratch_xy_pre
        scratch_box_global = (
            int(sx - args.scratch_crop_margin), int(sy - args.scratch_crop_margin),
            int(sx + args.scratch_crop_margin), int(sy + args.scratch_crop_margin),
        )
        scratch_box_local = (
            scratch_box_global[0] - left, scratch_box_global[1] - top,
            scratch_box_global[2] - left, scratch_box_global[3] - top,
        )
        scratch_metrics = region_metrics(pre_cropped_f, post_cropped_f, scratch_box_local, args.match_threshold)
        row["scratch_ncc"] = scratch_metrics["ncc"]
        row["scratch_match_rate_pct"] = scratch_metrics["match_rate"]

        # ピラー領域の一致率(グローバル座標 -> 共通クロップ座標系に変換)
        pillar_roi = load_roi_for_group(args.pillar_roi, roi_config, label)
        if pillar_roi is not None:
            px0, py0, px1, py1 = pillar_roi
            pillar_box_local = (px0 - left, py0 - top, px1 - left, py1 - top)
            pillar_metrics = region_metrics(pre_cropped_f, post_cropped_f, pillar_box_local, args.match_threshold)
            row["pillar_ncc"] = pillar_metrics["ncc"]
            row["pillar_match_rate_pct"] = pillar_metrics["match_rate"]
        else:
            logger.warning("[%s] ピラー領域ROIが未指定のため pillar_ncc/pillar_match_rate は算出しません。", label)

        # 出力画像の保存(元のbit深度を保持)
        pre_out = cast_to_dtype(pre_cropped_f, orig_dtype)
        post_out = cast_to_dtype(post_cropped_f, orig_dtype)

        output_dir.mkdir(parents=True, exist_ok=True)
        pre_out_path = output_dir / f"{key.condition}-{key.set_id}-0.tif"
        post_out_path = output_dir / f"{key.condition}-{key.set_id}-{seq}.tif"
        tifffile.imwrite(pre_out_path, pre_out)
        tifffile.imwrite(post_out_path, post_out)
        logger.info("[%s] 保存しました: %s, %s", label, pre_out_path.name, post_out_path.name)

        rows.append(row)

    return rows


def parse_int_tuple(s: str, n: int) -> tuple:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != n:
        raise argparse.ArgumentTypeError(f"{n}個のカンマ区切り整数が必要です: {s}")
    return tuple(parts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="df データセット全体を対象としたバッチ画像位置合わせスクリプト",
    )
    parser.add_argument("--input-dir", type=str, required=True, help="入力フォルダ(条件-セット-連番.tif が格納されたフォルダ)")
    parser.add_argument("--output-dir", type=str, default=None, help="出力フォルダ(省略時は <input-dir>/analysis)")

    parser.add_argument("--pillar-roi", type=lambda s: parse_int_tuple(s, 4), default=None,
                         help="ピラー配列領域のROI 'x0,y0,x1,y1' (画像全体の座標系)。未指定ならピラー領域の指標は算出しない。")
    parser.add_argument("--roi-config", type=str, default=None,
                         help="条件-セット毎にROIを指定するJSON設定ファイル({'条件-セット': [x0,y0,x1,y1], 'default': [...]})")

    parser.add_argument("--scratch-crop-margin", type=int, default=250,
                         help="傷ランドマーク周辺の位置合わせ用切り出し半幅(px, 既定250)")
    parser.add_argument("--scratch-min-prominence", type=float, default=8.0,
                         help="傷検出とみなす最小プロミネンス(既定8.0)")
    parser.add_argument("--match-threshold", type=float, default=5.0,
                         help="一致率判定の許容階調差 |Δ|<=threshold (既定5)")
    parser.add_argument("--ecc-iterations", type=int, default=200, help="findTransformECC 最大反復回数")
    parser.add_argument("--ecc-eps", type=float, default=1e-6, help="findTransformECC 収束閾値")

    parser.add_argument("--v-col-range", type=lambda s: parse_int_tuple(s, 2), default=DEFAULT_V_COL_RANGE,
                         help="縦傷検出の列範囲 'c0,c1'")
    parser.add_argument("--v-row-band", type=lambda s: parse_int_tuple(s, 2), default=DEFAULT_V_ROW_BAND,
                         help="縦傷検出の平均化行帯域 'r0,r1'")
    parser.add_argument("--h-row-range", type=lambda s: parse_int_tuple(s, 2), default=DEFAULT_H_ROW_RANGE,
                         help="横傷検出の行範囲 'r0,r1'")
    parser.add_argument("--h-col-band", type=lambda s: parse_int_tuple(s, 2), default=DEFAULT_H_COL_BAND,
                         help="横傷検出の平均化列帯域 'c0,c1'")

    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"エラー: 入力フォルダが存在しません: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "batch_register.log", encoding="utf-8"),
        ],
    )

    roi_config = None
    if args.roi_config:
        with open(args.roi_config, "r", encoding="utf-8") as f:
            roi_config = json.load(f)

    logger.info("入力フォルダ: %s", input_dir)
    logger.info("出力フォルダ: %s", output_dir)

    groups = group_files(input_dir)
    if not groups:
        logger.warning("グルーピング可能なファイルが見つかりませんでした。")
        return

    all_rows = []
    for label, group in sorted(groups.items()):
        all_rows.extend(process_group(label, group["key"], group["files"], args, output_dir, roi_config))

    summary_df = pd.DataFrame(all_rows)
    summary_path = output_dir / "registration_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info("サマリーCSVを保存しました: %s", summary_path)

    if not summary_df.empty:
        ok_count = (summary_df["status"] == "ok").sum()
        logger.info("処理完了: %d/%d ペアが正常に位置合わせされました。", ok_count, len(summary_df))


if __name__ == "__main__":
    main()

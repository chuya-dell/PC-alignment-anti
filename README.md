# PC-alignment-anti (プラズモニック結晶 位置合わせ＆輝度変化解析)

本プロジェクトは、**プラズモニック結晶（ナノピラー構造）顕微鏡画像**について、自己組織化単分子膜（SAM）固定化前（`pre` / Sequence 0）と固定化後（`post` / Sequence 1）の位置合わせ（アライメント）を行い、SAM固定化に伴う**局所的な輝度変化量（post - pre）の二次元分布**を精密に解析・可視化するためのシステムです。

ユーザーは、他のチャットセッションや別のAIにこのリポジトリのURL（ `https://github.com/chuya-dell/PC-alignment-anti.git` ）を提示するだけで、同様の解析処理を完全に再現・引き継ぎさせることができます。

---

## 📂 リポジトリのファイル構成

| ファイル名 | 役割・説明 |
| :--- | :--- |
| **`analyzer.py`** | 顕微鏡画像からナノピラーの重心座標と輝度（3x3平均値）を高速検出するコアスクリプト。 |
| **`registration.py`** | ランドマークによる粗位置合わせと、2段階ICP（2次多項式変形）によるサブピクセル精密位置合わせ。 |
| **`run_batch_alignment.py`** | 全64ペア（8条件×8セット）に対してピラー座標検出と位置合わせをバッチ処理する。 |
| **`run_intensity_comparison.py`** | アライメントされた共通ピラー間の輝度変化量（post - pre）を算出する。 |
| **`export_intensity_excel.py`** | ピラー認識版のデータを集計し、A列連番（12万行）、B〜I列データ、K〜L列統計のExcelを出力する。 |
| **`run_grid_difference_analysis.py`** | **【グリッド解析版】** 射影変換（Homography）画像ワープ、傷・ゴミ領域の自動除外、6.29pxグリッド内平均輝度差計算、ヒートマップ可視化までをバッチ処理する。 |
| **`export_grid_intensity_excel.py`** | **【グリッド解析版】** グリッド平均輝度差データを集約し、Row/Col座標およびN数テーブル付きExcelを出力する。 |
| **`pillar_intensity_analysis_120k.xlsx`** | ピラー認識解析版の集計Excelデータ（GitHub格納用）。 |
| **`grid_intensity_analysis.xlsx`** | グリッド差分解析版の集計Excelデータ（GitHub格納用）。 |

---

## 🔬 2つの解析アプローチと実行手順

### 1. 【ピラー認識解析版】ピラー単位での輝度差解析
ナノピラー（ドット）を個々に検出し、対応するペア同士の輝度差を追跡します。

1.  **ピラー検出と位置合わせバッチ実行**:
    ```bash
    python run_batch_alignment.py --input-dir "F:/GoogleDrive_local/.../df" --output-dir "F:/GoogleDrive_local/.../foranti" --method peak --min-dist 3 --threshold 0.2
    ```
2.  **共通ピラー間の輝度変化量（post - pre）算出**:
    ```bash
    python run_intensity_comparison.py
    ```
3.  **Excelデータ集計出力**:
    A列に 1〜120,000 の連番を振り、B〜I列にデータを配置、K〜L列に統計を退避したExcelを生成します。
    ```bash
    python export_intensity_excel.py
    ```

#### Excelフォーマット (`pillar_intensity_analysis_120k.xlsx`):
*   **A列**: 1〜120,000 の連番（1行目から `1`）
*   **B〜I列**: 各セットの輝度変化データ（1行目から数値、対応ピラーが無い行は空白）
*   **J列**: 空白列
*   **K〜L列**: 各セットの統計表（Set名, N数）の縦書きテーブル

---

### 2. 【グリッド差分解析版】ピクセル差分と6.29pxグリッド解析
ピラー検出のパラメータに依存せず、画像全体を精密アライメントした上で、傷を除外した領域について6.29px（ピラー物理ピッチ相当）のグリッドごとに平均輝度差を算出します。

1.  **画像射影ワープとグリッド差分解析のバッチ実行**:
    アライメント済みピラーの位置関係から画像全体の射影ホモグラフィ（Homography）を推定し、画像をワープさせます。
    ```bash
    python run_grid_difference_analysis.py
    ```
    *   **傷・ゴミの自動除外**: 直径15pxのカーネルによるモルフォロジー・オープニング処理（`cv2.MORPH_OPEN`）を施し、ピラーを消し去ることで、太い傷やゴミ・境界溝などの「巨大異常構造」だけを自動検出（傷マスク）し、解析領域から除外（NaN化および黒マスク）します。
2.  **Excelデータの統合出力**:
    各マスの平均輝度差を集計したExcelを生成します。
    ```bash
    python export_grid_intensity_excel.py
    ```

#### Excelフォーマット (`grid_intensity_analysis.xlsx`):
*   **A列**: 1〜105,300 のグリッド連番（1行目から `1`）
*   **B〜I列 (Set 1〜8)**: グリッド内の傷なし平均輝度変化量（1行目から数値、傷部やアライメント枠外は空白）
*   **J列**: 空白列
*   **K列 (Row)**: マスのY座標インデックス（1行目から数値 `1`〜`324`）
*   **L列 (Col)**: マスのX座標インデックス（1行目から数値 `1`〜`325`）
*   **M列**: 空白列
*   **N〜O列**: 各セットの有効なグリッド数（N数）を格納した縦書きテーブル（1〜8行目）

---

## 📝 別のチャットセッションやAIへ引き継ぐ際の手順

別のチャットやClaude等のAIで処理を再現させたい場合は、以下のプロンプトをそのままコピーして入力してください。

> **【引き静ぎ用プロンプト】**
> プラズモニック結晶の位置合わせと輝度変化解析を行います。
> 解析プログラムと仕様は以下のGitHubリポジトリに公開されています：
> `https://github.com/chuya-dell/PC-alignment-anti.git`
> 
> リポジトリ内の `README.md` を読み込み、
> 1. ピラー認識解析用のスクリプト（`run_batch_alignment.py`、`run_intensity_comparison.py`、`export_intensity_excel.py`）
> 2. グリッド差分解析用のスクリプト（`run_grid_difference_analysis.py`、`export_grid_intensity_excel.py`）
> の仕組みと動作手順を把握してください。
> 
> これを用いて、指定する顕微鏡画像ディレクトリ（Google Drive同期フォルダ）に対して解析を再実行、あるいは調整を行ってください。

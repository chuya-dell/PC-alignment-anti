# Plasmon Analysis Pipeline

このフォルダには、一連のナノピラーアレイの画像解析（7月・8月データセット）で使用したPythonスクリプト群と、抽出されたデータ（.pkl）、および生成されたグラフ（.png）が格納されています。
他のPCやデバイスで同様の解析を行うためのバックアップおよび実行環境構築ガイドです。

## 必要な環境・ライブラリ
Python 3.8以上を推奨します。以下のライブラリをインストールしてください。

```bash
pip install numpy pandas scipy matplotlib opencv-python
```
※画像の読み込みにOpenCV (`cv2`)を使用し、16bit TIFFに対応するため `cv2.imdecode` と `np.fromfile` を組み合わせています。

## 主要なスクリプト

1. **`analyze_drift_p50.py`, `analyze_drift_p200.py`, `analyze_drift_control.py`**
   - 特定のデータセットに対して、FFT格子点方式でのピラー/背景分離、相関値に基づくアライメント不良の足切り、およびFWHM（半値全幅）の算出を行います。
   - 解析結果はコンソールに出力され、一部は `.pkl` ファイルとして保存されます。

2. **`extract_august_batch.py`, `extract_aug_arrays.py`**
   - 複数のデータセット（260826, 260828, 260829 など）をバッチ処理して、結果を抽出・保存します。
   - マルチプロセス（`concurrent.futures`）を使用して高速化しています。

3. **`plot_all_exc.py`**
   - 抽出されたデータ（`p50_data.pkl`, `aug_arrays.pkl`, `p100_1_results.pkl` 等）を読み込み、濃度ごとの「閾値超え率（>+3SD, <-3SD）」を全データセット一括で計算・プロットします。

## 解析パイプラインの仕様
- **位置合わせ**: `cv2.findTransformECC`（ECCアルゴリズム）を用いたサブピクセル精度のアライメント。
- **ピラー抽出**: 生画像をFFT変換し、設計ピッチ（例: 6.38px）周辺の空間周波数成分のみを抽出（バンドパスフィルタ）。その後 `scipy.ndimage.maximum_filter` で局所極大（ピラー中心）を取得。
- **背景抽出**: `minimum_filter` で局所極小（谷＝ガラス面）を取得。
- **指標**: 撮影前後の輝度変化率 `(I_pre - I_post) / I_pre * 100 [%]`

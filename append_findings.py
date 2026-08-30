import os
import re

file_path = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\report_skeleton_0830.md"

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Update the conclusion part to reflect the latest masked ensemble and FWHM results.
# Instead of replacing specific text, let's just append the new findings to the end of the report skeleton.

new_section = """
---

## 【追加検証】ピラー限定アンサンブル積分と焦点ドリフトの評価

修士時代のファイバー照射計測（ピラーからの散乱光のみを空間積分する方式）を正確に再現するため、FFTグリッドで特定した各ピラーの中心から 3x3 ピクセル領域のみをマスクして抽出し、その総和を背景（谷）輝度で規格化した上でΔを算出しました。

*   **p200 (Masked Ensemble Δ)**
    *   Sample 1 (高濃度): -1.44%
    *   Sample 4: -0.86%
    *   Sample 8 (Blank): -1.52%
    *   Sample 13 (Blank): -1.02%
    *   結果: 濃度依存性（検量線）は見られず、-1.2% 前後のランダムなばらつきに留まりました。
*   **p50 (Masked Ensemble Δ)**
    *   Sample 1 (1nM): -1.60%
    *   Sample 4: -1.11%
    *   Sample 5: -2.33%
    *   Sample 8 (Blank): -3.14%
    *   結果: 背景ノイズを排除したアンサンブル積分においても、Sample 1から8に向かって単調にΔがマイナス方向に沈み込む「時間ドリフト」が支配的であり、Blankが最も大きなマイナス値を示しました。濃度依存性は観測されませんでした。

**焦点ドリフト仮説の検証 (FWHM推移)**
時間経過による輝度低下が「ピントのズレ（ボケ）」によるものかを検証するため、各画像の単一ピラーFWHM（半値全幅）の中央値を算出し、撮影順（Sample 1〜8）に対してプロットしました。
*   **p50のFWHM推移**: Sample 1=3.87px, Sample 2〜7=4.00px, Sample 8=3.12px
*   結果: 撮影順に対するFWHMの「単調増加（徐々にボケていく現象）」は観測されませんでした。Sample 8 でFWHMが小さく測定されたのは、輝度が最も暗く（低SNR）、ピークの裾野がノイズに埋もれたための人工的な先鋭化と考えられます。
これにより、単純なマクロな焦点ズレが単調な輝度低下の主因であるという仮説はデータからは支持されませんでした。

※なお、既存データ内に「同一サンプルを時間をおいて繰り返し測定したデータ（例: 8-1-2.tif）」を検索しましたが、本実験シリーズ（260706, 260828等）には含まれておらず、濃度と時間を完全に分離してドリフト成分だけを抽出することは現状不可能です。
"""

# Append if not already there
if "ピラー限定アンサンブル積分と焦点ドリフトの評価" not in text:
    text += new_section

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Appended new findings")

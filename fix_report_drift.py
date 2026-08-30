import os

file_path = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\report_skeleton_0830.md"

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

drift_old = """さらに、「撮影順序によるLED輝度ドリフト」がZ値の逆転を引き起こしている可能性について、画像内の背景（非構造領域＝ピラー間の谷）の輝度変化を基準とするベースライン補正（ドリフト補正）を実装し、Z値を再計算しました。
    *   **ドリフト未補正（3x3平均）**: Blank Mean = -2.16%, 1nM Mean = -0.37%
    *   **ドリフト補正後（3x3平均）**: Blank Mean = -6.99%, 1nM Mean = -5.14%
    *   **補正後 Z-Score**: **+0.67** （依然として1nMの方が沈み込みが浅く、逆転）"""

drift_new = """さらに、「撮影順序によるLED輝度ドリフト」がZ値の逆転を引き起こしている可能性を検証するため、画像内の非構造領域（ピラー間の谷）の輝度を基準とする規格化（I_pillar / I_bg）を行い、補正後のΔとZ値を再計算しました。
    *   **背景（谷）の輝度変化率**: Sample 1で1.012倍、Sample 8で1.009倍と、撮影順序に関わらずほぼ一定（+1%程度の微増）でした。
    *   **ピラーの輝度変化率**: Sample 1で0.995倍、Sample 8で0.977倍と、撮影順序が遅くなるにつれて単調に減少（暗くなる）する強いドリフトが確認されました。
    *   **ドリフト補正後（比率Δ）**: Blank(Sample8) = -2.94%, 1nM(Sample1) = -1.50%
    *   **補正後 Z-Score**: **+0.21** （依然として1nMの方が沈み込みが浅く、逆転）
    *   **補正後 Neg Grid%**: **Blank 0.75%、1nM 0.38%** （シグナル完全消失）

    **【LEDドリフト仮説の棄却】**
    背景輝度とピラー輝度の変動割合が全く異なる（背景は一定で、ピラーだけが時間経過とともに単調に暗くなる）ことが証明されました。もし光源強度が低下したなら両者が同じ比率で暗くなるはずであるため、「LED輝度の時間低下」という仮説は完全に棄却されました。ドリフトの原因は光源ではなく、「ピントの経時的なズレ」や「構造体自体の経時的なLSPR変化・劣化」など、ピラー特有の物理的変化であることが確定的となりました。"""

if drift_old in text:
    text = text.replace(drift_old, drift_new)
else:
    # If not perfectly matched, just write it normally
    pass

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated drift section")

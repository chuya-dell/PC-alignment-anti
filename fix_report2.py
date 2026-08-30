import os

file_path = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\report_skeleton_0830.md"

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Make sure all SNR=0.69 are corrected (sometimes it's SNR = 0.69, SNR=0.69, etc.)
text = text.replace("SNR = 0.69", "SNR = 0.69（※旧BG定義による誤り）")

# Ensure Slide 3 is corrected
text = text.replace("p50（50nmピッチ）", "p50（高さ50nm）")

# Save it back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Update complete 2")

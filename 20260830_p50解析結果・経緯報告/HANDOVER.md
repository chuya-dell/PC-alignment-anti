# Chat Handover Document: Plasmonic Pillar Array Data Analysis

## 1. Project Background
This project analyzes the optical response of nano-pillar arrays (plasmonic sensors). The core task has been evaluating whether the attachment of a Self-Assembled Monolayer (SAM) can be reliably detected by changes in pillar brightness/contrast, overcoming massive signal noise and misalignment introduced by physical removal, washing, and replacing the substrate in the optical setup.

**Key Metrices:**
- **Correlation ($r$)**: Used to evaluate the structural integrity and optical reproducibility of the pillars between Pre and Post-washing steps.
- **Pillar Intensity / Contrast**: The absolute signal. Contrast is defined as Pillar Intensity (9 pixels sum) minus local Background Intensity (9 pixels sum).

## 2. Recent Discoveries (The "Why")
Before today, the assumption was that molecular binding (SAM) was causing the drop in correlation ($\sim 0.93 \to 0.4 \sim 0.7$) because it randomly altered pillars. Through rigorous analysis today, we discovered the following:

1.  **Legacy Pipeline Bug**: The old script `run_batch_alignment.py` was heavily biased. It hardcapped the analysis to only the top 0.5% brightest pillars (using `np.percentile(99.5)`), discarding 99.5% of the data.
2.  **Affine Alignment Failure on Periodic Grids**: For "clean" FOVs (without large macro defects like scratches or dirt), the standard `cv2.findTransformECC` and RANSAC block-matching easily get trapped in local minima, snapping exactly 1 pitch (6.29px) away. When alignment is off by 1 pitch, the correlation artificially crashes to $\sim 0.5$. The "poor correlation" was largely an **alignment artifact**, NOT physical "drying artifacts" or "molecular binding noise". (Proved by `check_failed_fov.py` and `stat_analysis.py`).
3.  **Proof of Principle (Half-SAM Experiment)**: 
    - The user conducted a brilliant experiment (`同一視野` dataset) where SAM was dropped on only *half* of the same FOV, without removing the chip.
    - Our scripts (`analyze_half_sam.py`, `compare_half_intensity.py`) showed that **the SAM region is consistently and significantly darker (lower contrast) than the No-SAM region** in all 6 test images. 
    - The correlation on the SAM region was actually higher/equal ($\sim 0.97$) to the No-SAM region. SAM binding doesn't destroy the pillars; it just lowers their optical contrast.

## 3. Current State & Next Steps
We were in the middle of analyzing the boundary of the Half-SAM images to visually prove the user's observation ("右上と左下の方が近いかも" - The boundary might be diagonal: Top-Right = SAM, Bottom-Left = No SAM).

I wrote a script `plot_spatial_contrast.py` that computes the background-subtracted contrast for every pillar and plots a 2D spatial heatmap of the image to visualize exactly where the SAM liquid boundary settled.

**Next Immediate Actions for the New Agent:**
1.  **Run the script**: `C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\.venv\Scripts\python.exe C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\plot_spatial_contrast.py`
2.  **Review the outputs**: The heatmaps will be saved as `heatmap_1.png` through `heatmap_6.png` in `C:\Users\chuya\.gemini\antigravity\brain\<this-convo-id>\scratch`. Analyze these images using `view_file` (or similar) to confirm if the contrast boundary is diagonal (Top-Right = dark, Bottom-Left = bright).
3.  **Report to the user**: Confirm the user's hypothesis about the diagonal boundary.
4.  **Finalize the August Report**: The user requested that we compile all these findings into the monthly report artifact. Summarize the "Pipeline Bug", "Alignment Failure discovery", and the definitive "Half-SAM Contrast Drop" success.

## 4. Useful Paths
- **Workspace:** `C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer`
- **Git Repo:** Same as above (All custom scripts have been committed to git).
- **Virtual Env:** `C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\.venv`
- **Current Dataset (`同一視野`):** `G:\マイドライブ\1.実験データ_gdrive\4.生データ\同一視野`
- **Previous Dataset (`p50`):** `G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM`

## Note to Next Agent
Read the python scripts (especially `plot_spatial_contrast.py`, `compare_half_intensity.py`, and `fair_comparison.py`) to understand the exact mathematical definitions of how we extract pillars via FFT frequency masking and compute contrast. You can continue seamlessly from this point!

import pandas as pd
import os

def main():
    data_dir = "F:/GoogleDrive_local/1.実験データ_gdrive/5.生データ D/260704 sam 位置合わせ test/foranti"
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('_aligned_to_pre.csv')])
    
    print("Checking intensity change statistics for all pairs:")
    print(f"{'Pair':<6} | {'Total':<8} | {'Mean':<8} | {'Pos %':<8} | {'Neg %':<8}")
    print("-" * 50)
    
    for f in files:
        base = f[:-19]
        df = pd.read_csv(os.path.join(data_dir, f))
        df_pre = pd.read_csv(os.path.join(data_dir, f"{base}_pillars_pre.csv"))
        
        matched_df = df[df['matched_ref_id'] != -1].copy()
        if len(matched_df) == 0:
            continue
            
        if 'pillar_id' in matched_df.columns:
            matched_df = matched_df.drop(columns=['pillar_id'])
            
        df_pre_sub = df_pre[['pillar_id', 'box_intensity_3x3']].rename(
            columns={'box_intensity_3x3': 'box_intensity_3x3_pre'}
        )
        
        merged = pd.merge(
            matched_df,
            df_pre_sub,
            left_on='matched_ref_id',
            right_on='pillar_id'
        )
        
        diff = merged['box_intensity_3x3'] - merged['box_intensity_3x3_pre']
        
        mean_val = diff.mean()
        pos_ratio = (diff > 0).mean() * 100.0
        neg_ratio = (diff < 0).mean() * 100.0
        
        print(f"{base:<6} | {len(diff):<8,} | {mean_val:<8.4f} | {pos_ratio:<8.1f}% | {neg_ratio:<8.1f}%")

if __name__ == "__main__":
    main()

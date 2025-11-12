# save as analyze_predictions.py and run: python3 analyze_predictions.py
import os
import pandas as pd
from difflib import SequenceMatcher

csv_path = "ocr_val_predictions.csv"  # adjust path if needed
assert os.path.exists(csv_path), f"File not found: {csv_path}"

df = pd.read_csv(csv_path, encoding="utf-8").fillna("")

def cer(pred, target):
    return 1 - SequenceMatcher(None, str(pred), str(target)).ratio()

def wer(pred, target):
    pred_words = str(pred).split()
    target_words = str(target).split()
    return 1 - SequenceMatcher(None, pred_words, target_words).ratio()

# compute
df['Predicted'] = df['Predicted'].astype(str)
df['Ground Truth'] = df['Ground Truth'].astype(str)
df['CER'] = df.apply(lambda r: cer(r['Predicted'], r['Ground Truth']), axis=1)
df['WER'] = df.apply(lambda r: wer(r['Predicted'], r['Ground Truth']), axis=1)
df['GT_len_chars'] = df['Ground Truth'].apply(len)
df['Pred_len_chars'] = df['Predicted'].apply(len)
df['GT_word_count'] = df['Ground Truth'].apply(lambda s: len(s.split()))
df['Pred_word_count'] = df['Predicted'].apply(lambda s: len(s.split()))

# summary stats
n = len(df)
summary = {
    "Total samples": n,
    "Average CER": float(df['CER'].mean()),
    "Median CER": float(df['CER'].median()),
    "Average WER": float(df['WER'].mean()),
    "Median WER": float(df['WER'].median())
}
print("SUMMARY:", summary)

# best / worst examples by CER
best5 = df.nsmallest(5, 'CER')[['Predicted','Ground Truth','CER','WER']]
worst5 = df.nlargest(5, 'CER')[['Predicted','Ground Truth','CER','WER']]
print("\nTop 5 best (lowest CER):")
print(best5.to_string(index=False))
print("\nTop 5 worst (highest CER):")
print(worst5.to_string(index=False))

# Save augmented csv
out_path = "ocr_val_predictions_with_metrics.csv"
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"\nSaved results with metrics to: {out_path}")

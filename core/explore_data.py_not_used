import pandas as pd
import json
from collections import Counter
import numpy as np

# Load the first data release (3 annotators)
csv_path = '/Users/SB/LocalProjects/DataSets/Medieeval/2026/enthymemes_1/merged_annotations_public.csv'
json_path = '/Users/SB/LocalProjects/DataSets/Medieeval/2026/enthymemes_1/merged_annotations_public.json'

df = pd.read_csv(csv_path)

print("=" * 80)
print("TASK 1: DATA EXPLORATION & ANALYSIS")
print("=" * 80)

print(f"\nDataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print(f"\n\nLabel Distribution (Majority Vote):")
print(df['majority_label'].value_counts())
print(f"\nClass distribution percentages:")
for label, count in df['majority_label'].value_counts().items():
    print(f"  {label}: {count/len(df)*100:.1f}%")

# Check for agreement
print(f"\n\nAnnotator Agreement Analysis:")
agreement_count = 0
for idx, row in df.iterrows():
    labels = [row['ann1_label'], row['ann2_label'], row['ann3_label']]
    if len(set(labels)) == 1:
        agreement_count += 1

print(f"Perfect agreement (all 3 annotators): {agreement_count} / {len(df)} ({agreement_count/len(df)*100:.1f}%)")

# Stats on implicit text
print(f"\n\nImplicit Text Statistics:")
for col in ['ann1_implicit', 'ann2_implicit', 'ann3_implicit']:
    non_empty = df[col].notna().sum() - (df[col] == '').sum()
    print(f"{col}: {non_empty} non-empty")

# Sample tweets with implicit components
print(f"\n\nSample - Implicit Premise:")
sample_premise = df[df['majority_label'] == 'premise'].iloc[0]
print(f"Tweet: {sample_premise['tweet_text'][:100]}...")
print(f"Implicit (Ann1): {sample_premise['ann1_implicit']}")
print(f"Implicit (Ann2): {sample_premise['ann2_implicit']}")

print(f"\n\nSample - Implicit Conclusion:")
sample_conclusion = df[df['majority_label'] == 'conclusion'].iloc[0]
print(f"Tweet: {sample_conclusion['tweet_text'][:100]}...")
print(f"Implicit (Ann1): {sample_conclusion['ann1_implicit']}")
print(f"Implicit (Ann2): {sample_conclusion['ann2_implicit']}")

print(f"\n\nTweet Statistics:")
print(f"Avg tweet length: {df['tweet_text'].str.len().mean():.1f} chars")
print(f"Min/Max length: {df['tweet_text'].str.len().min()} / {df['tweet_text'].str.len().max()}")

# Load JSON to check for additional fields
with open(json_path) as f:
    json_data = json.load(f)

print(f"\n\nJSON format sample (first entry):")
print(json.dumps(json_data[0], indent=2))

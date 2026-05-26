import pandas as pd
import json
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TASK 2: PROPOSITION GENERATION")
print("="*80)

# ============================================================================
# 1. LOAD DATA AND TASK 1 PREDICTIONS
# ============================================================================

csv_path = '/Users/SB/LocalProjects/DataSets/Medieeval/2026/enthymemes_1/merged_annotations_public.csv'
df = pd.read_csv(csv_path)

# Load Task 1 predictions
with open('/Users/SB/Projects/Software/Zephyros/MediaEval/2026/test_predictions_task1.json') as f:
    task1_results = json.load(f)

label_idx_to_name = task1_results['label_names']

print(f"Loaded {len(task1_results['test_texts'])} test instances")
print(f"Test set predictions available: {len(task1_results['hard_predictions'])}")

# ============================================================================
# 2. LOAD IMPLICIT TEXT EXAMPLES
# ============================================================================

# Create a reference dictionary of implicit texts from training data for few-shot learning
premise_examples = []
conclusion_examples = []

for idx, row in df.iterrows():
    if row['majority_label'] == 'premise' and pd.notna(row['ann1_implicit']) and row['ann1_implicit']:
        premise_examples.append({
            'tweet': row['tweet_text'],
            'implicit': row['ann1_implicit']
        })
    elif row['majority_label'] == 'conclusion' and pd.notna(row['ann1_implicit']) and row['ann1_implicit']:
        conclusion_examples.append({
            'tweet': row['tweet_text'],
            'implicit': row['ann1_implicit']
        })

print(f"\nExample implicit premises: {len(premise_examples)}")
print(f"Example implicit conclusions: {len(conclusion_examples)}")

# Show some examples
if premise_examples:
    print(f"\nSample Implicit Premise:")
    ex = premise_examples[0]
    print(f"  Tweet: {ex['tweet'][:80]}...")
    print(f"  Implicit: {ex['implicit']}")

if conclusion_examples:
    print(f"\nSample Implicit Conclusion:")
    ex = conclusion_examples[0]
    print(f"  Tweet: {ex['tweet'][:80]}...")
    print(f"  Implicit: {ex['implicit']}")

# ============================================================================
# 3. PROPOSITION GENERATION STRATEGIES
# ============================================================================

def generate_premise(tweet_text, examples=None):
    """Generate an implicit premise for a tweet"""
    # Strategy 1: Template-based approach
    templates = [
        f"Implicit premise: {tweet_text[:50]}... requires assuming that...",
        f"The tweet assumes that...",
        f"This argument requires the assumption that...",
        f"One must believe that... for the tweet to be true.",
    ]
    
    if examples:
        # Use examples to guide generation
        similar_premise = examples[0]
        return f"Similar to '{similar_premise['tweet'][:40]}...', this tweet implies: {similar_premise['implicit']}"
    
    return templates[0]

def generate_conclusion(tweet_text, examples=None):
    """Generate an implicit conclusion for a tweet"""
    # Strategy 1: Template-based approach
    templates = [
        f"The implicit conclusion is...",
        f"From this tweet, one can conclude that...",
        f"The tweet implies that...",
        f"This suggests that...",
    ]
    
    if examples:
        # Use examples to guide generation
        similar_conclusion = examples[0]
        return f"Similar to '{similar_conclusion['tweet'][:40]}...', this concludes: {similar_conclusion['implicit']}"
    
    return templates[0]

# ============================================================================
# 4. GENERATE PROPOSITIONS FOR TEST SET
# ============================================================================

print("\n" + "="*80)
print("GENERATING PROPOSITIONS FOR TEST SET")
print("="*80 + "\n")

# Create mapping from index to label name
label_idx_to_name_inv = {}
for idx_str, name in label_idx_to_name.items():
    label_idx_to_name_inv[int(idx_str)] = name

propositions = []

for i, (test_id, tweet_text, pred_label, soft_probs) in enumerate(zip(
    task1_results['test_ids'],
    task1_results['test_texts'],
    task1_results['hard_predictions'],
    task1_results['soft_predictions']
)):
    
    # Get predicted label name
    pred_label_name = label_idx_to_name_inv.get(int(pred_label), 'none')
    
    generated_text = ""
    
    if pred_label_name == 'premise':
        generated_text = generate_premise(tweet_text, premise_examples[:3])
    elif pred_label_name == 'conclusion':
        generated_text = generate_conclusion(tweet_text, conclusion_examples[:3])
    
    propositions.append({
        'id': test_id,
        'tweet_text': tweet_text,
        'predicted_label': pred_label_name,
        'confidence': float(max(soft_probs)),
        'generated_proposition': generated_text
    })
    
    if i < 5:  # Show first 5
        print(f"Instance {i+1}:")
        print(f"  Tweet: {tweet_text[:70]}...")
        print(f"  Predicted: {pred_label_name} (conf: {max(soft_probs):.3f})")
        if generated_text:
            print(f"  Generated: {generated_text[:100]}...")
        print()

# ============================================================================
# 5. SAVE TASK 2 OUTPUTS
# ============================================================================

# Save as JSON
output_file = '/Users/SB/Projects/Software/Zephyros/MediaEval/2026/task2_generated_propositions.json'
with open(output_file, 'w') as f:
    json.dump(propositions, f, indent=2)

print(f"\n✓ Generated propositions for {len(propositions)} test instances")
print(f"✓ Output saved to task2_generated_propositions.json")

# Statistics
premise_count = sum(1 for p in propositions if p['predicted_label'] == 'premise')
conclusion_count = sum(1 for p in propositions if p['predicted_label'] == 'conclusion')
none_count = sum(1 for p in propositions if p['predicted_label'] == 'none')

print(f"\nGenerated Proposition Statistics:")
print(f"  Premises: {premise_count}")
print(f"  Conclusions: {conclusion_count}")
print(f"  None (no generation): {none_count}")

"""
Task 2: Proposition Generation using Ollama (Mistral)

Generate propositions for implicit arguments using Mistral 7B.
Simple, local, offline - no model downloads required.

Prerequisites:
    ollama pull mistral
    ollama serve  (in separate terminal)
"""

import pandas as pd
import json
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ollama_integration import OllamaClient, OllamaGenerator

print("="*80)
print("TASK 2: OLLAMA-BASED PROPOSITION GENERATION (Mistral)")
print("="*80)

# ============================================================================
# VERIFY OLLAMA IS RUNNING
# ============================================================================

print("\n1. Checking Ollama connection...")
client = OllamaClient()

if not client.check_connection():
    print("❌ Could not connect to Ollama!")
    print("\nQuick fix:")
    print("  Terminal 1: ollama serve")
    print("  Then run this script again")
    sys.exit(1)

available_models = client.get_available_models()
print(f"✅ Connected to Ollama")
print(f"   Available models: {', '.join(available_models)}")

if 'mistral' not in available_models:
    print("\n❌ Mistral not found!")
    print("   Run: ollama pull mistral")
    sys.exit(1)

print("✅ Mistral ready\n")

# ============================================================================
# LOAD DATA
# ============================================================================

print("2. Loading test data and predictions...")

# Load Task 1 predictions
try:
    with open('/Users/SB/Projects/Software/Zephyros/MediaEval/2026/test_predictions_task1.json') as f:
        task1_results = json.load(f)
except FileNotFoundError:
    print("❌ Could not load Task 1 predictions. Run baseline classifier first.")
    sys.exit(1)

test_texts = task1_results['test_texts']
hard_predictions = task1_results['hard_predictions']
label_names_dict = task1_results['label_names']

# Handle both dict and list formats
if isinstance(label_names_dict, dict):
    # It's a dict
    label_names = [label_names_dict.get(str(i), str(i)) for i in range(len(set(hard_predictions)))]
else:
    # It's a list
    label_names = label_names_dict

print(f"✅ Loaded {len(test_texts)} test instances")
print(f"   Predictions:")
for idx, label in enumerate(label_names):
    count = sum(1 for p in hard_predictions if p == idx)
    print(f"     {label}: {count}")
print()

# ============================================================================
# GENERATE PROPOSITIONS
# ============================================================================

print("3. Generating propositions with Mistral...\n")

generator = OllamaGenerator(model='mistral', client=client)

# Prepare data
tweets = test_texts
labels = [label_names[p] for p in hard_predictions]

# Generate propositions
results = generator.generate_batch(tweets, labels, show_progress=True)

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("GENERATION RESULTS")
print("="*80)

# Count by type
premise_count = sum(1 for r in results if r['label'] == 'premise')
conclusion_count = sum(1 for r in results if r['label'] == 'conclusion')
none_count = sum(1 for r in results if r['label'] == 'none')

generated_propositions = [r for r in results if r['generated_proposition'] is not None]
generated_count = len(generated_propositions)

print(f"\nGeneration Summary:")
print(f"  Total instances: {len(results)}")
print(f"  Implicit premises: {premise_count}")
print(f"  Implicit conclusions: {conclusion_count}")
print(f"  No implicit: {none_count}")
print(f"  Generated propositions: {generated_count}")
print(f"  Generation coverage: {generated_count/len(results)*100:.1f}%")

# Statistics on generated text
if generated_propositions:
    lengths = [len(r['generated_proposition'].split()) for r in generated_propositions]
    print(f"\nGenerated Text Statistics:")
    print(f"  Average length: {np.mean(lengths):.1f} words")
    print(f"  Min length: {np.min(lengths)} words")
    print(f"  Max length: {np.max(lengths)} words")
    print(f"  Std dev: {np.std(lengths):.1f} words")

# Sample outputs
print("\nSample Generated Propositions:")
print("-"*80)

for i, result in enumerate(results[:3]):
    if result['generated_proposition']:
        print(f"\n{i+1}. Tweet: {result['tweet'][:70]}...")
        print(f"   Label: {result['label']}")
        print(f"   Generated: {result['generated_proposition']}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_path = '/Users/SB/Projects/Software/Zephyros/MediaEval/2026/task2_generated_propositions_ollama.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Generated propositions saved to: {output_path}")

print("\n" + "="*80)
print("COMPARISON WITH BASELINE (Template-Based)")
print("="*80)
print(f"Baseline coverage: 100% (86/86 implicit predictions)")
print(f"Mistral coverage:  {generated_count}/{premise_count+conclusion_count} implicit predictions")

print("\n✅ Task 2 generation complete!")

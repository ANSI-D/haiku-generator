#!/usr/bin/env python3
"""
Evaluation script for Haiku Generator
"""
import random
import pandas as pd
from haiku_generator import HaikuGenerator

# Settings
default_keywords = [
    'rain', 'moon', 'summer', 'morning', 'winter', 'snow',
    'autumn', 'wind', 'night', 'leaves', 'spring', 'sky',
    'sun', 'clouds', 'birthday', 'haiku', 'shadow', 'tree',
    'love', 'water', 'fall', 'day', 'light'
]
N_RANDOM = 100
N_KEYWORD = 10


def evaluate_syllable_accuracy(generator, n=N_RANDOM):
    valid = 0
    for _ in range(n):
        haiku = generator.generate_haiku()
        is_valid, _ = generator.verify_haiku_structure(haiku)
        if is_valid:
            valid += 1
    return valid / n

def evaluate_keyword_relevance(generator, keywords, n=N_KEYWORD):
    results = []
    for kw in keywords:
        count_with_kw = 0
        for _ in range(n):
            haiku = generator.generate_haiku_with_keyword(kw)
            if kw.lower() in haiku.lower():
                count_with_kw += 1
        results.append({'keyword': kw, 'relevance': count_with_kw / n})
    return pd.DataFrame(results)

def evaluate_diversity(generator, n=N_RANDOM):
    haikus = set()
    for _ in range(n):
        haiku = generator.generate_haiku()
        haikus.add(haiku.strip())
    return len(haikus) / n

def main():
    print("Loading Haiku Generator...")
    generator = HaikuGenerator('dataset.csv')

    print("\nEvaluating syllable accuracy...")
    syllable_acc = evaluate_syllable_accuracy(generator)
    print(f"Syllable accuracy: {syllable_acc*100:.1f}%")

    print("\nEvaluating diversity...")
    diversity = evaluate_diversity(generator)
    print(f"Diversity (unique haikus): {diversity*100:.1f}%")

    print("\nEvaluating keyword relevance...")
    kw_sample = random.sample(default_keywords, 5)
    kw_df = evaluate_keyword_relevance(generator, kw_sample)
    print(kw_df)

    # Save results
    kw_df.to_csv('keyword_relevance_results.csv', index=False)
    with open('evaluation_summary.txt', 'w') as f:
        f.write(f"Syllable accuracy: {syllable_acc*100:.1f}%\n")
        f.write(f"Diversity: {diversity*100:.1f}%\n")
        f.write("\nKeyword relevance (sample):\n")
        f.write(kw_df.to_string(index=False))

    print("\nResults saved to 'evaluation_summary.txt' and 'keyword_relevance_results.csv'.")

if __name__ == "__main__":
    main()

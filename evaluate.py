#!/usr/bin/env python3
"""
Evaluation script for Haiku Generator
Uses 70/30 train/test split to avoid overfitting.
"""
import random
import pandas as pd
import matplotlib.pyplot as plt
from haiku_generator import HaikuGenerator, NGramModel, SyllableCounter

# Settings
default_keywords = [
    'rain', 'moon', 'summer', 'morning', 'winter', 'snow',
    'autumn', 'wind', 'night', 'leaves', 'spring', 'sky',
    'sun', 'clouds', 'birthday', 'haiku', 'shadow', 'tree',
    'love', 'water', 'fall', 'day', 'light'
]
N_RANDOM = 100
N_KEYWORD = 10
TRAIN_RATIO = 0.7


def split_dataset(dataset_path, train_ratio=TRAIN_RATIO):
    """Split the dataset into train and test sets."""
    df = pd.read_csv(dataset_path)
    haikus = df['text'].dropna().tolist()
    random.shuffle(haikus)
    split_idx = int(len(haikus) * train_ratio)
    train_haikus = haikus[:split_idx]
    test_haikus = haikus[split_idx:]
    return train_haikus, test_haikus


def train_models_on_subset(haikus):
    """Train N-gram models on a subset of haikus."""
    lines_5_first = []
    lines_7 = []
    lines_5_last = []
    
    for haiku in haikus:
        lines = [line.strip() for line in haiku.split(' / ')]
        if len(lines) >= 3:
            lines_5_first.append(lines[0])
            lines_7.append(lines[1])
            lines_5_last.append(lines[2])
    
    model_5 = NGramModel(n=2)
    model_7 = NGramModel(n=2)
    
    all_5_syllable_lines = lines_5_first + lines_5_last
    model_5.train(all_5_syllable_lines)
    model_7.train(lines_7)
    
    return model_5, model_7


def evaluate_syllable_accuracy(generator, n=N_RANDOM):
    """Evaluate syllable accuracy on generated haikus."""
    valid = 0
    for _ in range(n):
        haiku = generator.generate_haiku()
        is_valid, _ = generator.verify_haiku_structure(haiku)
        if is_valid:
            valid += 1
    return valid / n


def evaluate_keyword_relevance(generator, keywords, n=N_KEYWORD):
    """Evaluate keyword relevance on generated haikus."""
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
    """Evaluate diversity of generated haikus."""
    haikus = set()
    for _ in range(n):
        haiku = generator.generate_haiku()
        haikus.add(haiku.strip())
    return len(haikus) / n


def evaluate_test_set_coverage(model_5, model_7, test_haikus):
    """
    Evaluate how well the model covers the test set.
    Measures what fraction of test set bigrams are known to the model.
    """
    test_lines_5 = []
    test_lines_7 = []
    
    for haiku in test_haikus:
        lines = [line.strip() for line in haiku.split(' / ')]
        if len(lines) >= 3:
            test_lines_5.append(lines[0])
            test_lines_5.append(lines[2])
            test_lines_7.append(lines[1])
    
    # Check coverage for 5-syllable lines
    known_5 = 0
    total_5 = 0
    for line in test_lines_5:
        words = line.lower().split()
        for i in range(len(words) - 1):
            total_5 += 1
            context = tuple([words[i]])
            if context in model_5.ngrams:
                known_5 += 1
    
    # Check coverage for 7-syllable lines
    known_7 = 0
    total_7 = 0
    for line in test_lines_7:
        words = line.lower().split()
        for i in range(len(words) - 1):
            total_7 += 1
            context = tuple([words[i]])
            if context in model_7.ngrams:
                known_7 += 1
    
    coverage_5 = known_5 / total_5 if total_5 > 0 else 0
    coverage_7 = known_7 / total_7 if total_7 > 0 else 0
    
    return coverage_5, coverage_7


def main():
    print("=" * 60)
    print("HAIKU GENERATOR EVALUATION (70/30 Train/Test Split)")
    print("=" * 60)
    
    # Split dataset
    print("\nSplitting dataset into 70% train, 30% test...")
    train_haikus, test_haikus = split_dataset('dataset.csv')
    print(f"Training set: {len(train_haikus)} haikus")
    print(f"Test set: {len(test_haikus)} haikus")
    
    # Train models on training set only
    print("\nTraining models on training set...")
    model_5, model_7 = train_models_on_subset(train_haikus)
    
    # Create a generator and replace its models with our trained ones
    print("\nInitializing generator with trained models...")
    generator = HaikuGenerator('dataset.csv', pre_train_keywords=False)
    generator.model_5 = model_5
    generator.model_7 = model_7
    
    # Evaluate test set coverage (how well training data covers test data)
    print("\nEvaluating test set coverage...")
    coverage_5, coverage_7 = evaluate_test_set_coverage(model_5, model_7, test_haikus)
    print(f"5-syllable model coverage on test set: {coverage_5*100:.1f}%")
    print(f"7-syllable model coverage on test set: {coverage_7*100:.1f}%")
    avg_coverage = (coverage_5 + coverage_7) / 2
    print(f"Average coverage: {avg_coverage*100:.1f}%")
    
    # Evaluate syllable accuracy
    print("\nEvaluating syllable accuracy...")
    syllable_acc = evaluate_syllable_accuracy(generator)
    print(f"Syllable accuracy: {syllable_acc*100:.1f}%")
    
    # Evaluate diversity
    print("\nEvaluating diversity...")
    diversity = evaluate_diversity(generator)
    print(f"Diversity (unique haikus): {diversity*100:.1f}%")
    
    # Evaluate keyword relevance
    print("\nEvaluating keyword relevance...")
    kw_sample = random.sample(default_keywords, 5)
    kw_df = evaluate_keyword_relevance(generator, kw_sample)
    print(kw_df)
    
    # Save results to text files
    kw_df.to_csv('keyword_relevance_results.csv', index=False)
    with open('evaluation_summary.txt', 'w') as f:
        f.write("EVALUATION RESULTS (70/30 Train/Test Split)\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Training set size: {len(train_haikus)} haikus\n")
        f.write(f"Test set size: {len(test_haikus)} haikus\n\n")
        f.write(f"Test set coverage (5-syl model): {coverage_5*100:.1f}%\n")
        f.write(f"Test set coverage (7-syl model): {coverage_7*100:.1f}%\n")
        f.write(f"Average coverage: {avg_coverage*100:.1f}%\n\n")
        f.write(f"Syllable accuracy: {syllable_acc*100:.1f}%\n")
        f.write(f"Diversity: {diversity*100:.1f}%\n\n")
        f.write("Keyword relevance (sample):\n")
        f.write(kw_df.to_string(index=False))
    
    # =====================
    # Generate Charts/Graphs
    # =====================
    print("\nGenerating charts and graphs...")
    
    # 1. Overall Metrics Bar Chart
    plt.figure(figsize=(8, 5))
    metrics = ['Syllable\nAccuracy', 'Diversity', 'Avg Test\nCoverage']
    values = [syllable_acc * 100, diversity * 100, avg_coverage * 100]
    colors = ['#4F81BD', '#C0504D', '#9BBB59']
    bars = plt.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)
    plt.ylim(0, 110)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Haiku Generator Evaluation Metrics\n(70/30 Train/Test Split)', fontsize=14)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('overall_metrics.png', dpi=150)
    plt.close()
    print("  Saved: overall_metrics.png")
    
    # 2. Test Set Coverage Comparison (5-syl vs 7-syl)
    plt.figure(figsize=(6, 5))
    coverage_labels = ['5-Syllable Model', '7-Syllable Model']
    coverage_values = [coverage_5 * 100, coverage_7 * 100]
    colors = ['#5B9BD5', '#ED7D31']
    bars = plt.bar(coverage_labels, coverage_values, color=colors, edgecolor='black', linewidth=1.2)
    plt.ylim(0, 110)
    plt.ylabel('Coverage (%)', fontsize=12)
    plt.title('Test Set Coverage by Model Type', fontsize=14)
    for bar, val in zip(bars, coverage_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('coverage_comparison.png', dpi=150)
    plt.close()
    print("  Saved: coverage_comparison.png")
    
    # 3. Keyword Relevance Bar Chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(kw_df['keyword'], kw_df['relevance'] * 100, color='#70AD47', edgecolor='black', linewidth=1.2)
    plt.ylim(0, 110)
    plt.ylabel('Relevance (%)', fontsize=12)
    plt.xlabel('Keyword', fontsize=12)
    plt.title('Keyword Relevance (Sample of 5 Keywords)', fontsize=14)
    for bar, val in zip(bars, kw_df['relevance'] * 100):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.0f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('keyword_relevance.png', dpi=150)
    plt.close()
    print("  Saved: keyword_relevance.png")
    
    # 4. Train/Test Split Pie Chart
    plt.figure(figsize=(6, 6))
    sizes = [len(train_haikus), len(test_haikus)]
    labels = [f'Training Set\n({len(train_haikus)} haikus)', f'Test Set\n({len(test_haikus)} haikus)']
    colors = ['#4472C4', '#ED7D31']
    explode = (0.02, 0.02)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 11})
    plt.title('Dataset Split (70/30)', fontsize=14)
    plt.tight_layout()
    plt.savefig('dataset_split.png', dpi=150)
    plt.close()
    print("  Saved: dataset_split.png")
    
    print("\nResults saved to 'evaluation_summary.txt' and 'keyword_relevance_results.csv'.")
    print("Charts saved: overall_metrics.png, coverage_comparison.png, keyword_relevance.png, dataset_split.png")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot evaluation results for Haiku Generator
"""
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics():
    # Load summary
    with open('evaluation_summary.txt') as f:
        lines = f.readlines()
    syllable_acc = float(lines[0].split(':')[1].replace('%','').strip())
    diversity = float(lines[1].split(':')[1].replace('%','').strip())

    # Bar plot for overall metrics
    plt.figure(figsize=(5,4))
    plt.bar(['Syllable Accuracy', 'Diversity'], [syllable_acc, diversity], color=['#4F81BD', '#C0504D'])
    plt.ylim(0, 100)
    plt.ylabel('Percentage')
    plt.title('Haiku Generator Evaluation Metrics')
    plt.tight_layout()
    plt.savefig('overall_metrics.png')
    plt.close()

    # Keyword relevance plot
    df = pd.read_csv('keyword_relevance_results.csv')
    plt.figure(figsize=(7,4))
    plt.bar(df['keyword'], df['relevance']*100, color='#9BBB59')
    plt.ylim(0, 110)
    plt.ylabel('Relevance (%)')
    plt.title('Keyword Relevance (Sample)')
    plt.tight_layout()
    plt.savefig('keyword_relevance.png')
    plt.close()

if __name__ == "__main__":
    plot_metrics()

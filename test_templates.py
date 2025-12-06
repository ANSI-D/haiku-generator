#!/usr/bin/env python3
"""Quick test of template-based generation"""

from haiku_generator import HaikuGenerator

print("Initializing generator with templates...")
gen = HaikuGenerator('dataset.csv', pre_train_keywords=False)

print(f"\nTemplates extracted:")
print(f"  5-syllable: {len(gen.templates_5)}")
print(f"  7-syllable: {len(gen.templates_7)}")

if len(gen.templates_5) > 0:
    print(f"\nExample 5-syllable template: {gen.templates_5[0][:3]}")
if len(gen.templates_7) > 0:
    print(f"Example 7-syllable template: {gen.templates_7[0][:3]}")

print("\nGenerating 3 haikus...")
for i in range(3):
    haiku, score, breakdown = gen.generate_haiku(return_score=True)
    print(f"\n=== Haiku {i+1} ===")
    print(haiku)
    print(f"Score: {score:.1f}/100")

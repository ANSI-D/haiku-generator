#!/usr/bin/env python3
from haiku_generator import HaikuGenerator

print('\nInitializing without pre-training...')
g = HaikuGenerator('dataset.csv', pre_train_keywords=False)
print('\n--- Generating Test Haiku with DistilBERT ---')
h = g.generate_haiku(attempts=3)
print(h)
print('\nDone!')

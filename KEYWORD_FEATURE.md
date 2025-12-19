# Keyword-Based Haiku Generation Feature

## Overview
This feature allows users to generate haikus based on a specific keyword or theme. The implementation uses a hybrid approach combining keyword filtering and specialized model training.

## How It Works

### 1. Keyword Filtering
- Searches the dataset for haikus containing or related to the keyword
- Matches against both haiku text and the keyword field in the dataset
- Supports both single-word and multi-word keywords

### 2. Specialized Model Training
- If enough matching haikus are found (≥50), trains a specialized n-gram model
- Uses only haikus related to the keyword for more thematic coherence
- Falls back to main model with keyword seeding if insufficient data

### 3. Keyword Seeding
- Forces the keyword to appear in one of the three haiku lines
- Preferentially places keywords in 5-syllable lines (1st or 3rd)
- Builds the rest of the line around the keyword to match syllable count

## Usage

### In Code
```python
from haiku_generator import HaikuGenerator

generator = HaikuGenerator('dataset.csv')

# Generate a haiku with keyword
haiku = generator.generate_haiku_with_keyword('moon')
print(haiku)
```

### Interactive Mode
```bash
python haiku_generator.py
```
Then type a keyword when prompted, or press Enter for random generation.

### Keyword Demo Script
```bash
python keyword_demo.py
```

## Examples

### Keyword: "moon"
- Found 1724 matching haikus
- Trained specialized model
- Output example:
  ```
  No need ocean swell
  Moonlight comes up my problems
  Home is moon in the
  ```

### Keyword: "ocean"
- Found 140 matching haikus
- Trained specialized model
- Output example:
  ```
  Swells ocean sunrise
  The calm ocean surrounding
  Listening to the
  ```

### Keyword: "winter"
- Found 1203 matching haikus
- Trained specialized model
- Output example:
  ```
  Nights winter sun shines
  Darkest coldest time of the
  The slot deep winter
  ```

## Implementation Details

### Key Methods
- `_filter_haikus_by_keyword()`: Finds haikus matching the keyword
- `_train_keyword_models()`: Trains specialized models on filtered data
- `generate_line_with_keyword()`: Generates a line containing the keyword
- `generate_haiku_with_keyword()`: Main orchestration method

### Dataset Statistics
- Total haikus: 47,200
- Single-word keywords: 16,788 (35.6%)
- Multi-word keywords: 30,412 (64.4%)

## Files Modified/Added
- `haiku_generator.py`: Added keyword generation methods
- `keyword_demo.py`: New demo script for testing
- `KEYWORD_FEATURE.md`: This documentation

## Technical Notes
- Uses n-gram (n=2) models for generation
- Maintains 5-7-5 syllable structure
- Verifies keyword appears in final output
- Attempts up to 10 generations to find valid haiku

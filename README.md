# Haiku Generator - NLP Class Project

A functional Haiku generator that uses N-gram language models to create 5-7-5 syllable structured poems.

## Features

- **Syllable-Aware Generation**: Uses CMU Pronouncing Dictionary for accurate syllable counting
- **N-gram Language Model**: Trigram model trained on your haiku dataset
- **Separate Line Models**: Different models for 5-syllable and 7-syllable lines
- **Structure Validation**: Automatically verifies generated haikus follow 5-7-5 pattern
- **Interactive Mode**: Generate haikus on demand

## Project Structure

```
├── dataset.csv              # Your cleaned haiku dataset
├── haiku_generator.py       # Main generator script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The script will auto-download required NLTK data (CMUdict and punkt tokenizer) on first run.

## Usage

### Run the generator:
```bash
python haiku_generator.py
```

### Import as a module:
```python
from haiku_generator import HaikuGenerator

# Initialize
generator = HaikuGenerator('dataset.csv')

# Generate a single haiku
haiku = generator.generate_haiku()
print(haiku)

# Generate multiple haikus
haikus = generator.generate_multiple_haikus(count=10)

# Verify haiku structure
is_valid, message = generator.verify_haiku_structure(haiku)
```

## How It Works

### 1. Data Processing
- Loads haikus from CSV dataset
- Splits haikus into lines (5-7-5 syllable structure)
- Separates lines by syllable count for targeted training

### 2. N-gram Model Training
- **Trigram Model (n=3)**: Learns word sequences from training data
- **Two Separate Models**:
  - 5-syllable line model (trained on 1st and 3rd lines)
  - 7-syllable line model (trained on 2nd lines)
- Stores word transition probabilities

### 3. Syllable Counting
- **Primary**: CMU Pronouncing Dictionary (accurate phoneme-based)
- **Fallback**: Heuristic vowel-group counting for unknown words
- Ensures generated lines match target syllable counts

### 4. Generation Process
- Starts with random seed word from training data
- Generates words sequentially using n-gram probabilities
- Continuously checks syllable count
- Backtracks if syllable target exceeded
- Returns complete 5-7-5 haiku

## Performance

- **Training Time**: ~2-5 seconds for ~47K haikus
- **Generation Time**: ~0.1-0.5 seconds per haiku
- **CPU-Based**: Traditional NLP methods don't benefit from GPU acceleration
- **Memory Efficient**: Uses simple dictionaries and counters

## Examples of generated Haikus

```
Haiku #1:
The squeak of paper
Winds blow off my voice tune of
Almost forgot ghost

Haiku #2:
Only if you go
Under the practice room glows
I have gotten so
```
(May not be as good as the advertised ones)

## Limitations

- Generated haikus may lack deep semantic meaning
- Word choice depends on training data quality
- No guarantee of grammatical correctness
- Some generated combinations may be unusual

## Improvements for Future

- Add POS tagging for better grammar
- Implement semantic coherence checks
- Add theme/keyword seeding
- Use more sophisticated models (RNN/LSTM/Transformer)
- Add rhyme and meter constraints

## Technical Details

### N-gram Model
- **Type**: Trigram (3-gram)
- **Context Window**: 2 previous words predict next word
- **Probability**: Frequency-based word selection
- **Smoothing**: Random selection within probability distribution

### Syllable Counter
- **Primary Method**: CMU Pronouncing Dictionary phoneme counting
- **Fallback**: Vowel cluster heuristic
- **Accuracy**: ~95% with CMUdict, ~85% with heuristic

## Under the hood

This project utilizes:
- N-gram language modeling
- Probabilistic text generation
- Syllable analysis and phonetics
- Constraint-based generation
- Traditional NLP techniques


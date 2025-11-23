#!/usr/bin/env python3
"""
Haiku Generator using N-gram Language Model
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import cmudict
from collections import defaultdict, Counter
import random
import re
import string

# Download required NLTK data
try:
    nltk.data.find('corpora/cmudict.zip')
except LookupError:
    print("Downloading CMU Pronouncing Dictionary...")
    nltk.download('cmudict')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt tokenizer...")
    nltk.download('punkt')


class SyllableCounter:
    """Handles syllable counting for words using CMUdict and fallback heuristics."""
    
    def __init__(self):
        self.cmudict = cmudict.dict()
    
    def count_syllables(self, word):
        """
        Count syllables in a word.
        Uses CMUdict first, then falls back to heuristic counting.
        """
        word = word.lower().strip(string.punctuation)
        
        # Try CMUdict first
        if word in self.cmudict:
            # Get the first pronunciation
            phonemes = self.cmudict[word][0]
            # Count stress markers (0, 1, 2) which indicate syllables
            return len([p for p in phonemes if p[-1].isdigit()])
        
        # Fallback heuristic
        return self._heuristic_syllable_count(word)
    
    def _heuristic_syllable_count(self, word):
        """
        Heuristic syllable counting for words not in CMUdict.
        Based on vowel groups.
        """
        if not word:
            return 0
        
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent "e" at the end
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        # Make sure we return at least 1
        return max(1, syllable_count)
    
    def count_syllables_in_phrase(self, phrase):
        """Count total syllables in a phrase."""
        words = phrase.lower().split()
        return sum(self.count_syllables(word) for word in words)


class NGramModel:
    """N-gram language model for text generation."""
    
    def __init__(self, n=2):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.start_words = []
        self.all_words = set()
    
    def train(self, texts):
        """
        Train the n-gram model on a list of text strings.
        
        Args:
            texts: List of text strings (haiku lines)
        """
        for text in texts:
            words = self._tokenize(text)
            
            # Store all words
            self.all_words.update(words)
            
            # Store possible start words
            if len(words) > 0:
                self.start_words.append(words[0])
            
            # Build n-grams
            for i in range(len(words) - self.n + 1):
                gram = tuple(words[i:i+self.n-1])
                next_word = words[i+self.n-1]
                self.ngrams[gram][next_word] += 1
    
    def _tokenize(self, text):
        """Tokenize text into words, removing all punctuation."""
        # Remove line separators
        text = text.replace(' / ', ' ')
        # Simple tokenization
        words = text.split()
        # Remove punctuation from each word
        cleaned_words = []
        for word in words:
            # Strip all punctuation from word
            cleaned = word.lower().strip(string.punctuation)
            # Remove punctuation from inside brackets too
            cleaned = re.sub(r'[\[\]\(\)\{\}]', '', cleaned)
            if cleaned:  # Only add non-empty words
                cleaned_words.append(cleaned)
        return cleaned_words
    
    def generate_next_word(self, context):
        """
        Generate the next word given a context.
        
        Args:
            context: Tuple of previous words
            
        Returns:
            Next word or None if context not found
        """
        context = tuple(context)
        if context not in self.ngrams:
            return None
        
        # Get possible next words and their frequencies
        next_words = self.ngrams[context]
        words = list(next_words.keys())
        frequencies = list(next_words.values())
        
        # Convert frequencies to probabilities
        total = sum(frequencies)
        probabilities = [f / total for f in frequencies]
        
        # Sample a word based on probabilities
        return np.random.choice(words, p=probabilities)
    
    def get_random_start_word(self):
        """Get a random starting word."""
        if not self.start_words:
            return None
        return random.choice(self.start_words)


class HaikuGenerator:
    """Main Haiku Generator class."""
    
    def __init__(self, dataset_path='dataset.csv'):
        print("Initializing Haiku Generator...")
        self.syllable_counter = SyllableCounter()
        self.dataset_path = dataset_path
        
        # Models for each line type (5, 7, 5 syllables)
        # Using bigrams (n=2) for better coherence
        self.model_5 = NGramModel(n=2)
        self.model_7 = NGramModel(n=2)
        
        print("Loading and processing dataset...")
        self.load_and_train()
        print("Training complete!")
    
    def load_and_train(self):
        """Load dataset and train the models."""
        # Load the CSV
        df = pd.read_csv(self.dataset_path)
        
        # Extract haiku text column
        haikus = df['text'].dropna().tolist()
        
        print(f"Loaded {len(haikus)} haikus from dataset")
        
        # Split haikus into lines
        lines_5_first = []
        lines_7 = []
        lines_5_last = []
        
        for haiku in haikus:
            # Split by ' / ' separator
            lines = [line.strip() for line in haiku.split(' / ')]
            
            if len(lines) >= 3:
                lines_5_first.append(lines[0])
                lines_7.append(lines[1])
                lines_5_last.append(lines[2])
        
        print(f"Processing {len(lines_5_first)} haiku structures...")
        
        # Train separate models for 5 and 7 syllable lines
        # Combine first and last lines for 5-syllable model
        all_5_syllable_lines = lines_5_first + lines_5_last
        
        print("Training 5-syllable line model...")
        self.model_5.train(all_5_syllable_lines)
        
        print("Training 7-syllable line model...")
        self.model_7.train(lines_7)
    
    def generate_line(self, target_syllables, model, max_attempts=200):
        """
        Generate a single line with the target syllable count.
        
        Args:
            target_syllables: Target number of syllables (5 or 7)
            model: The n-gram model to use
            max_attempts: Maximum number of generation attempts
            
        Returns:
            Generated line or None if failed
        """
        for attempt in range(max_attempts):
            # Start with a random word
            start_word = model.get_random_start_word()
            if not start_word:
                continue
            
            line = [start_word]
            current_syllables = self.syllable_counter.count_syllables(start_word)
            
            # If start word already equals target, we're done
            if current_syllables == target_syllables:
                return ' '.join(line)
            
            # If start word exceeds target, try again
            if current_syllables > target_syllables:
                continue
            
            # Keep adding words until we reach target syllables
            max_words = 20  # Prevent infinite loops (because a single line attempt cannot run more than 20 times, preventing an infinite loop.)
            context = [start_word]
            stuck_count = 0
            
            for _ in range(max_words):
                remaining_syllables = target_syllables - current_syllables
                
                if remaining_syllables == 0:
                    break
                
                # Generate next word
                next_word = model.generate_next_word(context[-(model.n-1):])
                
                if next_word is None:
                    # Try with shorter context
                    if len(context) > 1:
                        next_word = model.generate_next_word(context[-1:])
                    
                    if next_word is None:
                        stuck_count += 1
                        if stuck_count > 3:
                            break
                        # Get any random word from the model
                        next_word = model.get_random_start_word()
                        if next_word is None:
                            break
                
                # Check if adding this word would work
                word_syllables = self.syllable_counter.count_syllables(next_word)
                
                if word_syllables == remaining_syllables:
                    # Perfect fit!
                    line.append(next_word)
                    current_syllables += word_syllables
                    break
                elif word_syllables < remaining_syllables:
                    # Can add this word
                    line.append(next_word)
                    current_syllables += word_syllables
                    context.append(next_word)
                    stuck_count = 0
                else:
                    # Word is too big, try a few more times
                    stuck_count += 1
                    if stuck_count > 5:
                        break
            
            # Check if we hit the target exactly
            final_syllables = sum(self.syllable_counter.count_syllables(w) for w in line)
            if final_syllables == target_syllables and len(line) > 0:
                return ' '.join(line)
        
        return None
    
    def _clean_line(self, line):
        """Clean up and format a haiku line."""
        if not line:
            return line
        
        # Remove extra spaces
        line = ' '.join(line.split())
        
        # Remove any remaining punctuation
        line = line.translate(str.maketrans('', '', string.punctuation))
        
        # Capitalize first letter
        line = line[0].upper() + line[1:] if len(line) > 1 else line.upper()
        
        return line
    
    def generate_haiku(self, attempts=10):
        """
        Generate a complete haiku.
        
        Args:
            attempts: Number of attempts to generate a valid haiku
            
        Returns:
            Formatted haiku string or error message
        """
        for attempt in range(attempts):
            # Generate three lines: 5-7-5
            line1 = self.generate_line(5, self.model_5)
            line2 = self.generate_line(7, self.model_7)
            line3 = self.generate_line(5, self.model_5)
            
            if line1 and line2 and line3:
                # Clean and format each line (removes punctuation)
                line1 = self._clean_line(line1)
                line2 = self._clean_line(line2)
                line3 = self._clean_line(line3)
                
                # Format the haiku (no punctuation added)
                haiku = f"{line1}\n{line2}\n{line3}"
                return haiku
        
        return "Failed to generate a valid haiku. Please try again."
    
    def generate_multiple_haikus(self, count=5):
        """Generate multiple haikus."""
        haikus = []
        for i in range(count):
            print(f"\nGenerating haiku {i+1}/{count}...")
            haiku = self.generate_haiku()
            haikus.append(haiku)
        return haikus
    
    def verify_haiku_structure(self, haiku):
        """Verify that a haiku follows the 5-7-5 structure."""
        lines = haiku.strip().split('\n')
        if len(lines) != 3:
            return False, "Must have exactly 3 lines"
        
        syllables = [self.syllable_counter.count_syllables_in_phrase(line) for line in lines]
        expected = [5, 7, 5]
        
        if syllables == expected:
            return True, "Valid haiku structure!"
        else:
            return False, f"Syllable count is {syllables}, expected [5, 7, 5]"


def main():
    """Main function to demonstrate the haiku generator."""
    print("=" * 60)
    print("HAIKU GENERATOR - NLP Class Project")
    print("=" * 60)
    print()
    
    # Initialize the generator
    generator = HaikuGenerator('dataset.csv')
    
    print("\n" + "=" * 60)
    print("GENERATING HAIKUS")
    print("=" * 60)
    
    # Generate multiple haikus
    num_haikus = 5
    haikus = generator.generate_multiple_haikus(num_haikus)
    
    print("\n" + "=" * 60)
    print("GENERATED HAIKUS")
    print("=" * 60)
    
    for i, haiku in enumerate(haikus, 1):
        print(f"\nHaiku #{i}:")
        print("-" * 40)
        print(haiku)
        
        # Verify structure
        is_valid, message = generator.verify_haiku_structure(haiku)
        print(f"Verification: {message}")
        print("-" * 40)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("\nPress Enter to generate a new haiku, or type 'quit' to exit.")
    
    while True:
        user_input = input("\n> ").strip().lower()
        if user_input in ['quit', 'exit', 'q']:
            print("Thanks for using the Haiku Generator!")
            break
        
        haiku = generator.generate_haiku()
        print("\n" + haiku)
        is_valid, message = generator.verify_haiku_structure(haiku)
        print(f"\n{message}")


if __name__ == "__main__":
    main()

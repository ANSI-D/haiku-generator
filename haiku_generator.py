#!/usr/bin/env python3
"""
Haiku Generator using N-gram Language Model
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import cmudict, stopwords
from collections import defaultdict, Counter
import random
import re
import string
from functools import lru_cache
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords')

# Load English stopwords
STOPWORDS = set(stopwords.words('english'))


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
    
    def __init__(self, dataset_path='dataset.csv', pre_train_keywords=True, model_cache_dir='models'):
        print("Initializing Haiku Generator...")
        self.syllable_counter = SyllableCounter()
        self.dataset_path = dataset_path
        self.model_cache_dir = model_cache_dir
        
        # Create model cache directory if it doesn't exist
        os.makedirs(model_cache_dir, exist_ok=True)
        
        # Models for each line type (5, 7, 5 syllables)
        # Using bigrams (n=2) for better coherence
        self.model_5 = NGramModel(n=2)
        self.model_7 = NGramModel(n=2)
        
        # Store the full dataset for keyword filtering
        self.df = None
        
        # Cache for keyword-specific models
        self.keyword_model_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Top keywords to pre-train (50+ occurrences)
        self.top_keywords = [
            'rain', 'moon', 'summer', 'morning', 'winter', 'snow',
            'autumn', 'wind', 'night', 'leaves', 'spring', 'sky',
            'sun', 'clouds', 'birthday', 'haiku', 'shadow', 'tree',
            'love', 'water', 'fall', 'day', 'light'
        ]
        
        # TF-IDF vectorizer for quality scoring
        self.tfidf_vectorizer = None
        self.training_vectors = None
        
        # Try to load models from cache
        if self._load_models():
            print("Loaded models from cache!")
            self._setup_tfidf_scorer()
        else:
            print("Loading and processing dataset...")
            self.load_and_train()
            print("Training complete!")
            self._setup_tfidf_scorer()
            self._save_models()
            
            # Pre-train models for common keywords
            if pre_train_keywords:
                self._pre_train_common_keywords()
                self._save_keyword_models()
    
    def _save_models(self):
        """Save trained models to disk."""
        model_file = os.path.join(self.model_cache_dir, 'main_models.pkl')
        print(f"Saving models to {model_file}...")
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model_5': self.model_5,
                'model_7': self.model_7,
                'df': self.df
            }, f)
        print("Models saved!")
    
    def _load_models(self):
        """Load trained models from disk."""
        model_file = os.path.join(self.model_cache_dir, 'main_models.pkl')
        if not os.path.exists(model_file):
            return False
        
        print(f"Loading models from {model_file}...")
        try:
            with open(model_file, 'rb') as f:
                data = pickle.load(f)
                self.model_5 = data['model_5']
                self.model_7 = data['model_7']
                self.df = data['df']
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def _save_keyword_models(self):
        """Save keyword models to disk."""
        keyword_file = os.path.join(self.model_cache_dir, 'keyword_models.pkl')
        print(f"Saving keyword models to {keyword_file}...")
        with open(keyword_file, 'wb') as f:
            pickle.dump(self.keyword_model_cache, f)
        print(f"Saved {len(self.keyword_model_cache)} keyword models!")
    
    def _load_keyword_models(self):
        """Load keyword models from disk."""
        keyword_file = os.path.join(self.model_cache_dir, 'keyword_models.pkl')
        if not os.path.exists(keyword_file):
            return False
        
        print(f"Loading keyword models from {keyword_file}...")
        try:
            with open(keyword_file, 'rb') as f:
                self.keyword_model_cache = pickle.load(f)
            print(f"Loaded {len(self.keyword_model_cache)} keyword models from cache!")
            return True
        except Exception as e:
            print(f"Error loading keyword models: {e}")
            return False
    
    def _setup_tfidf_scorer(self):
        """Setup TF-IDF vectorizer for quality scoring."""
        print("Setting up quality scorer...")
        
        # Collect all lines from dataset for TF-IDF training
        all_training_lines = []
        for _, row in self.df.iterrows():
            lines = [line.strip() for line in row['text'].split(' / ')]
            all_training_lines.extend(lines)
        
        # Create and fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.training_vectors = self.tfidf_vectorizer.fit_transform(all_training_lines)
        print(f"Quality scorer ready with {len(all_training_lines)} training lines")
    
    def load_and_train(self):
        """Load dataset and train the models."""
        # Load the CSV
        self.df = pd.read_csv(self.dataset_path)
        
        # Extract haiku text column
        haikus = self.df['text'].dropna().tolist()
        
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
    
    def score_line_quality(self, line):
        """
        Score the quality of a generated line (0-100 scale).
        Higher score = better quality.
        """
        if not line or not line.strip():
            return 0
        
        score = 50.0  # Start at neutral
        words = line.lower().split()
        
        # CRITICAL: Detect nonsensical patterns
        # Single letter words (except 'a' and 'i') are almost always errors
        for word in words:
            if len(word) == 1 and word not in ['a', 'i']:
                score -= 40  # Heavy penalty for garbage like "B"
        
        # Penalty for word repetition
        unique_words = len(set(words))
        if len(words) > 0 and unique_words < len(words):
            repetition_ratio = 1 - (unique_words / len(words))
            score -= 30 * repetition_ratio
        
        # Penalty for very short lines (single word)
        if len(words) < 2:
            score -= 25
        elif len(words) == 2:
            score -= 10  # Slight penalty for very short
        
        # Penalty for overly long lines (too many words)
        if len(words) > 8:
            score -= 15
        
        # Check for reasonable word lengths (not all tiny or all huge)
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len < 2.5:  # Mostly tiny words
                score -= 15
            elif avg_word_len > 9:  # Mostly huge words
                score -= 10
        
        # Coherence check via TF-IDF similarity to training data
        if self.tfidf_vectorizer is not None and self.training_vectors is not None:
            try:
                line_vector = self.tfidf_vectorizer.transform([line])
                similarities = cosine_similarity(line_vector, self.training_vectors)
                avg_similarity = similarities.mean()
                
                # Similarity-based scoring (weighted towards coherence, not just vocab matching)
                # Average similarity of 0.1 = +10 points, 0.2 = +20, etc.
                coherence_score = min(40, avg_similarity * 200)  # Cap at 40 points
                score += coherence_score
            except:
                score -= 10  # Penalty if TF-IDF fails (likely garbage text)
        
        # Penalty if last word is a stopword
        if len(words) > 0 and words[-1] in STOPWORDS:
            score -= 35
        
        # Check for grammar issues: repeated punctuation, weird capitalization patterns
        if any(word.isupper() and len(word) > 1 for word in line.split()):
            score -= 10  # ALL CAPS words
        
        # Ensure score stays in 0-100 range
        return max(0, min(100, score))
    
    def score_haiku_quality(self, haiku):
        """
        Score the overall quality of a complete haiku (0-100 scale).
        Returns (score, breakdown) tuple where score is 0-100.
        """
        lines = haiku.strip().split('\n')
        if len(lines) != 3:
            return 0, {"error": "Invalid haiku structure"}
        
        # Score each line individually (each line is already capped at 0-100)
        line_scores = [self.score_line_quality(line) for line in lines]
        avg_line_score = sum(line_scores) / len(line_scores)
        
        # Start with average line quality (0-100)
        overall_score = avg_line_score
        
        # Check for cross-line coherence issues
        all_words = []
        for line in lines:
            all_words.extend(line.lower().split())
        
        # Penalize if any line has extremely low score (indicates garbage)
        min_line_score = min(line_scores)
        if min_line_score < 30:
            overall_score *= 0.7  # Reduce overall score if any line is terrible
        
        # Variety bonus (but don't let it push score over 100)
        unique_ratio = len(set(all_words)) / len(all_words) if all_words else 0
        variety_bonus = unique_ratio * 8  # Reduced from 10
        
        # Detect nonsensical haikus: check for single-letter words across all lines
        single_letter_count = sum(1 for w in all_words if len(w) == 1 and w not in ['a', 'i'])
        if single_letter_count > 0:
            overall_score -= 25 * single_letter_count  # Heavy penalty
        
        # Check overall haiku coherence using TF-IDF (weight coherence more heavily)
        coherence_bonus = 0
        if self.tfidf_vectorizer is not None:
            try:
                full_text = ' '.join(lines)
                haiku_vector = self.tfidf_vectorizer.transform([full_text])
                similarities = cosine_similarity(haiku_vector, self.training_vectors)
                avg_sim = similarities.mean()
                
                # Coherence contributes up to 20 points
                coherence_bonus = min(20, avg_sim * 100)
            except:
                coherence_bonus = -5  # Penalty if TF-IDF fails
        
        # Combine scores (line quality is primary, coherence and variety are secondary)
        final_score = overall_score * 0.75 + coherence_bonus + variety_bonus
        
        # Ensure final score is in 0-100 range
        final_score = min(100, max(0, final_score))
        
        breakdown = {
            'line_scores': [round(s, 1) for s in line_scores],
            'avg_line_score': round(avg_line_score, 1),
            'variety_bonus': round(variety_bonus, 1),
            'coherence_bonus': round(coherence_bonus, 1),
            'final_score': round(final_score, 1)
        }
        
        return final_score, breakdown
    
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
                # Check if last word is a stopword - if so, reject this line
                last_word = line[-1].lower()
                if last_word not in STOPWORDS:
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
    
    def generate_haiku(self, attempts=10, return_score=False, temperature=0.4):
        """
        Generate a complete haiku. Generates multiple candidates and picks the best.
        Stanford research shows temperature 0-0.4 produces best quality haikus.
        
        Args:
            attempts: Number of haiku candidates to generate
            return_score: If True, returns (haiku, score, breakdown) tuple
            temperature: Controls randomness (0=deterministic, 0.4=default/balanced)
            
        Returns:
            Formatted haiku string, or tuple with (haiku, score, breakdown) if return_score=True
        """
        candidates = []
        
        for attempt in range(attempts):
            # Generate three lines: 5-7-5 with temperature control
            line1 = self.generate_line(5, self.model_5, temperature=temperature)
            line2 = self.generate_line(7, self.model_7, temperature=temperature)
            line3 = self.generate_line(5, self.model_5, temperature=temperature)
            
            if line1 and line2 and line3:
                # Clean and format each line (removes punctuation)
                line1 = self._clean_line(line1)
                line2 = self._clean_line(line2)
                line3 = self._clean_line(line3)
                
                # Format the haiku (no punctuation added)
                haiku = f"{line1}\n{line2}\n{line3}"
                
                # Score the haiku
                score, breakdown = self.score_haiku_quality(haiku)
                candidates.append((haiku, score, breakdown))
        
        if candidates:
            # Return the best scoring haiku
            best = max(candidates, key=lambda x: x[1])
            if return_score:
                return best
            else:
                return best[0]
        
        if return_score:
            return "Failed to generate a valid haiku. Please try again.", 0, {}
        return "Failed to generate a valid haiku. Please try again."
    
    def generate_multiple_haikus(self, count=5):
        """Generate multiple haikus."""
        haikus = []
        for i in range(count):
            print(f"\nGenerating haiku {i+1}/{count}...")
            haiku = self.generate_haiku()
            haikus.append(haiku)
        return haikus
    
    def get_cache_stats(self):
        """Get statistics about the keyword model cache."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cached_models': len(self.keyword_model_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
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
    
    def _pre_train_common_keywords(self):
        """Pre-train models for the most common keywords."""
        # Try to load from cache first
        if self._load_keyword_models():
            return
        
        print("\nPre-training models for common keywords...")
        successful = 0
        
        for keyword in self.top_keywords:
            filtered_haikus = self._filter_haikus_by_keyword(keyword, min_samples=50, verbose=False)
            if filtered_haikus:
                model_5, model_7 = self._train_keyword_models(filtered_haikus)
                self.keyword_model_cache[keyword.lower()] = (model_5, model_7)
                successful += 1
        
        print(f"Pre-trained {successful}/{len(self.top_keywords)} keyword models")
        print(f"Cache size: ~{successful * 2} MB")
    
    def _normalize_text(self, text):
        """Normalize text for keyword matching."""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def _filter_haikus_by_keyword(self, keyword, min_samples=50, verbose=True):
        """Filter haikus that contain or relate to the given keyword."""
        keyword_normalized = self._normalize_text(keyword)
        keyword_words = set(keyword_normalized.split())
        
        # Find haikus containing the keyword
        matching_haikus = []
        
        for _, row in self.df.iterrows():
            haiku_text = self._normalize_text(row['text'])
            haiku_keyword = self._normalize_text(str(row['keywords']))
            
            # Check if keyword appears in haiku text or keyword field
            if keyword_normalized in haiku_text or keyword_normalized in haiku_keyword:
                matching_haikus.append(row['text'])
            # Also check for partial word matches
            elif any(word in haiku_text or word in haiku_keyword for word in keyword_words):
                matching_haikus.append(row['text'])
        
        if verbose:
            print(f"Found {len(matching_haikus)} haikus matching '{keyword}'")
        
        if len(matching_haikus) >= min_samples:
            return matching_haikus
        else:
            return None
    
    def _train_keyword_models(self, haikus):
        """Train specialized models on filtered haikus."""
        # Split haikus into lines
        lines_5_first = []
        lines_7 = []
        lines_5_last = []
        
        for haiku in haikus:
            lines = [line.strip() for line in haiku.split(' / ')]
            
            if len(lines) >= 3:
                lines_5_first.append(lines[0])
                lines_7.append(lines[1])
                lines_5_last.append(lines[2])
        
        # Create temporary models
        temp_model_5 = NGramModel(n=2)
        temp_model_7 = NGramModel(n=2)
        
        # Train on filtered data
        all_5_syllable_lines = lines_5_first + lines_5_last
        temp_model_5.train(all_5_syllable_lines)
        temp_model_7.train(lines_7)
        
        return temp_model_5, temp_model_7
    
    def generate_line_with_keyword(self, target_syllables, model, keyword, max_attempts=200, num_candidates=10):
        """Generate a line containing the keyword with target syllable count.
        Uses multiple candidates and quality scoring.
        """
        candidates = []
        attempts_per_candidate = max(20, max_attempts // num_candidates)
        
        # Generate multiple candidate lines
        for _ in range(num_candidates):
            line = self._generate_single_line_with_keyword(target_syllables, model, keyword, attempts_per_candidate)
            if line:
                score = self.score_line_quality(line)
                candidates.append((line, score))
        
        # Return the best candidate
        if candidates:
            best_line = max(candidates, key=lambda x: x[1])
            return best_line[0]
        
        return None
    
    def _generate_single_line_with_keyword(self, target_syllables, model, keyword, max_attempts=20):
        """Generate a single line with keyword attempt (internal helper)."""
        keyword_words = keyword.lower().split()
        keyword_syllables = sum(self.syllable_counter.count_syllables(w) for w in keyword_words)
        
        # Check if keyword fits in the target syllable count
        if keyword_syllables > target_syllables:
            print(f"Warning: Keyword '{keyword}' has {keyword_syllables} syllables, more than target {target_syllables}")
            return None
        
        for attempt in range(max_attempts):
            # Start with the keyword
            line = keyword_words.copy()
            current_syllables = keyword_syllables
            remaining_syllables = target_syllables - current_syllables
            
            if remaining_syllables == 0:
                return ' '.join(line)
            
            # Decide whether to add words before or after the keyword
            add_before = random.choice([True, False])
            
            # Keep adding words until we reach target
            max_words = 20
            stuck_count = 0
            
            for _ in range(max_words):
                remaining_syllables = target_syllables - current_syllables
                
                if remaining_syllables == 0:
                    break
                
                # Get a random word that fits
                if add_before and len(line) > 0:
                    # Try to find a word that could come before
                    context = [line[0]] if len(line) > 0 else []
                else:
                    # Try to find a word that could come after
                    context = [line[-1]] if len(line) > 0 else []
                
                next_word = model.generate_next_word(context[-(model.n-1):]) if context else model.get_random_start_word()
                
                if next_word is None:
                    next_word = model.get_random_start_word()
                    if next_word is None:
                        stuck_count += 1
                        if stuck_count > 3:
                            break
                        continue
                
                word_syllables = self.syllable_counter.count_syllables(next_word)
                
                if word_syllables == remaining_syllables:
                    # Perfect fit
                    if add_before:
                        line.insert(0, next_word)
                    else:
                        line.append(next_word)
                    current_syllables += word_syllables
                    break
                elif word_syllables < remaining_syllables:
                    # Can add this word
                    if add_before:
                        line.insert(0, next_word)
                    else:
                        line.append(next_word)
                    current_syllables += word_syllables
                    stuck_count = 0
                    # Alternate direction
                    add_before = not add_before
                else:
                    stuck_count += 1
                    if stuck_count > 5:
                        break
            
            # Check if we hit the target
            final_syllables = sum(self.syllable_counter.count_syllables(w) for w in line)
            if final_syllables == target_syllables:
                # Check if last word is a stopword - if so, reject this line
                last_word = line[-1].lower()
                if last_word not in STOPWORDS:
                    return ' '.join(line)
        
        return None
    
    def generate_haiku_with_keyword(self, keyword, attempts=10, return_score=False):
        """Generate a haiku based on or containing the given keyword."""
        print(f"\nGenerating haiku with keyword: '{keyword}'")
        
        # Normalize keyword
        keyword_normalized = keyword.lower().strip()
        
        # Check cache first
        if keyword_normalized in self.keyword_model_cache:
            print(f"Using cached model for '{keyword}' (cache hit)")
            model_5, model_7 = self.keyword_model_cache[keyword_normalized]
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            # Try to filter haikus by keyword
            filtered_haikus = self._filter_haikus_by_keyword(keyword_normalized)
            
            # Determine which models to use
            if filtered_haikus:
                print(f"Training specialized model on {len(filtered_haikus)} related haikus...")
                model_5, model_7 = self._train_keyword_models(filtered_haikus)
                
                # Cache the models for future use (limit cache size to 50 keywords)
                if len(self.keyword_model_cache) < 50:
                    self.keyword_model_cache[keyword_normalized] = (model_5, model_7)
                    print(f"Model cached for future use")
            else:
                print(f"Not enough matching haikus, using main model with keyword seeding...")
                model_5, model_7 = self.model_5, self.model_7
        
        # Generate multiple haiku candidates with keyword
        candidates = []
        
        for attempt in range(attempts):
            # Randomly decide which line will contain the keyword
            # Prefer line 1 or 3 (5-syllable lines)
            keyword_line_position = random.choice([0, 0, 2, 2, 1])  # Weighted toward 5-syllable lines
            
            lines = [None, None, None]
            
            # Generate the line with the keyword
            if keyword_line_position in [0, 2]:
                lines[keyword_line_position] = self.generate_line_with_keyword(5, model_5, keyword_normalized)
            else:
                lines[keyword_line_position] = self.generate_line_with_keyword(7, model_7, keyword_normalized)
            
            # If keyword line generation failed, try again
            if lines[keyword_line_position] is None:
                continue
            
            # Generate the other lines normally
            for i in range(3):
                if lines[i] is None:
                    if i == 1:
                        lines[i] = self.generate_line(7, model_7)
                    else:
                        lines[i] = self.generate_line(5, model_5)
            
            # Check if all lines generated successfully
            if all(lines):
                # Clean and format
                lines = [self._clean_line(line) for line in lines]
                haiku = f"{lines[0]}\n{lines[1]}\n{lines[2]}"
                
                # Verify keyword is in the haiku
                haiku_normalized = self._normalize_text(haiku)
                if keyword_normalized in haiku_normalized:
                    # Score the haiku
                    score, breakdown = self.score_haiku_quality(haiku)
                    candidates.append((haiku, score, breakdown))
        
        if candidates:
            # Return the best scoring haiku
            best = max(candidates, key=lambda x: x[1])
            if return_score:
                return best
            else:
                return best[0]
        
        error_msg = f"Failed to generate a haiku with keyword '{keyword}'. Please try again or use a different keyword."
        if return_score:
            return error_msg, 0, {}
        return error_msg


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
    print("\nOptions:")
    print("  - Press Enter to generate a random haiku")
    print("  - Type a keyword to generate a haiku about that topic")
    print("  - Type 'quit' or 'exit' to quit")
    
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Thanks for using the Haiku Generator!")
            break
        
        if user_input == "":
            # Generate random haiku
            haiku = generator.generate_haiku()
        else:
            # Generate keyword-based haiku
            haiku = generator.generate_haiku_with_keyword(user_input)
        
        print("\n" + haiku)
        is_valid, message = generator.verify_haiku_structure(haiku)
        print(f"\n{message}")


if __name__ == "__main__":
    main()

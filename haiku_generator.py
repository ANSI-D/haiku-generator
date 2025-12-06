#!/usr/bin/env python3
"""
Haiku Generator using N-gram Language Model
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import cmudict, stopwords
from nltk import pos_tag, word_tokenize
from collections import defaultdict, Counter
import random
import re
import string
from functools import lru_cache
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# DistilBERT for masked language model coherence checking
try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    import torch
    print("Loading DistilBERT model for coherence checking...")
    bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = AutoModelForMaskedLM.from_pretrained('distilbert-base-uncased')
    bert_model.eval()
    print("DistilBERT loaded successfully")
except ImportError:
    print("transformers not installed. Install with: pip install transformers torch")
    bert_tokenizer = None
    bert_model = None

# spaCy for dependency parsing and advanced grammar checking
try:
    import spacy
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        nlp = None
except ImportError:
    print("spaCy not installed. Install with: pip install spacy")
    nlp = None

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

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading POS tagger...")
    nltk.download('averaged_perceptron_tagger')

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
                'df': self.df,
                'templates_5': self.templates_5,
                'templates_7': self.templates_7
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
                self.templates_5 = data.get('templates_5', [])
                self.templates_7 = data.get('templates_7', [])
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
        
        # Collect all lines from dataset for TF-IDF training - use vectorized operations!
        all_training_lines = []
        for text in self.df['text'].values:  # Much faster than iterrows()
            if pd.notna(text):  # Check for NaN
                lines = [line.strip() for line in text.split(' / ')]
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
        
        # Extract templates from training data
        print("Extracting grammatical templates from training data...")
        self._extract_templates(all_5_syllable_lines, lines_7)
    
    def _extract_templates(self, lines_5, lines_7):
        """
        Extract POS tag templates from training data for template-based generation.
        Stores common grammatical patterns for 5 and 7 syllable lines.
        """
        self.templates_5 = []
        self.templates_7 = []
        
        print("Analyzing 5-syllable line patterns...")
        for line in lines_5[:2000]:  # Sample more lines
            tokens = word_tokenize(line)
            if len(tokens) < 2 or len(tokens) > 8:
                continue
                
            pos_tags = pos_tag(tokens)
            
            # Store template: [(word, POS, syllables), ...]
            template = []
            total_syllables = 0
            for word, pos in pos_tags:
                syllables = self.syllable_counter.count_syllables(word)
                template.append((word.lower(), pos, syllables))
                total_syllables += syllables
            
            # Only keep templates that are actually 5 syllables and reasonable length
            if 2 <= len(template) <= 6 and 4 <= total_syllables <= 6:
                self.templates_5.append(template)
        
        print("Analyzing 7-syllable line patterns...")
        for line in lines_7[:2000]:
            tokens = word_tokenize(line)
            if len(tokens) < 2 or len(tokens) > 10:
                continue
                
            pos_tags = pos_tag(tokens)
            
            template = []
            total_syllables = 0
            for word, pos in pos_tags:
                syllables = self.syllable_counter.count_syllables(word)
                template.append((word.lower(), pos, syllables))
                total_syllables += syllables
            
            # Only keep templates that are actually 7 syllables
            if 2 <= len(template) <= 8 and 6 <= total_syllables <= 8:
                self.templates_7.append(template)
        
        print(f"Extracted {len(self.templates_5)} templates for 5-syllable lines")
        print(f"Extracted {len(self.templates_7)} templates for 7-syllable lines")
    
    def _bert_score_word_in_context(self, context_words, candidate_word):
        """
        Score how well a candidate word fits in the given context using DistilBERT.
        Returns a score (higher = better fit).
        """
        if bert_tokenizer is None or bert_model is None:
            return 0.5  # Neutral score
        
        try:
            # Create sentence with [MASK] where candidate would go
            if context_words:
                text = ' '.join(context_words) + ' [MASK]'
            else:
                text = '[MASK]'
            
            inputs = bert_tokenizer(text, return_tensors='pt')
            mask_token_index = torch.where(inputs['input_ids'][0] == bert_tokenizer.mask_token_id)[0]
            
            if len(mask_token_index) == 0:
                return 0.5
            
            with torch.no_grad():
                outputs = bert_model(**inputs)
                logits = outputs.logits
            
            mask_logits = logits[0, mask_token_index[0]]
            probs = torch.softmax(mask_logits, dim=0)
            
            # Get probability of candidate word
            candidate_tokens = bert_tokenizer.encode(candidate_word, add_special_tokens=False)
            if len(candidate_tokens) > 0:
                prob = probs[candidate_tokens[0]].item()
                return prob
            return 0.0
        except:
            return 0.5
    
    def _check_bert_coherence(self, line, threshold=-5.0):
        """
        Check line coherence using DistilBERT masked language model.
        Masks each word and checks if DistilBERT predicts something reasonable.
        Returns True if line is coherent (low perplexity).
        """
        if bert_tokenizer is None or bert_model is None:
            return True  # Skip if BERT not available
        
        try:
            words = line.split()
            if len(words) < 2:
                return True
            
            total_log_prob = 0
            count = 0
            
            # For each word, mask it and see if BERT thinks it's reasonable
            for i in range(len(words)):
                # Create masked version
                masked_words = words.copy()
                original_word = masked_words[i]
                masked_words[i] = '[MASK]'
                masked_text = ' '.join(masked_words)
                
                # Tokenize
                inputs = bert_tokenizer(masked_text, return_tensors='pt')
                mask_token_index = torch.where(inputs['input_ids'][0] == bert_tokenizer.mask_token_id)[0]
                
                if len(mask_token_index) == 0:
                    continue
                
                # Get BERT predictions
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    logits = outputs.logits
                
                # Get probability of original word
                mask_logits = logits[0, mask_token_index[0]]
                probs = torch.softmax(mask_logits, dim=0)
                
                # Get token ID of original word
                original_token_ids = bert_tokenizer.encode(original_word, add_special_tokens=False)
                if len(original_token_ids) > 0:
                    original_prob = probs[original_token_ids[0]].item()
                    if original_prob > 0:
                        total_log_prob += np.log(original_prob)
                        count += 1
            
            if count == 0:
                return True
            
            # Average log probability (higher = more coherent)
            avg_log_prob = total_log_prob / count
            return avg_log_prob > threshold  # Threshold around -5.0
        except:
            return True  # Skip check on error
    
    def _check_sentence_coherence(self, line):
        """
        Check sentence coherence using NLTK POS tagging and spaCy dependency parsing.
        Returns a score from 0-30 (higher = more coherent).
        """
        coherence_score = 15.0  # Start neutral
        
        # NLTK Part-of-Speech tagging
        try:
            tokens = word_tokenize(line)
            pos_tags = pos_tag(tokens)
            
            # Check for basic grammatical patterns
            pos_sequence = [tag for word, tag in pos_tags]
            
            # Penalty for broken patterns
            # Check for repeated determiners: "the the"
            for i in range(len(pos_sequence) - 1):
                if pos_sequence[i] == 'DT' and pos_sequence[i+1] == 'DT':
                    coherence_score -= 8
                # Check for verb followed by verb without conjunction
                if pos_sequence[i].startswith('VB') and pos_sequence[i+1].startswith('VB'):
                    coherence_score -= 6
            
            # Bonus for good patterns
            has_noun = any(tag.startswith('NN') for tag in pos_sequence)
            has_verb = any(tag.startswith('VB') for tag in pos_sequence)
            
            if has_noun and has_verb:
                coherence_score += 8  # Good subject-verb structure
            elif has_noun or has_verb:
                coherence_score += 3  # At least has one
            
            # Check for reasonable distribution (not all determiners/pronouns)
            content_words = sum(1 for tag in pos_sequence if tag.startswith(('NN', 'VB', 'JJ', 'RB')))
            if len(pos_sequence) > 0:
                content_ratio = content_words / len(pos_sequence)
                if content_ratio > 0.5:
                    coherence_score += 5
                elif content_ratio < 0.3:
                    coherence_score -= 5
        except:
            coherence_score -= 5  # Penalty if POS tagging fails
        
        # spaCy dependency parsing for advanced grammar checking
        if nlp is not None:
            try:
                doc = nlp(line)
                
                # Check for proper sentence structure
                has_root = any(token.dep_ == 'ROOT' for token in doc)
                if has_root:
                    coherence_score += 5
                
                # Penalty for excessive punctuation dependencies
                punct_count = sum(1 for token in doc if token.dep_ == 'punct')
                if punct_count > 2:
                    coherence_score -= 3
                
                # Check for broken dependencies (orphaned words)
                orphan_count = sum(1 for token in doc if token.dep_ == 'dep')
                if orphan_count > 0:
                    coherence_score -= 4 * orphan_count
                
            except:
                pass  # If spaCy fails, just skip this check
        
        return max(0, min(30, coherence_score))
    
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
        
        # Sentence coherence check using NLTK + spaCy
        sentence_coherence = self._check_sentence_coherence(line)
        score += sentence_coherence
        
        # NOTE: BERT check removed from here for performance
        # BERT will only validate the final best haiku, not every candidate
        
        # Coherence check via TF-IDF similarity to training data
        if self.tfidf_vectorizer is not None and self.training_vectors is not None:
            try:
                line_vector = self.tfidf_vectorizer.transform([line])
                similarities = cosine_similarity(line_vector, self.training_vectors)
                avg_similarity = similarities.mean()
                
                # Similarity-based scoring (reduced weight since we now have grammar checking)
                # Average similarity of 0.1 = +5 points, 0.2 = +10, etc.
                vocab_score = min(20, avg_similarity * 100)  # Cap at 20 points
                score += vocab_score
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
        
        # Detect nonsensical haikus: check for single-letter words across all lines
        single_letter_count = sum(1 for w in all_words if len(w) == 1 and w not in ['a', 'i'])
        if single_letter_count > 0:
            overall_score -= 25 * single_letter_count  # Heavy penalty
        
        # Check overall haiku coherence using TF-IDF (coherence is primary factor)
        coherence_bonus = 0
        if self.tfidf_vectorizer is not None:
            try:
                full_text = ' '.join(lines)
                haiku_vector = self.tfidf_vectorizer.transform([full_text])
                similarities = cosine_similarity(haiku_vector, self.training_vectors)
                avg_sim = similarities.mean()
                
                # Coherence contributes up to 30 points
                coherence_bonus = min(30, avg_sim * 150)
            except:
                coherence_bonus = -5  # Penalty if TF-IDF fails
        
        # Combine scores (line quality + coherence, no variety)
        final_score = overall_score * 0.7 + coherence_bonus
        
        # Ensure final score is in 0-100 range
        final_score = min(100, max(0, final_score))
        
        breakdown = {
            'line_scores': [round(s, 1) for s in line_scores],
            'avg_line_score': round(avg_line_score, 1),
            'coherence_bonus': round(coherence_bonus, 1),
            'final_score': round(final_score, 1)
        }
        
        return final_score, breakdown
    
    def generate_line(self, target_syllables, model, max_attempts=50):
        """
        Generate a single line with the target syllable count using templates.
        
        Args:
            target_syllables: Target number of syllables (5 or 7)
            model: The n-gram model to use
            max_attempts: Maximum number of generation attempts (reduced for speed)
            
        Returns:
            Generated line or None if failed
        """
        # Select appropriate templates
        templates = self.templates_5 if target_syllables == 5 else self.templates_7
        
        # BERT during generation is too slow - use smart N-gram generation instead
        # BERT will be used for post-filtering complete haikus
        return self._generate_line_ngram(target_syllables, model, max_attempts=100)
    
    def _generate_line_bert_guided(self, target_syllables, model, max_attempts=50):
        """
        BERT-guided N-gram generation: use BERT to score word choices.
        """
        best_line = None
        best_score = -float('inf')
        
        for attempt in range(max_attempts):
            line = []
            current_syllables = 0
            
            # Start word
            start_word = model.get_random_start_word()
            if not start_word:
                continue
            
            line.append(start_word)
            current_syllables = self.syllable_counter.count_syllables(start_word)
            
            if current_syllables == target_syllables:
                return ' '.join(line)
            
            if current_syllables > target_syllables:
                continue
            
            # Build line word by word
            max_words = 15
            context = [start_word]
            
            for _ in range(max_words):
                remaining_syllables = target_syllables - current_syllables
                if remaining_syllables == 0:
                    break
                
                # Get candidate next words from N-gram model
                context_tuple = tuple(context[-(model.n-1):])
                if context_tuple in model.ngrams:
                    candidates = model.ngrams[context_tuple]
                    
                    # Use BERT scoring sparingly (only for first 2 words and last word for speed)
                    use_bert = bert_model is not None and len(candidates) > 1 and (len(line) <= 2 or remaining_syllables <= 3)
                    
                    if use_bert:
                        scored_candidates = []
                        # Only score top 3 most frequent candidates (for speed)
                        for word, freq in sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:3]:
                            syl = self.syllable_counter.count_syllables(word)
                            if syl <= remaining_syllables:
                                bert_score = self._bert_score_word_in_context(line, word)
                                # Combine BERT score with frequency
                                combined_score = bert_score * 0.8 + (freq / sum(candidates.values())) * 0.2
                                scored_candidates.append((word, syl, combined_score))
                        
                        if scored_candidates:
                            # Pick best scoring word that fits
                            scored_candidates.sort(key=lambda x: x[2], reverse=True)
                            next_word, word_syllables, _ = scored_candidates[0]
                        else:
                            next_word = model.generate_next_word(context[-(model.n-1):])
                            if next_word:
                                word_syllables = self.syllable_counter.count_syllables(next_word)
                            else:
                                break
                    else:
                        next_word = model.generate_next_word(context[-(model.n-1):])
                        if next_word:
                            word_syllables = self.syllable_counter.count_syllables(next_word)
                        else:
                            break
                else:
                    next_word = model.get_random_start_word()
                    if not next_word:
                        break
                    word_syllables = self.syllable_counter.count_syllables(next_word)
                
                if word_syllables == remaining_syllables:
                    line.append(next_word)
                    current_syllables += word_syllables
                    break
                elif word_syllables < remaining_syllables:
                    line.append(next_word)
                    current_syllables += word_syllables
                    context.append(next_word)
                else:
                    # Try to find a word with exact remaining syllables
                    exact_match = [w for w in model.all_words 
                                  if self.syllable_counter.count_syllables(w) == remaining_syllables]
                    if exact_match:
                        next_word = random.choice(exact_match[:5])  # Pick from top 5
                        line.append(next_word)
                        current_syllables += remaining_syllables
                        break
                    else:
                        break
            
            # Validate line
            final_syllables = sum(self.syllable_counter.count_syllables(w) for w in line)
            if final_syllables == target_syllables and len(line) > 1:
                last_word = line[-1].lower()
                if last_word not in STOPWORDS:
                    line_text = ' '.join(line)
                    has_garbage = any(len(w) == 1 and w.lower() not in ['a', 'i'] for w in line)
                    if not has_garbage:
                        # Quick quality check
                        if best_line is None:
                            best_line = line_text
                            best_score = 0
                        else:
                            # Keep the best one
                            return line_text
        
        return best_line
    
    def _generate_line_ngram(self, target_syllables, model, max_attempts=200):
        """
        Original N-gram based line generation (fallback method).
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
            max_words = 20
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
                        next_word = model.get_random_start_word()
                        if next_word is None:
                            break
                
                # Check if adding this word would work
                word_syllables = self.syllable_counter.count_syllables(next_word)
                
                if word_syllables == remaining_syllables:
                    line.append(next_word)
                    current_syllables += word_syllables
                    break
                elif word_syllables < remaining_syllables:
                    line.append(next_word)
                    current_syllables += word_syllables
                    context.append(next_word)
                    stuck_count = 0
                else:
                    stuck_count += 1
                    if stuck_count > 5:
                        break
            
            # Check if we hit the target exactly
            final_syllables = sum(self.syllable_counter.count_syllables(w) for w in line)
            if final_syllables == target_syllables and len(line) > 0:
                last_word = line[-1].lower()
                if last_word not in STOPWORDS:
                    line_text = ' '.join(line)
                    
                    has_garbage = any(len(w) == 1 and w.lower() not in ['a', 'i'] for w in line)
                    if not has_garbage:
                        return line_text
        
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
    
    def generate_haiku(self, attempts=15, return_score=False, temperature=0.4):
        """
        Generate a complete haiku. Generates multiple candidates and picks the best.
        BERT is used to select the most coherent candidate.
        
        Args:
            attempts: Number of haiku candidates to generate (more = better quality)
            return_score: If True, returns (haiku, score, breakdown) tuple
            temperature: Controls randomness (0=deterministic, 0.4=default/balanced)
            
        Returns:
            Formatted haiku string, or tuple with (haiku, score, breakdown) if return_score=True
        """
        candidates = []
        
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
                
                # Score the haiku
                score, breakdown = self.score_haiku_quality(haiku)
                candidates.append((haiku, score, breakdown))
        
        if candidates:
            # Use BERT to re-rank top candidates for coherence
            if bert_model is not None and bert_tokenizer is not None and len(candidates) > 1:
                # Get top 5 candidates by score
                top_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:5]
                
                # Score each with BERT
                bert_scored = []
                for haiku, score, breakdown in top_candidates:
                    lines = haiku.split('\n')
                    # Check BERT coherence for each line
                    bert_checks = [self._check_bert_coherence(line, threshold=-6.0) for line in lines]
                    bert_score = sum(bert_checks) / len(bert_checks)  # Fraction of lines passing
                    
                    # Combine original score with BERT score
                    final_score = score * 0.6 + bert_score * 40  # BERT can add up to 40 points
                    bert_scored.append((haiku, final_score, breakdown, bert_score))
                
                # Pick best after BERT re-ranking
                best = max(bert_scored, key=lambda x: x[1])
                best_haiku, best_score, best_breakdown, bert_boost = best
                
                if bert_boost < 0.5:
                    print(f"Note: Generated haiku has moderate coherence (BERT score: {bert_boost:.2f})")
            else:
                # No BERT - use regular scoring
                best = max(candidates, key=lambda x: x[1])
                best_haiku, best_score, best_breakdown = best
            
            if return_score:
                return (best_haiku, best_score, best_breakdown)
            else:
                return best_haiku
        
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
        """Generate a single line with keyword attempt using templates and SBERT validation."""
        keyword_words = keyword.lower().split()
        keyword_syllables = sum(self.syllable_counter.count_syllables(w) for w in keyword_words)
        
        # Check if keyword fits in the target syllable count
        if keyword_syllables > target_syllables:
            print(f"Warning: Keyword '{keyword}' has {keyword_syllables} syllables, more than target {target_syllables}")
            return None
        
        # Try template-based generation with keyword
        templates = self.templates_5 if target_syllables == 5 else self.templates_7
        
        if hasattr(self, 'templates_5') and templates:
            for attempt in range(max_attempts):
                # Pick a template that can accommodate the keyword
                template = random.choice(templates)
                
                # Find a position to insert the keyword
                keyword_pos = random.randint(0, len(template))
                
                # Build line by filling template around keyword
                line = []
                current_syllables = 0
                success = True
                
                for i, (template_word, pos_tag, syl_count) in enumerate(template):
                    # Insert keyword at chosen position
                    if i == keyword_pos:
                        line.extend(keyword_words)
                        current_syllables += keyword_syllables
                    
                    # Skip template slots that would exceed syllable count
                    if current_syllables + syl_count > target_syllables:
                        continue
                    
                    # Try to find matching word from model
                    candidates = []
                    for word in model.all_words:
                        word_syllables = self.syllable_counter.count_syllables(word)
                        if word_syllables != syl_count:
                            continue
                        
                        try:
                            word_pos = pos_tag([word])[0][1]
                            if (pos_tag.startswith('NN') and word_pos.startswith('NN')) or \
                               (pos_tag.startswith('VB') and word_pos.startswith('VB')) or \
                               (pos_tag.startswith('JJ') and word_pos.startswith('JJ')) or \
                               (pos_tag.startswith('RB') and word_pos.startswith('RB')) or \
                               (pos_tag == word_pos):
                                candidates.append(word)
                        except:
                            continue
                    
                    if candidates:
                        chosen_word = random.choice(candidates)
                        line.append(chosen_word)
                        current_syllables += syl_count
                    else:
                        # Try any word with right syllables
                        fallback = [w for w in model.all_words 
                                   if self.syllable_counter.count_syllables(w) == syl_count]
                        if fallback and current_syllables + syl_count <= target_syllables:
                            line.append(random.choice(fallback))
                            current_syllables += syl_count
                
                # Add keyword at end if not yet added
                if keyword.lower() not in ' '.join(line).lower():
                    if current_syllables + keyword_syllables <= target_syllables:
                        line.extend(keyword_words)
                        current_syllables += keyword_syllables
                
                # Validate
                if current_syllables == target_syllables and len(line) > 0:
                    line_text = ' '.join(line)
                    
                    # Check keyword is in line
                    if keyword.lower() not in line_text.lower():
                        continue
                    
                    last_word = line[-1].lower()
                    if last_word in STOPWORDS:
                        continue
                    
                    has_garbage = any(len(w) == 1 and w.lower() not in ['a', 'i'] for w in line)
                    if has_garbage:
                        continue
                    
                    # Return line - BERT check in final scoring only
                    return line_text
        
        # Fallback to N-gram generation
        return self._generate_single_line_with_keyword_ngram(target_syllables, model, keyword, max_attempts)
    
    def _generate_single_line_with_keyword_ngram(self, target_syllables, model, keyword, max_attempts=20):
        """Original N-gram keyword line generation (fallback)."""
        keyword_words = keyword.lower().split()
        keyword_syllables = sum(self.syllable_counter.count_syllables(w) for w in keyword_words)
        
        if keyword_syllables > target_syllables:
            return None
        
        for attempt in range(max_attempts):
            line = keyword_words.copy()
            current_syllables = keyword_syllables
            remaining_syllables = target_syllables - current_syllables
            
            if remaining_syllables == 0:
                return ' '.join(line)
            
            add_before = random.choice([True, False])
            max_words = 20
            stuck_count = 0
            
            for _ in range(max_words):
                remaining_syllables = target_syllables - current_syllables
                
                if remaining_syllables == 0:
                    break
                
                if add_before and len(line) > 0:
                    context = [line[0]]
                else:
                    context = [line[-1]]
                
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
                    if add_before:
                        line.insert(0, next_word)
                    else:
                        line.append(next_word)
                    current_syllables += word_syllables
                    break
                elif word_syllables < remaining_syllables:
                    if add_before:
                        line.insert(0, next_word)
                    else:
                        line.append(next_word)
                    current_syllables += word_syllables
                    stuck_count = 0
                    add_before = not add_before
                else:
                    stuck_count += 1
                    if stuck_count > 5:
                        break
            
            final_syllables = sum(self.syllable_counter.count_syllables(w) for w in line)
            if final_syllables == target_syllables:
                last_word = line[-1].lower()
                if last_word not in STOPWORDS:
                    line_text = ' '.join(line)
                    has_garbage = any(len(w) == 1 and w.lower() not in ['a', 'i'] for w in line)
                    if not has_garbage:
                        return line_text
        
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

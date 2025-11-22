#!/usr/bin/env python3
"""
Pure Python ARPA Language Model Generator

Generates ARPA format language models without requiring KenLM's lmplz binary.
Implements n-gram counting and Kneser-Ney smoothing for realistic LM probabilities.
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ARPAGenerator:
    """Generate ARPA language models using n-gram statistics."""
    
    def __init__(self, order: int = 3, discount: float = 0.75):
        """
        Initialize ARPA generator.
        
        Args:
            order: Maximum n-gram order (e.g., 3 for trigram)
            discount: Kneser-Ney discount factor (0-1)
        """
        self.order = order
        self.discount = discount
        
        # Storage for n-gram counts
        self.counts = {}  # {order: {ngram_tuple: count}}
        self.contexts = {}  # {order: {context_tuple: set(words)}}
        self.vocab = set()
        self.vocab_size = 0
        self.total_tokens = 0
        self.unk_count = 0
        
        logger.info(f"Initialized ARPA Generator: order={order}, discount={discount}")
    
    def add_sentence(self, words: List[str], unk_token: str = "<unk>"):
        """
        Add a sentence to the language model, building n-gram statistics.
        
        Args:
            words: List of words/tokens
            unk_token: Token for unknown words
        """
        # Add start/end tokens
        words = ["<s>"] + words + ["</s>"]
        
        # Track vocabulary
        for word in words:
            if word not in ["<s>", "</s>"]:
                self.vocab.add(word)
                self.total_tokens += 1
        
        # Count n-grams up to the specified order
        for order in range(1, self.order + 1):
            if order not in self.counts:
                self.counts[order] = Counter()
                self.contexts[order] = defaultdict(set)
            
            # Extract n-grams of this order
            for i in range(len(words) - order + 1):
                ngram = tuple(words[i:i + order])
                self.counts[order][ngram] += 1
                
                # Track contexts (n-1 gram preceding the word)
                if order > 1:
                    context = ngram[:-1]
                    word = ngram[-1]
                    self.contexts[order][context].add(word)
        
        self.vocab_size = len(self.vocab)
    
    def compute_probabilities(self) -> Dict[int, Dict[Tuple, float]]:
        """
        Compute n-gram probabilities using Kneser-Ney smoothing.
        
        Returns:
            Dictionary mapping order to {ngram: log10_probability}
        """
        logger.info("Computing n-gram probabilities...")
        probs = {}
        backoff_weights = {}  # For backing off to lower-order models
        
        for order in range(1, self.order + 1):
            probs[order] = {}
            backoff_weights[order] = {}
            counts = self.counts.get(order, Counter())
            
            if not counts:
                logger.warning(f"No {order}-grams found")
                continue
            
            # Unigram probabilities (simple MLE with add-one smoothing)
            if order == 1:
                total_count = sum(counts.values())
                for ngram, count in counts.items():
                    # Laplace smoothing: (count + 1) / (total + vocab_size)
                    prob = (count + 1) / (total_count + self.vocab_size)
                    probs[order][ngram] = math.log10(max(prob, 1e-10))
            else:
                # Higher-order n-grams: Kneser-Ney smoothing
                context_counts = defaultdict(int)
                
                # Count how many times each context appears
                for ngram, count in counts.items():
                    context = ngram[:-1]
                    context_counts[context] += count
                
                # Compute backoff weights for backing off from order to order-1
                lower_probs = probs.get(order - 1, {})
                
                for ngram, count in counts.items():
                    context = ngram[:-1]
                    word = ngram[-1]
                    context_count = context_counts[context]
                    
                    # Kneser-Ney: discounted count / context count
                    discounted = max(count - self.discount, 0)
                    prob = discounted / context_count if context_count > 0 else 1e-10
                    
                    # Add backoff weight contribution
                    if (word,) in lower_probs:
                        backoff_prob = 10 ** lower_probs[(word,)]
                        prob += (self.discount * len(self.contexts[order].get(context, set())) / context_count) * backoff_prob
                    
                    probs[order][ngram] = math.log10(max(prob, 1e-10))
                
                # Compute backoff weights
                for context in context_counts:
                    discounted_mass = self.discount * len(self.contexts[order].get(context, set())) / context_counts[context]
                    backoff_weights[order][context] = math.log10(max(discounted_mass, 1e-10))
        
        self.backoff_weights = backoff_weights
        logger.info(f"Computed probabilities for {len(probs)} orders")
        return probs
    
    def write_arpa(self, output_path: Path, probs: Dict[int, Dict[Tuple, float]]):
        """
        Write probabilities in ARPA format.
        
        Args:
            output_path: Path to write ARPA file
            probs: Dictionary of probabilities from compute_probabilities()
        """
        logger.info(f"Writing ARPA file to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("\\data\\\n")
            
            # Count n-grams for each order (add 1 for <unk> in unigrams)
            for order in range(1, self.order + 1):
                count = len(probs.get(order, {}))
                if order == 1:
                    count += 1  # Add <unk> token
                if count > 0:
                    f.write(f"ngram {order}={count}\n")
            
            f.write("\n")
            
            # Write n-grams by order
            for order in range(1, self.order + 1):
                order_probs = probs.get(order, {})
                if not order_probs and order > 1:
                    continue
                
                f.write(f"\\{order}-grams:\n")
                
                # Add <unk> token for unigrams
                if order == 1:
                    unk_prob = math.log10(1e-5)  # Default low probability for unknown
                    f.write(f"{unk_prob:.6f}\t<unk>\t0.000000\n")
                
                # Sort for consistency
                for ngram in sorted(order_probs.keys()):
                    prob = order_probs[ngram]
                    ngram_str = " ".join(ngram)
                    
                    # Backoff weight (if not the highest order)
                    if order < self.order:
                        context = ngram[:-1]
                        backoff_weight = self.backoff_weights.get(order, {}).get(context, 0)
                        f.write(f"{prob:.6f}\t{ngram_str}\t{backoff_weight:.6f}\n")
                    else:
                        f.write(f"{prob:.6f}\t{ngram_str}\n")
                
                f.write("\n")
            
            # Footer
            f.write("\\end\\\n")
        
        logger.info(f"✓ ARPA file written: {output_path}")
        logger.info(f"  - Vocabulary size: {self.vocab_size}")
        logger.info(f"  - Total tokens: {self.total_tokens}")


def load_corpus(corpus_path: Path, limit: Optional[int] = None, lowercase: bool = True) -> List[str]:
    """
    Load corpus from text file.
    
    Args:
        corpus_path: Path to corpus file (one sentence per line)
        limit: Maximum number of lines to load
        lowercase: Whether to lowercase all text
    
    Returns:
        List of sentences
    """
    sentences = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            
            line = line.strip()
            if not line:
                continue
            
            if lowercase:
                line = line.lower()
            
            sentences.append(line)
    
    logger.info(f"Loaded {len(sentences)} sentences from {corpus_path}")
    return sentences


def tokenize(text: str, charset: Optional[str] = None) -> List[str]:
    """
    Simple tokenization (character-level or word-level).
    
    Args:
        text: Text to tokenize
        charset: If provided, only keep characters in this set (for character-level)
    
    Returns:
        List of tokens
    """
    # Word-level tokenization (split on whitespace)
    tokens = text.split()
    
    if charset:
        # Filter to valid characters only
        valid_tokens = []
        for token in tokens:
            valid_chars = [c for c in token if c in charset]
            if valid_chars:
                valid_tokens.append(''.join(valid_chars))
        return valid_tokens
    
    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Generate ARPA language models using pure Python"
    )
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('lm/smartnotes_corpus.txt'),
        help='Path to corpus file (one sentence per line)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        default=Path('lm/smartnotes.arpa'),
        help='Output ARPA file path'
    )
    parser.add_argument(
        '--order',
        type=int,
        default=3,
        help='Maximum n-gram order (e.g., 3 for trigram)'
    )
    parser.add_argument(
        '--discount',
        type=float,
        default=0.75,
        help='Kneser-Ney discount factor (0-1)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit to first N sentences'
    )
    parser.add_argument(
        '--lowercase',
        action='store_true',
        default=True,
        help='Lowercase all text'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Verify corpus exists
    if not args.corpus.exists():
        logger.error(f"Corpus file not found: {args.corpus}")
        sys.exit(1)
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load corpus
        logger.info(f"Loading corpus from {args.corpus}")
        sentences = load_corpus(args.corpus, limit=args.limit, lowercase=args.lowercase)
        
        if not sentences:
            logger.error("No sentences loaded from corpus")
            sys.exit(1)
        
        # Initialize generator
        generator = ARPAGenerator(order=args.order, discount=args.discount)
        
        # Add sentences to model
        logger.info(f"Building {args.order}-gram model...")
        for i, sentence in enumerate(sentences):
            if (i + 1) % 1000 == 0:
                logger.info(f"  Processed {i + 1}/{len(sentences)} sentences...")
            
            tokens = tokenize(sentence)
            generator.add_sentence(tokens)
        
        logger.info(f"✓ Processed {len(sentences)} sentences")
        
        # Compute probabilities
        probs = generator.compute_probabilities()
        
        # Write ARPA file
        generator.write_arpa(args.output, probs)
        
        logger.info("✓ ARPA language model generation complete")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

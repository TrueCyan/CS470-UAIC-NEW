# 어휘 구축 구현 예정 

import nltk
from collections import Counter
import pickle
from typing import List, Tuple

# Assuming config.py is in the parent directory of utils, or accessible via PYTHONPATH
# For direct execution or testing, you might need to adjust the import path for config
try:
    from .. import config
except ImportError:
    # Fallback for cases where the module is run directly or in a different context
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import config

class Vocabulary:
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word_counts = Counter()

        # Add special tokens first
        self.add_word(config.PAD_TOKEN)
        self.add_word(config.BOS_TOKEN)
        self.add_word(config.EOS_TOKEN)
        self.add_word(config.UNK_TOKEN)
        self.add_word(config.NONE_TOKEN)

    def add_word(self, word: str):
        """Add a word to the vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        self.word_counts[word] += 1

    def add_sentence(self, sentence: str):
        """Tokenize and add all words from a sentence to the vocabulary."""
        # Basic tokenization, consider using a more robust tokenizer if needed
        tokens = nltk.tokenize.word_tokenize(sentence.lower())
        for token in tokens:
            self.add_word(token)

    def build_vocab(self, all_captions: List[str], min_word_freq: int = config.MIN_WORD_FREQ):
        """Build vocabulary from a list of captions.

        Args:
            all_captions: A list of strings, where each string is a caption.
            min_word_freq: Minimum frequency for a word to be included in the vocabulary.
        """
        # First, count all words
        for caption in all_captions:
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            self.word_counts.update(tokens)

        # Reset vocab and add special tokens first
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word(config.PAD_TOKEN)
        self.add_word(config.BOS_TOKEN)
        self.add_word(config.EOS_TOKEN)
        self.add_word(config.UNK_TOKEN)
        self.add_word(config.NONE_TOKEN) # Ensure NONE_TOKEN is part of the final vocab

        # Add words that meet the frequency threshold
        words = [word for word, count in self.word_counts.items() if count >= min_word_freq]
        for word in words:
            if word not in self.word2idx: # Avoid re-adding if somehow already present
                self.add_word(word) # Use add_word to maintain consistency and update idx
        
        # Re-initialize word_counts based on the final vocabulary to only count known words.
        # This is not strictly necessary for functionality but can be useful for debugging.
        final_vocab_words = list(self.word2idx.keys())
        temp_counts = Counter()
        for caption in all_captions:
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            for token in tokens:
                if token in final_vocab_words:
                    temp_counts[token] +=1
        self.word_counts = temp_counts


    def __call__(self, word: str) -> int:
        """Convert a word to its corresponding index."""
        if word not in self.word2idx:
            return self.word2idx[config.UNK_TOKEN]
        return self.word2idx[word]

    def get_word(self, idx: int) -> str:
        """Convert an index to its corresponding word."""
        if idx not in self.idx2word:
            # This case should ideally not happen if idx is valid
            return config.UNK_TOKEN
        return self.idx2word[idx]

    def __len__(self) -> int:
        return len(self.word2idx)

    def sentence_to_indices(self, sentence: str) -> List[int]:
        """Convert a sentence string to a list of token indices."""
        tokens = nltk.tokenize.word_tokenize(sentence.lower())
        return [self(token) for token in tokens]

    def indices_to_sentence(self, indices: List[int], remove_special_tokens: bool = True) -> str:
        """Convert a list of token indices back to a sentence string."""
        words = []
        for idx in indices:
            word = self.get_word(idx)
            if remove_special_tokens and word in [
                config.PAD_TOKEN, config.BOS_TOKEN, 
                config.EOS_TOKEN, config.NONE_TOKEN
            ]:
                if word == config.EOS_TOKEN: # Stop at EOS if removing special tokens
                    break
                continue
            if remove_special_tokens and word == config.UNK_TOKEN and not words: # Avoid starting with UNK
                 continue
            words.append(word)
        return ' '.join(words)

    @staticmethod
    def load_vocab(path: str) -> 'Vocabulary':
        """Load a vocabulary instance from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_vocab(self, path: str):
        """Save the vocabulary instance to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

if __name__ == '__main__':
    # Build vocabulary from MS COCO captions
    import json
    import os

    # Ensure NLTK data is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')

    # Load Karpathy split JSON
    print("Loading Karpathy split JSON...")
    with open(config.TRAIN_CAPTION_JSON, 'r') as f:
        data = json.load(f)

    # Collect all captions from training split
    print("Collecting captions...")
    all_captions = []
    for item in data['images']:
        if item['split'] == 'train':
            for sentence in item['sentences']:
                all_captions.append(sentence['raw'].lower())
    
    print(f"Found {len(all_captions)} captions")

    # Build vocabulary
    print("Building vocabulary...")
    vocab = Vocabulary()
    vocab.build_vocab(all_captions, min_word_freq=config.MIN_WORD_FREQ)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Saving vocabulary to {config.VOCAB_PATH}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config.VOCAB_PATH), exist_ok=True)
    vocab.save_vocab(config.VOCAB_PATH)
    
    print("Done!")
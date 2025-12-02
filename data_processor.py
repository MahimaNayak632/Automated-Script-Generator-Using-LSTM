import pandas as pd
import numpy as np
import re
import nltk
import os

class DataProcessor:
    def __init__(self, vocab_size=5000, sequence_length=20):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocabulary = []
        
    def create_sample_dataset(self):
        """Create a better sample dataset"""
        sample_data = {
            'character1': ['JOHN', 'MARY', 'DETECTIVE', 'FRIEND1', 'BOSS', 'HERO'],
            'character2': ['MARY', 'JOHN', 'SUSPECT', 'FRIEND2', 'EMPLOYEE', 'VILLAIN'],
            'dialogue': [
                "I love you more than anything in this world",
                "I feel the same way about you my darling",
                "Where were you last night tell me the truth",
                "I was at home all night I swear it",
                "That was the funniest thing I have ever seen",
                "I know right we should do that again sometime"
            ],
            'genre': ['romance', 'romance', 'thriller', 'thriller', 'comedy', 'comedy']
        }
        
        self.df = pd.DataFrame(sample_data)
        self.df.to_csv('movie_dialogs.csv', index=False)
        print("Sample dataset created!")
    
    def load_data(self, file_path='movie_dialogs.csv'):
        """Load and preprocess the dataset"""
        self.df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(self.df)} dialogues")
        
    def simple_tokenize(self, text):
        """Simple tokenizer that splits on whitespace"""
        if not isinstance(text, str):
            return []
        # Convert to lowercase and split
        text = text.lower().strip()
        tokens = text.split()
        return tokens
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep basic words and punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\?\!,]', '', text)
        return text.strip()
    
    def build_vocabulary(self):
        """Build vocabulary from dialogues"""
        all_text = ' '.join(self.df['dialogue'].apply(self.preprocess_text))
        
        # Use simple tokenizer
        tokens = self.simple_tokenize(all_text)
        
        # Add character names and genres to vocabulary
        character_names = []
        for col in ['character1', 'character2']:
            character_names.extend([name.lower() for name in self.df[col].unique()])
        
        genres = [genre.lower() for genre in self.df['genre'].unique()]
        
        # Get most frequent words
        word_freq = {}
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
        
        # Add character names and genres to frequency dict
        for name in character_names:
            word_freq[name] = word_freq.get(name, 0) + 10
        
        for genre in genres:
            word_freq[genre] = word_freq.get(genre, 0) + 10
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        common_words = [word for word, freq in sorted_words[:self.vocab_size - 8]]
        
        # Build vocabulary with special tokens
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>', '<COMEDY>', '<DRAMA>', '<THRILLER>', '<ROMANCE>', '<ACTION>']
        self.vocabulary = special_tokens + common_words
        
        # Create mapping dictionaries
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.index_to_word = {idx: word for idx, word in enumerate(self.vocabulary)}
        
        print(f"Vocabulary built with {len(self.vocabulary)} words")
        print(f"Sample words: {list(self.vocabulary[:15])}")
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        # Preprocess and tokenize
        processed_text = self.preprocess_text(text)
        tokens = self.simple_tokenize(processed_text)
        
        sequence = []
        for token in tokens:
            if token in self.word_to_index:
                sequence.append(self.word_to_index[token])
            else:
                sequence.append(self.word_to_index['<UNK>'])
        
        return sequence
    
    def sequence_to_text(self, sequence):
        """Convert sequence of indices back to text"""
        words = []
        for idx in sequence:
            if idx < len(self.index_to_word):
                word = self.index_to_word[idx]
                if word not in ['<PAD>', '<START>', '<END>']:
                    words.append(word)
        
        return ' '.join(words)
    
    def prepare_training_data(self):
        """Prepare input-output pairs for training - FIXED VERSION"""
        input_sequences = []
        target_words = []
        
        for _, row in self.df.iterrows():
            # Add character and genre context
            character_tag = f"{row['character1'].lower()}"
            genre_tag = f"<{row['genre'].upper()}>"
            
            # Preprocess dialogue
            dialogue_text = row['dialogue'].lower()
            
            # Create full text with context
            full_text = f"{genre_tag} {character_tag} {dialogue_text}"
            full_seq = self.text_to_sequence(full_text)
            
            # Create training pairs (input sequence -> next word)
            for i in range(len(full_seq) - 1):
                # Input: sequence up to current position
                input_seq = full_seq[:i+1]
                
                # Pad or truncate input sequence
                if len(input_seq) < self.sequence_length:
                    input_seq = input_seq + [self.word_to_index['<PAD>']] * (self.sequence_length - len(input_seq))
                else:
                    input_seq = input_seq[-self.sequence_length:]
                
                # Target: next word
                target_word = full_seq[i+1] if i+1 < len(full_seq) else self.word_to_index['<END>']
                
                input_sequences.append(input_seq)
                target_words.append(target_word)
        
        print(f"Created {len(input_sequences)} training samples")
        return np.array(input_sequences), np.array(target_words)

# Test the data processor
if __name__ == "__main__":
    processor = DataProcessor()
    processor.create_sample_dataset()
    processor.load_data()
    processor.build_vocabulary()
    
    # Test sequence conversion
    test_text = "I love you more than anything"
    test_seq = processor.text_to_sequence(test_text)
    reconstructed = processor.sequence_to_text(test_seq)
    print(f"Test - Original: '{test_text}'")
    print(f"Test - Sequence: {test_seq}")
    print(f"Test - Reconstructed: '{reconstructed}'")
    
    # Test training data preparation
    X, y = processor.prepare_training_data()
    print(f"Training data - X shape: {X.shape}, y shape: {y.shape}")
    
    print("Data processor ready!")
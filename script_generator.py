import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import random

class AdvancedScriptGenerator:
    def __init__(self, vocab_size, sequence_length, embedding_dim=256):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.model = None
        
    def build_advanced_model(self):
        """Build advanced LSTM model with better architecture"""
        # Text input
        text_input = Input(shape=(self.sequence_length,), name='text_input')
        
        # Embedding layer
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.sequence_length,
            mask_zero=True
        )(text_input)
        
        # First LSTM layer
        lstm1 = LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(embedding)
        
        # Second LSTM layer
        lstm2 = LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(lstm1)
        
        # Third LSTM layer
        lstm3 = LSTM(256, dropout=0.2, recurrent_dropout=0.1)(lstm2)
        
        # Dense layers
        dense1 = Dense(256, activation='relu')(lstm3)
        dropout1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(128, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        
        # Output layer
        output = Dense(self.vocab_size, activation='softmax')(dropout2)
        
        self.model = Model(inputs=text_input, outputs=output)
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Advanced LSTM model built successfully!")
        print(self.model.summary())
    
    def train(self, X_train, y_train, epochs=20, batch_size=64):
        """Train the model with callbacks"""
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.0001,
            verbose=1
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def generate_story_script(self, data_processor, characters, genre, max_length=200, temperature=0.8):
        """Generate story-driven script with temperature sampling"""
        if not self.model:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        try:
            # Start with genre context
            start_phrases = {
                'comedy': ["So I was thinking", "You'll never believe what happened", "Guess what just happened"],
                'drama': ["I need to tell you something", "This changes everything", "I've been thinking"],
                'thriller': ["I found something disturbing", "We're not alone", "There's something I need to show you"],
                'romance': ["I have to be honest with you", "From the moment I saw you", "I never thought I'd say this"],
                'action': ["We've got company", "The mission has changed", "There's no time to explain"]
            }
            
            genre_phrase = random.choice(start_phrases.get(genre, start_phrases['comedy']))
            start_text = f"{characters[0]}: {genre_phrase}"
            
            print(f"Starting story with: {start_text}")
            
            # Convert to sequence
            start_seq = data_processor.text_to_sequence(start_text)
            
            # Pad sequence
            if len(start_seq) < self.sequence_length:
                current_sequence = start_seq + [data_processor.word_to_index['<PAD>']] * (self.sequence_length - len(start_seq))
            else:
                current_sequence = start_seq[:self.sequence_length]
            
            generated_text = [start_text]
            current_char_index = 0
            conversation_depth = 0
            
            for i in range(max_length):
                # Predict next word
                input_seq = np.array([current_sequence])
                predictions = self.model.predict(input_seq, verbose=0)[0]
                
                # Apply temperature for creativity
                predictions = self._apply_temperature(predictions, temperature)
                
                # Sample from predictions
                predicted_id = self._sample_from_predictions(predictions)
                predicted_word = data_processor.index_to_word.get(predicted_id, '<UNK>')
                
                # Handle special tokens and character switching
                if predicted_word == '<END>' or len(generated_text) > max_length * 0.8:
                    break
                    
                if predicted_word not in ['<PAD>', '<START>', '<END>']:
                    current_text = generated_text[-1]
                    
                    # Switch characters after reasonable conversation length
                    if predicted_word in ['.', '!', '?']:
                        conversation_depth += 1
                        
                        if conversation_depth >= random.randint(2, 4):
                            current_char_index = (current_char_index + 1) % len(characters)
                            conversation_depth = 0
                            
                            # Add character tag
                            current_char = characters[current_char_index]
                            generated_text.append(f"{current_char}:")
                    
                    # Add word to current line
                    if ':' in current_text:
                        # This is a character line
                        parts = current_text.split(':')
                        if len(parts) == 2:
                            character = parts[0]
                            dialogue = parts[1].strip()
                            new_dialogue = dialogue + ' ' + predicted_word if dialogue else predicted_word
                            generated_text[-1] = f"{character}: {new_dialogue}"
                    else:
                        # This is narration or context
                        generated_text[-1] += ' ' + predicted_word
                
                # Update sequence
                current_sequence = current_sequence[1:] + [predicted_id]
            
            # Format final script
            formatted_script = self._format_final_script(generated_text)
            return formatted_script
            
        except Exception as e:
            print(f"Error in story generation: {e}")
            return self._generate_fallback_story(characters, genre)
    
    def _apply_temperature(self, predictions, temperature):
        """Apply temperature to predictions for more creative sampling"""
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions + 1e-8) / temperature
        exp_preds = np.exp(predictions)
        return exp_preds / np.sum(exp_preds)
    
    def _sample_from_predictions(self, predictions):
        """Sample from predictions with diversity"""
        # Sometimes choose top-3 for better quality
        if random.random() < 0.7:  # 70% of the time, choose from top-3
            top_indices = np.argsort(predictions)[-3:]
            top_probs = predictions[top_indices]
            top_probs = top_probs / np.sum(top_probs)
            return np.random.choice(top_indices, p=top_probs)
        else:
            return np.random.choice(len(predictions), p=predictions)
    
    def _format_final_script(self, generated_lines):
        """Format the generated lines into a proper script"""
        formatted_lines = []
        
        for line in generated_lines:
            if ':' in line:
                # Character dialogue
                parts = line.split(':', 1)
                if len(parts) == 2:
                    character = parts[0].strip()
                    dialogue = parts[1].strip()
                    
                    # Basic capitalization and punctuation
                    if dialogue and not dialogue.endswith(('.', '!', '?')):
                        dialogue += '.'
                    
                    formatted_lines.append(f"{character}: {dialogue.capitalize()}")
            else:
                # Narration/context (add as parenthetical)
                if line.strip():
                    formatted_lines.append(f"({line.strip().capitalize()})")
        
        return '\n'.join(formatted_lines)
    
    def _generate_fallback_story(self, characters, genre):
        """Generate a meaningful fallback story"""
        story_templates = {
            'comedy': [
                f"{characters[0]}: So I accidentally adopted three dogs today.",
                f"{characters[1]}: You did what? How does that even happen?",
                f"{characters[0]}: Well, I went to buy one, but they were having a special!",
                f"{characters[1]}: We live in a tiny apartment! Where will they sleep?",
                f"{characters[0]}: I was thinking... our bed?",
                f"{characters[1]}: This is going to be interesting."
            ],
            'drama': [
                f"{characters[0]}: I got the job in Paris.",
                f"{characters[1]}: Paris? When were you going to tell me?",
                f"{characters[0]}: I'm telling you now. It starts next month.",
                f"{characters[1]}: What about us? What about everything we built here?",
                f"{characters[0]}: I thought you'd come with me.",
                f"{characters[1]}: It's not that simple."
            ],
            'thriller': [
                f"{characters[0]}: The files are gone. All of them.",
                f"{characters[1]}: What do you mean gone? They were here an hour ago.",
                f"{characters[0]}: Someone's been in the system. They left a message.",
                f"{characters[1]}: What message?",
                f"{characters[0]}: 'We're watching. Don't look for us.'",
                f"{characters[1]}: This is bigger than we thought."
            ]
        }
        
        return '\n'.join(story_templates.get(genre, story_templates['comedy']))
    
    def save_model(self, model_path='advanced_script_model.h5'):
        """Save the trained model"""
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='advanced_script_model.h5'):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")







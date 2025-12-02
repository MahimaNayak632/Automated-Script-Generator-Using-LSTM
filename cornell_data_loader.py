import pandas as pd
import os
import urllib.request
import zipfile
import numpy as np
import re

class CornellDataLoader:
    def __init__(self):
        self.corpus_path = "cornell_movie-dialogs-corpus"
        self.movie_lines_file = os.path.join(self.corpus_path, "movie_lines.txt")
        self.movie_conversations_file = os.path.join(self.corpus_path, "movie_conversations.txt")
        
    def download_cornell_corpus(self):
        """Download the actual Cornell Movie Dialogs Corpus"""
        if os.path.exists(self.corpus_path):
            print("Cornell corpus already exists.")
            return True
            
        print("Downloading Cornell Movie Dialogs Corpus...")
        url = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
        zip_path = "cornell_movie_dialogs_corpus.zip"
        
        try:
            # Download the file
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Clean up
            os.remove(zip_path)
            print("Cornell Movie Dialogs Corpus downloaded and extracted!")
            return True
            
        except Exception as e:
            print(f"Error downloading Cornell corpus: {e}")
            return False
    
    def load_movie_data(self):
        """Load and parse the movie data with proper formatting"""
        try:
            # Load movie lines with correct separator
            lines_df = pd.read_csv(self.movie_lines_file, 
                                  sep=' \+\+\+\$\+\+\+ ',
                                  engine='python',
                                  header=None,
                                  names=['lineID', 'characterID', 'movieID', 'character', 'text'],
                                  encoding='utf-8',
                                  errors='ignore')
            
            # Load conversations
            conv_df = pd.read_csv(self.movie_conversations_file,
                                 sep=' \+\+\+\$\+\+\+ ',
                                 engine='python',
                                 header=None,
                                 names=['character1ID', 'character2ID', 'movieID', 'utteranceIDs'],
                                 encoding='utf-8',
                                 errors='ignore')
            
            # Clean the text
            lines_df['text'] = lines_df['text'].str.strip()
            lines_df['character'] = lines_df['character'].str.strip()
            
            print(f"Loaded {len(lines_df)} movie lines")
            print(f"Loaded {len(conv_df)} conversations")
            
            return lines_df, conv_df
            
        except Exception as e:
            print(f"Error loading movie data: {e}")
            return None, None
    
    def create_training_pairs(self, num_conversations=1000):
        """Create conversation pairs for training"""
        lines_df, conv_df = self.load_movie_data()
        
        if lines_df is None:
            print("Failed to load movie data, using fallback data")
            return self._create_fallback_data()
        
        training_data = []
        
        # Create a dictionary for lineID to text mapping
        line_dict = {}
        character_dict = {}
        
        for _, row in lines_df.iterrows():
            line_id = row['lineID'].strip()
            line_dict[line_id] = row['text']
            character_dict[line_id] = row['character']
        
        count = 0
        for _, row in conv_df.iterrows():
            if count >= num_conversations:
                break
                
            utterance_ids_str = row['utteranceIDs'].strip()
            # Clean and parse the utterance IDs
            utterance_ids = re.findall(r'L\d+', utterance_ids_str)
            
            # Create conversation pairs
            for i in range(len(utterance_ids) - 1):
                input_line_id = utterance_ids[i]
                output_line_id = utterance_ids[i + 1]
                
                if input_line_id in line_dict and output_line_id in line_dict:
                    input_char = character_dict[input_line_id]
                    output_char = character_dict[output_line_id]
                    input_text = line_dict[input_line_id]
                    output_text = line_dict[output_line_id]
                    
                    # Clean text
                    input_text = self.clean_text(input_text)
                    output_text = self.clean_text(output_text)
                    
                    if input_text and output_text:
                        training_data.append({
                            'character1': input_char,
                            'character2': output_char,
                            'dialogue': input_text + " " + output_text,
                            'genre': 'drama'  # Default genre
                        })
                        count += 1
        
        print(f"Created {len(training_data)} training pairs")
        return pd.DataFrame(training_data)
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Remove special characters but keep basic punctuation and words
        text = re.sub(r'[^a-zA-Z0-9\s\.\?\!,\']', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def _create_fallback_data(self):
        """Create meaningful fallback training data"""
        fallback_data = [
            {
                'character1': 'JOHN', 'character2': 'MARY', 'dialogue': 'I love you more than anything in this world', 'genre': 'romance'
            },
            {
                'character1': 'DETECTIVE', 'character2': 'SUSPECT', 'dialogue': 'Where were you last night tell me the truth', 'genre': 'thriller'
            },
            {
                'character1': 'FRIEND1', 'character2': 'FRIEND2', 'dialogue': 'That was the funniest thing I have ever seen in my life', 'genre': 'comedy'
            },
            {
                'character1': 'BOSS', 'character2': 'EMPLOYEE', 'dialogue': 'Your work has been exceptional this quarter we are giving you a promotion', 'genre': 'drama'
            },
            {
                'character1': 'HERO', 'character2': 'VILLAIN', 'dialogue': 'This ends now you will not destroy our city', 'genre': 'action'
            },
            {
                'character1': 'LOVER1', 'character2': 'LOVER2', 'dialogue': 'I never thought I would find someone like you you complete me', 'genre': 'romance'
            },
            {
                'character1': 'SCIENTIST', 'character2': 'ASSISTANT', 'dialogue': 'The experiment worked but there are unexpected consequences we need to be careful', 'genre': 'thriller'
            },
            {
                'character1': 'COMEDIAN', 'character2': 'AUDIENCE', 'dialogue': 'Why did the chicken cross the road to get to the other side no wait that is not funny', 'genre': 'comedy'
            }
        ]
        
        return pd.DataFrame(fallback_data)

# Test the data loader
if __name__ == "__main__":
    loader = CornellDataLoader()
    success = loader.download_cornell_corpus()
    if success:
        df = loader.create_training_pairs(50)
        print(f"Loaded {len(df)} conversation pairs")
        print("\nSample conversations:")
        for i in range(min(3, len(df))):
            print(f"{df.iloc[i]['character1']}: {df.iloc[i]['dialogue']}")
            print("---")
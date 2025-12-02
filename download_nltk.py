import nltk

print("Downloading required NLTK data...")

# Download all required NLTK datasets
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

print("All NLTK data downloaded successfully!")
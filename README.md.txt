# Greeklish vs. English Text Classifier

This repository contains a Python-based solution for the Al Developer Technical Assessment, classifying sentences as "Greeklish" (Greek in Latin characters) or "English" using scraped data and Logistic Regression.

## Project Structure
- `main.ipynb`: Combined script for scraping, preprocessing, training, and evaluation.
- `model/`: Directory with trained model (`greeklish_classifier.pkl`) and vectorizer (`tfidf_vectorizer.pkl`).
- `initial_dataset.csv`: Raw scraped data.
- `preprocessed_sentences.csv`: Processed dataset.
- `requirements.txt`: Dependencies list.
- `Documentation.pdf`: Detailed project explanation.

## Setting Up the Development Environment

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local environment
- Reddit API credentials (client ID, secret)

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt


Running the Scripts
1. Scraping, Preprocessing, Training, and Evaluation
In Colab:
Open main.ipynb in a Colab notebook.
Update Reddit API credentials in scrape_reddit and scrape_reddit_sentences:
client_id='YOUR_CLIENT_ID', client_secret='YOUR_SECRET', user_agent='GreeklishScraper/0.0.1'

2. Create a model/ directory: !mkdir model.
Run the script:
Outputs: initial_dataset.csv, preprocessed_sentences.csv, model.zip.
Locally:
Update file paths in main.py (e.g., replace Colab files.download with local saves).
Run: python main.py.

Notes:

Scraping: Requires internet access and Reddit API setup. YouTube scraping may need adjustment (e.g., API or Selenium) for dynamic content.
Data: Targets ~600 sentences (300 per class), expanded to 904 after splitting.

## Testing the Trained Classifier

Steps

1- Load Model and Vectorizer:
import joblib
model = joblib.load('model/greeklish_classifier.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

2- Define Preprocessing:

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

3- Predict:
def predict_text(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(vectorizer.transform([processed_text]))[0]
    return prediction

print(predict_text("ti kaneis"))  # Output: greeklish
print(predict_text("Hello, how are you?"))  # Output: English

Example Output
"ti kaneis" → "greeklish"
"Hello, how are you?" → "English"

## Troubleshooting
NLTK Errors: Ensure punkt and stopwords are downloaded.
Reddit API Issues: Verify credentials and rate limits.
YouTube Scraping: If empty, use YouTube Data API or Selenium.
Model Not Found: Check model/ contains .pkl files.

## Author
Danyal Shah - Developed for Al Developer Technical Assessment, April 2025.
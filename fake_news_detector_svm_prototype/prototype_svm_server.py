from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import os

app = Flask(__name__)
CORS(app)

try:
    # Load the saved models
    tfidf_vectorizer = joblib.load(r'SVM\fake_news_detector\tfidf_vectorizer.pkl')
    calibrated_model = joblib.load(r'SVM\fake_news_detector\svm_model.pkl')

    # Ensure NLTK data is available
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
except Exception as e:
    print(f"Error loading models or NLTK data: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Vectorize the text
        vectorized_text = tfidf_vectorizer.transform([processed_text])
        
        # Get prediction probability
        probability = calibrated_model.predict_proba(vectorized_text)[0]
        suspicious_probability = float(probability[1])
        
        # Determine credibility
        credibility = "Credible" if suspicious_probability < 0.8 else "Suspicious"
        
        # Generate summary
        summary = generate_summary(text)
        
        return jsonify({
            'credibility': credibility,
            'suspicious_probability': suspicious_probability,
            'summary': summary,
        })
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

def preprocess_text(text):
    """Preprocesses the input text by tokenizing, removing stopwords, and lemmatizing."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return ' '.join(lemmatized_tokens)

def generate_summary(text, sentences_count=3):
    """Generates a summary of the input text using LSA summarization."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return " ".join([str(sentence) for sentence in summary])
    except Exception as e:
        print(f"Error in generate_summary: {str(e)}")
        return text[:200] + "..."  # Fallback summary

if __name__ == '__main__':
    app.run(debug=True)

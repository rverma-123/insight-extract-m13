from transformers import pipeline
from flask import Flask, render_template, request
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the sentiment analysis model
with open("C:/Users/manya/OneDrive/Desktop/Manya/text summarizer and sentiment analysis/model/restaurant_review_model.pkl", 'rb') as f:
    model = pickle.load(f)

# Load the vectorizer
with open("C:/Users/manya/OneDrive/Desktop/Manya/text summarizer and sentiment analysis/model/vectorizer.pkl", 'rb') as f:
    vectorizer = pickle.load(f)

# Initialize the Hugging Face Summarization Pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# List to store feedback results
feedback_history = []

# Home route for sentiment analysis
@app.route('/', methods=['GET', 'POST'])
def home():
    message = None  # Default message
    sentiment = None  # Default sentiment
    if request.method == 'POST':
        review = request.form.get('text', '').strip()
        if review:
            # Preprocess and predict sentiment
            review_vectorized = vectorizer.transform([review])  # Adjust this if preprocessing is required
            prediction = model.predict(review_vectorized)
            sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

            # Custom message based on sentiment
            if sentiment == 'Positive':
                message = "Thank you for your positive response! We're glad you enjoyed your experience. Hope to see you again soon!"
            else:
                message = "We appreciate your honest feedback and will work hard to improve our services for you. Thank you for letting us know."

            # Append to feedback history
            feedback_history.append((review, sentiment, message))

    return render_template('index.html', results=feedback_history)

# Text Summarizer route
@app.route('/text_summarizer', methods=['GET', 'POST'])
def text_summarizer():
    summary = None  # Default value for no summary
    if request.method == 'POST':
        input_text = request.form['text']
        if input_text.strip():
            # Generate summary using the transformer model
            try:
                summary_result = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
                summary = summary_result[0]['summary_text']
            except Exception as e:
                summary = f"Error generating summary: {e}"

    return render_template('text_summarizer.html', summary=summary)

# Main block
if __name__ == '__main__':
    app.run(debug=True)












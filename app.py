from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Initialize the sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

# Define the route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    response = None
    if request.method == 'POST':
        # Get the user input from the form
        user_input = request.form['user_input']
        
        # Use the model to analyze sentiment
        sentiment_result = sentiment_pipeline(user_input)[0]
        
        # Get sentiment label and confidence score
        sentiment = sentiment_result['label']
        confidence = sentiment_result['score']
        
        # Return the result to be displayed in the HTML template
        response = f"Sentiment: {sentiment} with confidence of {confidence:.2f}"
    
    # Render the index.html template, passing the response if available
    return render_template('index.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)

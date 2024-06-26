from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)
model = tf.keras.models.load_model('spam_model.h5')
tokenizer = Tokenizer(num_words=5000)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    seq = tokenizer.texts_to_sequences([data])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)
    return jsonify({'prediction': 'spam' if pred[0][0] > 0.5 else 'ham'})

if __name__ == '__main__':
    app.run(debug=True)

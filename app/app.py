from flask import Flask, request, jsonify
from model import Model
from utils import labels_to_ratings, get_device
from transformers import DistilBertTokenizer
import torch

app = Flask(__name__)

model = Model()    # randomly initialized classification head
model.eval()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

@app.route('/')
def hello_world():
    return 'Hello, World!'

def get_prediction(review):
  # Tokenize input
  print("encoding review...")
  encoding = tokenizer(review)

  # Add dummy batch dimension to inputs for model
  print("adding dummy batch dim...")
  input_ids = torch.tensor(encoding['input_ids']).unsqueeze(0)
  attention_mask = torch.tensor(encoding['attention_mask']).unsqueeze(0)

  # Get model predictions
  with torch.no_grad():
    print("starting forward pass...")
    outputs = model(input_ids, attention_mask=attention_mask)
  print("processing logits...")
  pred_labels = torch.argmax(outputs['logits'], dim=1)
  pred_ratings = labels_to_ratings(pred_labels)
  pred = pred_ratings[0]
  return pred

@app.route('/predict', methods=['GET', 'POST'])
def predict():
  if request.method == 'GET':
    return 'Please send a POST request to /predict'
  if request.method == 'POST':
    print("In /predict endpoint")
    review = (request.json)['review']
    print(f"Review: {review}")
    pred = get_prediction(review)
    return jsonify({'review': review, 'pred_stars': str(pred)})

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
  encoding = tokenizer(review)
  print("encoded review")

  # Add dummy batch dimension to inputs for model
  input_ids = torch.tensor(encoding['input_ids']).unsqueeze(0)
  attention_mask = torch.tensor(encoding['attention_mask']).unsqueeze(0)
  print("added dummy batch dimension")

  # Get model predictions
  with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    print("finished model forward pass")
  pred_labels = torch.argmax(outputs['logits'], dim=1)
  pred_ratings = labels_to_ratings(pred_labels)
  pred = pred_ratings[0]
  return pred

@app.route('/predict', methods=['GET', 'POST'])
def predict():
  if request.method == 'GET':
    return 'Please send a POST request to /predict'
  if request.method == 'POST':
    review = request.json['review']
    if review is None:
      return jsonify({"pred_stars": "0"})
    pred = get_prediction(review)
    return jsonify({'review': review, 'pred_stars': pred})

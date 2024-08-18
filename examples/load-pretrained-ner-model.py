import os
import torch
from dotenv import load_dotenv
load_dotenv()
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizerFast

DIR_DATA = os.getenv("DIR_DATA")
DIR_MODEL = os.path.join(DIR_DATA, 'trials')
TRIAL_NAME = "8870e04c-af15-4c1e-9b50-56d99b4b0f35"
CHECKPOINT = "checkpoint-510"
MODEL_PATH = os.path.join(DIR_MODEL, TRIAL_NAME, CHECKPOINT)
print(f"Model Path Exists => {os.path.exists(MODEL_PATH)}")

# Index To ID
id_to_index_mapping = {0: 'I-corporation', 1: 'O', 2: 'B-location', 3: 'B-group', 4: 'I-person', 5: 'B-creative-work', 6: 'B-corporation', 7: 'I-creative-work', 8: 'I-product', 9: 'B-person', 10: 'I-location', 11: 'I-group', 12: 'B-product'}


# Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Example input text
test_texts = ["Ford Motor Company is a great company"]


# Tokenize the input texts
inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")

# Make a prediction
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

# The logits (raw model outputs) before applying softmax
logits = outputs.logits

# Convert logits to probabilities
probabilities = torch.softmax(logits, dim=-1)

# Get the predicted class (index of the max probability)
predicted_class = probabilities.argmax(dim=-1)
idx_class = predicted_class.data.numpy()[0]
idx_label = id_to_index_mapping[idx_class]
print("Predicted Label => {}".format(idx_label))


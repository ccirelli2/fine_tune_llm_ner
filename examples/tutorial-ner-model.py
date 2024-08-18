"""
Description (from webpage)
    Next we will look at token classification. Rather than classifying an
    entire sequence, this task classifies token by token. We’ll demonstrate how
    to do this with Named Entity Recognition, which involves identifying tokens
    which correspond to a predefined set of “entities”. Specifically, we’ll use the
    W-NUT Emerging and Rare entities corpus. The data is given as a collection of
    pre-tokenized documents where each token is assigned a tag.

Dataset
- Requires a special function to load.
- location is an entity type,
    B- indicates the beginning of an entity, and
    I- indicates consecutive positions of the same entity (“Empire State Building” is considered one entity).
    O indicates the token does not correspond to any entity.

References
- Source: https://huggingface.co/transformers/v3.2.0/custom_datasets.html
- Datasets: https://github.com/huggingface/datasets
- Source: https://reybahl.medium.com/token-classification-in-python-with-huggingface-3fab73a6a20e

"""
import os
import re
import torch
import numpy as np
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import list_datasets
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, DistilBertForTokenClassification, Trainer, TrainingArguments

# Globals
load_dotenv()
DIR_DATA = os.getenv("DIR_DATA")
FILE_NAME = 'wnut17train.conll'
PATH_DATA = os.path.join(DIR_DATA, FILE_NAME)
DIR_OUTPUT = os.path.join(DIR_DATA, 'trials', str(uuid4()))
if not os.path.exists(DIR_OUTPUT):
    os.makedirs(DIR_OUTPUT)

# Load Data
def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

texts, tags = read_wnut(PATH_DATA)

# Create Train Test Split
train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)

# unique_tags = set(tag for doc in tags for tag in doc)
unique_tags = set([tag for doc in tags for tag in doc])
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}
print('Unique Tags', unique_tags)
print('Tag to ID', tag2id)
print('ID to Tag', id2tag)

# Encode Tokens
# return_offsets_mapping: whether or not to return (char_start, char_end) for each token
# For each sub-token returned by the tokenizer, the offset mapping gives us a tuple indicating the sub-token’s start position and end position relative to the original token it was split from.
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

# What does the tokenizer return_offsets_mapping parameter do?
print('Encoding Keys', train_encodings.keys())
print('Encoding Tokens', train_texts[0][:10])
print('Encoding Tokens tokenized', train_encodings[0].tokens[:10])
print('Word Ids', train_encodings[0].word_ids[:10])
print('Input IDs', train_encodings['input_ids'][0][:10])
print('Offset Mappings', train_encodings[0].offsets)



# Create a Special Function to Encode Tags for Token Classification.  Some tokens are split into multiple sub-tokens.
# We need to create a new array of labels for each sub-token.  We will use -100 to indicate that the sub-token is not part of an entity.
def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)

# Create A Dataset
class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
train_dataset = WNUTDataset(train_encodings, train_labels)
val_dataset = WNUTDataset(val_encodings, val_labels)

# Load a Pretrained Model
"""
model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(unique_tags))

# Trainer
training_args = TrainingArguments(
    output_dir=DIR_OUTPUT,          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=os.path.join(DIR_OUTPUT, './logs'),            # directory for storing logs
    logging_steps=10,
)

# Train
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
"""
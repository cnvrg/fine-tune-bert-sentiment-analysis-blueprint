import os
import pathlib
import sys
scripts_dir = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scripts_dir))
import numpy as np
import torch
from transformers import Trainer
from transformers import BertTokenizer, BertForSequenceClassification
import re
import pandas as pd


def preprocess(text, stem=False):
    # TEXT CLENAING
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if stem:
            tokens.append(stemmer.stem(token))
        else:
            tokens.append(token)
    return " ".join(tokens)

# Create torch dataset
class Dataset_test(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        # item['labels'] = self.labels[idx]
        print(item)
        return item
        # input_ids = torch.tensor(self.encodings['input_ids'])
        # target_ids = torch.tensor(self.labels[idx])
        # return {"input_ids": input_ids, "labels": target_ids}

    def __len__(self):
        return len(self.encodings["input_ids"])


# Define predict function
def predict(data):
    sentiment_label = ['negative', 'neutral', 'positive']

    input_data = data["text"]
    data = pd.DataFrame([input_data], columns=["text"])
    data_input=data["text"]

    data_input = data_input.apply(lambda x: preprocess(x))

    # os.chdir("fine-tune-inference/ft-fine-tuning-inference")
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    scripts_dir = pathlib.Path(__file__).parent.resolve()
    model_path = os.path.join(scripts_dir, 'checkpoint-50')

    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    print("loaded model")

     # Tokenize text
    X_test = list(data_input.astype(str))
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=300, return_tensors='pt')
    
    # Create torch dataset
    test_dataset = Dataset_test(X_test_tokenized)

    # Define test trainer
    test_trainer = Trainer(model)

    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    

    # Predict
    score = np.argmax(raw_pred, axis=1)
    # Decode sentiment
    label = sentiment_label[int(score.round().item())]
    return {"label": label, "score": float(score)}


# import json
# data_set = {"text": 'Good Weather'}

# json_dump = json.dumps(data_set)
# print('====json_dump====', json_dump)

# data = json.loads(json_dump)
# print(type(data["text"]))

# if __name__ == "__main__":
#     result = predict(data)
#     print(result)




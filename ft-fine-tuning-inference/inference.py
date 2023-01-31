# Copyright (c) 2023 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

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
import argparse

def parse_parameters():
    parser = argparse.ArgumentParser(description="""Inference prediction by using fine-tuned bert model""")
    model_path = os.path.join(scripts_dir, 'checkpoint-50')
    parser.add_argument('-model_path', '--model_path', action='store',
                        default=model_path, required=False,
                        help="""string. path for calling the model""")

    parser.add_argument('-text', '--text_column', action='store', dest='text_column', default='text', required=False,
                        help="""string. name of text column""")
        
    parser.add_argument('-model_name', '--model_name', action='store', default="bert-base-uncased", required=False,
                        help="""string. name for calling the model""")

    parser.add_argument('-cuda_device', '--cuda_device', action='store', default="0", required=False,
                        help="""envrionment variable setup""")

    return parser.parse_args()

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

class NoModelError(Exception):
    """Raise if model folder existing"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "NoModelError: It required model folder for inference!"
        

def validate_arguments(args):
    """Validates input arguments
    Checks if the model folder existing
    Args:
        args: argparse object
    Raises:
        NoModelError: If the model folder existing
    """
    if not ("checkpoint" in args.model_path):
        raise NoModelError()

# Define predict function
def predict(data):
    args = parse_parameters()
    validate_arguments(args)
    model_path = args.model_path
    text_column = args.text_column
    model_name = args.model_name
    cuda_device = args.cuda_device

    # for unittests
    # model_path = os.path.join(scripts_dir, 'checkpoint-50')
    # text_column = 'text'
    # model_name = "bert-base-uncased"
    # cuda_device = "0"

    sentiment_label = ['negative', 'neutral', 'positive']

    input_data = data[text_column]
    data = pd.DataFrame([input_data], columns=[text_column])
    data_input=data[text_column]

    data_input = data_input.apply(lambda x: preprocess(x))

    # os.chdir("fine-tune-inference/ft-fine-tuning-inference")
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

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
# data_set = {"text": 'Bad Weather'}

# json_dump = json.dumps(data_set)
# print('====json_dump====', json_dump)

# data = json.loads(json_dump)
# print(type(data["text"]))

# if __name__ == "__main__":
#     result = predict(data)
#     # print(result)




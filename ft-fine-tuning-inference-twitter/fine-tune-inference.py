import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
#from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import EarlyStoppingCallback
import re
#import modeling
import os
import pathlib
import sys
scripts_dir = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scripts_dir))
import argparse

model_path = os.path.join(scripts_dir, 'checkpoint-50')
parser = argparse.ArgumentParser(description="""finetune pre-trained Huggingface bert model""")
parser.add_argument('-input_filename', '--input_filename', action='store', dest='input_filename', default='./1.6m_twitts_small_small_inference.csv', required=False,
                    help="""string. csv train data file""")

parser.add_argument('-model_path', '--model_path', action='store', dest='model_path', default=model_path, required=False, 
                    help="""string. path for model path""")

parser.add_argument('-result_path', '--result_path', action='store',
                    default='/cnvrg', required=False,
                    help="""string. path for saving the result""")

parser.add_argument('-text', '--text_column', action='store', dest='text_column', default='text', required=False,
                    help="""string. name of text column""")

args = parser.parse_args()
result_path = args.result_path
text_column = args.text_column
input_filename = args.input_filename
model_path = args.model_path

DATASET_COLUMNS = ["target", "timestamp", "datetime", "query", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if stem:
            tokens.append(stemmer.stem(token))
        else:
            tokens.append(token)
    return " ".join(tokens)

# Define pretrained tokenizer and model
model_name = "bert-base-uncased"
current_path = os.getcwd()
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)

# ----- 2. predicton from Fine-tune pretrained model -----#
test_data = pd.read_csv(input_filename, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
test_data[text_column] = test_data[text_column].apply(lambda x: preprocess(x))
test_data.head(15)

# Preprocess data
X_test = list(test_data[text_column].astype(str))
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=300, return_tensors='pt')

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

    def __len__(self):
        return len(self.encodings["input_ids"])

# Create torch dataset
test_dataset = Dataset_test(X_test_tokenized)

# Load trained model
model_path = os.path.join(scripts_dir, 'checkpoint-50')
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)
print("====prediction Result====", y_pred)
test_data["prediction"] = y_pred

# Save the prediction result
test_data[['text', 'prediction']].to_csv(result_path+'/prediction_result.csv', header=True, index=False)

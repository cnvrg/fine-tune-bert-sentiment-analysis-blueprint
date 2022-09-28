import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer, logging
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from transformers import EarlyStoppingCallback
import re
import os
import argparse
from pynvml import *

print("Check if GPU available", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("If CPU or GPU Selected", device)

if device == "cuda:0":
    def print_gpu_utilization():
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used//1024**2} MB.")

    def print_summary(result):
        print(f"Time: {result.metrics['train_runtime']:.2f}")
        print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
        print_gpu_utilization()

    print("====Beginning GPU Utilization===")
    print_gpu_utilization()

parser = argparse.ArgumentParser(description="""finetune pre-trained Huggingface bert model""")
parser.add_argument('-input_filename', '--input_filename', action='store', dest='input_filename', required=False,
                    help="""string. csv train data file""")

parser.add_argument('-output_model_path', '--output_model_path', action='store',
                    default='/cnvrg/output', required=False,
                    help="""string. path for saving the model""")

parser.add_argument('-text', '--text_column', action='store', dest='text_column', default='text', required=False,
                    help="""string. name of text column""")

parser.add_argument('-target', '--label_column', action='store', dest='label_column', default='target', required=False,
                    help="""string. name of label column""")

parser.add_argument('--num_train_epochs', action='store', dest='num_train_epochs', default=1, required=False,
                    help="""int. number of training epochs to run""")

parser.add_argument('--batch_size_train', action='store', dest='batch_size_train', default=256, required=False,
                    help="""int. size of each batch to train on""")

parser.add_argument('--batch_size_val', action='store', dest='batch_size_val', default=256, required=False,
                    help="""int. size of each batch to evaluate on""")

parser.add_argument('--max_length', action='store', dest='max_length', default=300, required=False,
                    help="""int. size of max length for each squence""")

args = parser.parse_args()
output_model_path = args.output_model_path
text_column = args.text_column
label_column = args.label_column
num_train_epochs = int(args.num_train_epochs)
batch_size_train = int(args.batch_size_train)
batch_size_val = int(args.batch_size_val)
max_length = int(args.max_length)
input_filename = args.input_filename

DATASET_COLUMNS = ["target", "timestamp", "datetime", "query", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

# Read data
data = pd.read_csv(input_filename, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

# decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
decode_map = {0: 0, 2: 1, 4: 2}
def decode_sentiment(label):
    return decode_map[int(label)]


data.target = data.target.apply(lambda x: decode_sentiment(x))


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


data.text = data.text.apply(lambda x: preprocess(x))
data.head(15)

# Define pretrained tokenizer and model
model_name = "bert-base-uncased"
current_path = os.getcwd()
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
data.head()

# from torchinfo import summary
# summary(model)

# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
##############Finish Printing Model Structure################

# ----- 1. Preprocess data -----#
# Preprocess data
X = list(data[text_column].astype(str))
y = list(data[label_column].astype(int))
print(X[0:15])
print(y[0:15])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
print(X_train[0:15])
print(X_val[0:15])
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        print(item)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="weighted")
    precision = precision_score(y_true=labels, y_pred=pred, average="weighted")
    f1 = f1_score(y_true=labels, y_pred=pred, average="weighted")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# ----- 3. Define Trainer -----#

# logging.set_verbosity_error()
args = TrainingArguments(
    output_dir=output_model_path,
    evaluation_strategy="epoch",
    eval_steps=500,
    per_device_train_batch_size=batch_size_train,
    per_device_eval_batch_size=batch_size_val,
    num_train_epochs=1,
    seed=0,
    load_best_model_at_end=True,
    logging_strategy="epoch",
    save_strategy="epoch",
)
trainer = Trainer(
    model=model,
    args=args,
    data_collator=DataCollatorWithPadding(tokenizer),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train pre-trained model
trainer.train()

if device == "cuda:0":
    print("====Ending GPU Utilization===")
    print_gpu_utilization()
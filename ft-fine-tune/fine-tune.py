import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer, logging
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
#from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import EarlyStoppingCallback
import re
#import modeling
import os
import argparse
import yaml
# import logging
from pynvml import *

def parse_parameters():
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
    return parser.parse_args()


def decode_sentiment(label):
    # Read config file
    with open("fine-tune-config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
    decode_map = config["DECODE_MAP"]
    return decode_map[int(label)]


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


# from torchinfo import summary
# summary(model)

def model_structure(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:config["EMBEDDING_LAYER"]]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[config["EMBEDDING_LAYER"]:config["ENCODER_LAYER"]]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[config["OUTPUT_LAYER"]:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    ##############Finish Printing Model Structure################

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
        # input_ids = torch.tensor(self.encodings['input_ids'])
        # target_ids = torch.tensor(self.labels[idx])
        # return {"input_ids": input_ids, "labels": target_ids}

    def __len__(self):
        return len(self.encodings["input_ids"])


# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="weighted")
    precision = precision_score(y_true=labels, y_pred=pred, average="weighted")
    f1 = f1_score(y_true=labels, y_pred=pred, average="weighted")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define main function:
def main():
    # Read config file
    with open("fine-tune-config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

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

    args = parse_parameters()
    output_model_path = args.output_model_path
    text_column = args.text_column
    label_column = args.label_column
    num_train_epochs = int(args.num_train_epochs)
    batch_size_train = int(args.batch_size_train)
    batch_size_val = int(args.batch_size_val)
    max_length = int(args.max_length)
    input_filename = args.input_filename
    # input_filename = '1.6m_twitts.csv'

    # Read data
    DATASET_COLUMNS = config["DATASET_COLUMNS"]
    DATASET_ENCODING = config["DATASET_ENCODING"]
    data = pd.read_csv(input_filename, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
    
    data.target = data.target.apply(lambda x: decode_sentiment(x))

    data.text = data.text.apply(lambda x: preprocess(x))

    # ----- 1. Define pretrained tokenizer and model -----#
    model_name = "bert-base-uncased"
    # model_path = "./deepspeed_model/"
    current_path = os.getcwd()
    # config = modeling.BertConfig.from_json_file("bert_1.5b_config.json")
    # model = modeling.BertForPreTraining(config)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
    #print("====testing===", torch.load("pytorch_model_self.bin"))
    #print("====testing_deepspeed===", torch.load("pytorch_model.bin"))
    #model.load_state_dict(torch.load("pytorch_model_self.bin"))
    model_structure(model)

    # ----- 2. Preprocess data -----#
    # Preprocess data
    X = list(data[text_column].astype(str))
    y = list(data[label_column].astype(int))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    # ----- 3. Fine-tune pretrained model -----#
    # logging.set_verbosity_error()
    args = TrainingArguments(
        output_dir=output_model_path,
        evaluation_strategy="epoch",
        eval_steps=config["EVAL_STPES"],
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_val,
        num_train_epochs=num_train_epochs,
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

if __name__ == "__main__":
    main()
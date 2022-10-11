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
import yaml

# Read config file
with open("fine-tune-inference-config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

def parse_parameters():
    model_path = os.path.join(scripts_dir, 'checkpoint-50')
    
    parser = argparse.ArgumentParser(description="""finetune pre-trained Huggingface bert model""")
    parser.add_argument('-input_filename', '--input_filename', action='store', dest='input_filename', default='./1.6m_twitts_small_inference.csv', required=False,
                        help="""string. csv train data file""")

    parser.add_argument('-model_path', '--model_path', action='store', dest='model_path', default=model_path, required=False, 
                        help="""string. path for model path""")

    parser.add_argument('-result_path', '--result_path', action='store',
                        default='/cnvrg', required=False,
                        help="""string. path for saving the result""")

    parser.add_argument('-text', '--text_column', action='store', dest='text_column', default='text', required=False,
                        help="""string. name of text column""")
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

# Define main function
def main():
    args = parse_parameters()
    result_path = args.result_path
    text_column = args.text_column
    input_filename = args.input_filename
    model_path = args.model_path
    # input_filename = 'fine-tune-inference/1.6m_twitts_small_small_inference.csv'

    # Define pretrained tokenizer and model
    model_name = "bert-base-uncased"
    current_path = os.getcwd()
    # config = modeling.BertConfig.from_json_file("bert_1.5b_config.json")
    # model = modeling.BertForPreTraining(config)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    #print("====testing===", torch.load("pytorch_model_self.bin"))
    #print("====testing_deepspeed===", torch.load("pytorch_model.bin"))
    #model.load_state_dict(torch.load("pytorch_model_self.bin"))

    # ----- 2. predicton from Fine-tune pretrained model -----#
    DATASET_COLUMNS = config["DATASET_COLUMNS"]
    DATASET_ENCODING = config["DATASET_ENCODING"]
    test_data = pd.read_csv(input_filename, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
    test_data[text_column] = test_data[text_column].apply(lambda x: preprocess(x))
    test_data.head(15)

    # Preprocess data
    X_test = list(test_data[text_column].astype(str))
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=300, return_tensors='pt')

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
    test_data[config["DATASET_PREDICTION"]] = y_pred

    # Save the prediction result
    test_data[[config["DATASET_TEXT"], config["DATASET_PREDICTION"]]].to_csv(result_path+config["RESULT_FILE"], header=True, index=False)

if __name__ == "__main__":
    main()

# y = list(test_data[label_column].astype(int))
# prediction_accuracy = accuracy_score(y_true=y, y_pred=y_pred)
# print("====accuracy====", prediction_accuracy)

# from sklearn.metrics import confusion_matrix
# import pylab as pl
# cm = confusion_matrix(y, y_pred)
# pl.matshow(cm)
# pl.title('Confusion matrix of the classifier')
# pl.colorbar()
# pl.show()
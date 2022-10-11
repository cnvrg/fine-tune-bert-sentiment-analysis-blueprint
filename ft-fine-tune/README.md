#  cnvrg fine tune pre-trained model for sentiment analysis

Notes for this Component - 

## How It Works

The library trains a sentiment analysis model on given labeled data and produces a prediction model and a tokenizer.
By default the library needs the receive a single path (--input_filename) for a local file.
The library performs fine-tuning of the pre-trained bert model from huggingface with ts associated tokenizer.   


## How To Run

python3 fine-tune.py --input_filename [YOUR_LABLED_DATA_FILE]

run python3 fine-tune.py -f  for info about more optional parameters of hyper parameters.
                                     
## Parameters

`--input_filename` - (String) (Required param) Path to a local labeled data file.

`--output_model_path` - (String) (Default: 'pytorch_model.bin' and 'config.json') Path for saving the model and tokenizer.

`--text_column` - (String) (Default: 'text') Name of text column in dataframe.

`--label_column` - (String) (Default: 'target') Name of label column in dataframe.

`--um_train_epochs` - (int) (Default: 1) The number of epochs the algorithm performs in the training phase.

`--max_length` - (int) (Default: 300) The number of characters per sequence.

`--batch_size_train` - (int) (Default: 256) The number of texts the model goes over in each epoch for training phase.

`--batch_size_val` - (int) (Default: 256) The number of texts the model goes over in each epoch for evaluation phase.
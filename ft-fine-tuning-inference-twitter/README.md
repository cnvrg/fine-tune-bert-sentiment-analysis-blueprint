#  cnvrg batch prediction by using fine tune pre-trained model for sentiment_analysis

Notes for this Component - 

## How It Works

The library performs batch prediction by using a fine-tuned pre-trained bert sentiment analysis model on given labeled data to produce prediction results.
By default the library needs the receive a single path (--input_filename) for a local file with dataset and a model path (--model_path) with the model, tokenizer, optimizer and etc. The library performs batch prediction by using specified model and tokenizer.  

## How To Run

python3 fine-tune-inference.py --input_filename [YOUR_LABLED_DATA_FILE] --model_path [YOUR_MODEL_PATH]

run python3 fine-tune-inference.py -f  for info about more optional parameters
                                     
## Parameters

`--input_filename` - (String) (Required param) Path to a local labeled data file which contains the data that is used for prediction.

`--model_path` - (String) (Default: 'pytorch_model.bin' and 'config.json') Model path to use for prediction.

`--result_path` - (String) (Default: '/cnvrg') Path for saving the result.

`--text_column` - (String) (Default: 'text') Name of text column in dataframe.

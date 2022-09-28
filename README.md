You can use this blueprint to fine tune a pre-trained bert model that analyzes sentiment in text using your custom data.
In order to fine tune this model with your data, you would need to provide a dataset of text sentences and sentiment pairs.
For your convenience, you can use one of S3 connector prebuilt datasets.
1. Click on `Use Blueprint` button
2. You will be redirected to your blueprint flow page
3. In the flow, edit the dataset split block to select size of your dataset
4. If you change dataset size, the default model path will need to modify accordingly. e.g. 0.01% dataset connected with checkpoint-1 folder, 1% dataset connected with checkpoint-50 folder, 2% dataset with checkpoint-100 folder and etc.
4. Click on the 'Run Flow' button
5. In a few minutes you will fine-tune a new sentiment analysis model and produce a batch prediction result.
6. Go to the 'artifacts' tab in the last batch prediction block, Fine Tune Inferencer Twitter, and look for your final result in .csv

Congrats! You have fine-tuned a pre-trained huggingface model that can analyse sentiment in text!

[See here how we created this blueprint](https://github.com/cnvrg/fine-tune-bert-sentiment-analysis-blueprint)

You can use this blueprint to perform batch prediction by using a fine-tuned bert model that analyzes sentiment in text using your custom data.
In order to perferm batch prediction with your data, you would need to provide a dataset of text sentences.
For your convenience, a default datasets is provided for prediction task with 160 sentences. 
1. Click on `Use Blueprint` button
2. You will be redirected to your blueprint flow page
3. In the flow, edit the Fine Tune Inference block to select your dataset
4. Click on the 'Run Flow' button
5. In a few minutes you will produce a batch prediction result.
6. Go to the 'artifacts' tab in the batch prediction block, Fine Tune Inferencer Twitter, and look for your final result in .csv

Congrats! You have performed a batch prediction by using a fine-tuned huggingface model that can analyse sentiment in text!

[See here how we created this blueprint](https://github.com/cnvrg/fine-tune-bert-sentiment-analysis-blueprint)


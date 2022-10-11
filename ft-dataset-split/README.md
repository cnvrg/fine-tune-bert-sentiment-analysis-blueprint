# Fine Tune Bert Model for Sentiment Analysis Twitter Dataset Split
## _cnvrg_

The Fine Tune Dataset Split library splits the downloaded dataset from S3 connector into 1% and 0.01% sets to run on CPUs. It is only able to run small dataset on large pre-trained model on CPUs.  As this library splits on twitter dataset with 1.6M into smaller dataset with same proportional number of labels as original dataset to prevent impacting the accuracy of results. 

Click [here]() for more information on this library.

## Library Flow
The following list outlines this library's high-level flow:
- The user defines a set of input arguments to specify the dataset path and intended size of the validation set.
- The library validates these arguments and then splits the original dataset into 0.01% datasets.

## Inputs
This library assumes that the user has access to the preprocessed dataset via previous libraries in the Blueprint. The input dataset must be in CSV format.
The ADTS Train-Valid Split library requires the following inputs:
* `--dataset_path` - string, required. Provide the path to the original dataset in CSV format.
* `--valid_size` - string, optional. Specify size of the validation set as a number between 0.0 and 1.0. Default value: `0.0001`, which is 1% of original dataset.

## Sample Command
Refer to the following sample command:

```bash
python dataset_split.py --dataset_path /input/dev_fine_tune_s3_connector_2/sentiment_analysis_data/1.6m_twitts.csv --valid_size 0.0001
```

## Outputs
The Fine Tune Dataset Split library generates the following outputs :
- The library generates CSV files `1.6m_twitts_small.csv`.
- The library writes these output files to the default path `/cnvrg`.

## Troubleshooting
- Ensure the input arguments to the library are valid and accurate.
- Check the experiment's Artifacts section to confirm the library has generated the output files.
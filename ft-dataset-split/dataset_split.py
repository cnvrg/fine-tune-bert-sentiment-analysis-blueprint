import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# The path for listing items
path = '/input/'
file_list = []
for path, folders, files in os.walk(path):
    for file in files:
        file_list.append(os.path.join(path, file))
 
# Loop to print each filename separately
for filename in file_list:
    print("===All Files in input===", filename)

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
DATASET_COLUMNS = ["target", "timestamp", "datetime", "query", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

def parse_parameters():
    """Command line parser."""
    parser = argparse.ArgumentParser(description="""Train-Validation Split""")
    parser.add_argument('--dataset_path', action='store', dest='dataset_path', required=False, default='/input/dev_fine_tune_s3_connector_2/sentiment_analysis_data/sen1.6m_twitts.csv',
                            help="""--- Path to the preprocessed dataset in csv format ---""")
    parser.add_argument('--valid_size', action='store', dest='valid_size', required=False, default='0.01',
                            help="""--- size of validation set as percentage of entire dataset ---""")
    parser.add_argument('--output_dir', action='store', dest='output_dir', required=False, default=cnvrg_workdir, 
                            help="""--- The path to save train and validation dataset files to ---""")
    return parser.parse_args()


def validate_arguments(args):
    """Validates input arguments
    
    Makes sure that validation size lies between 0.0 and 0.4

    Args:
        args: argparse object

    Raises:
        AssertionError: If validation size does not lie between 0.0 and 0.4
    """
    assert(
        float(args.valid_size) > 0 and float(args.valid_size) <= 0.4
    ), "Validation size needs to be a value between 0.0 and 0.4."


def split_dataset_1(df, split_size):
    """Splits preprocessed data into train and validation sets
    
    The train and validation sets are split depending on the intended size of the validation set

    Args:
        df: The preprocessed dataset as a pandas dataframe
        split_size: size of the validation set as percentage of entire dataset

    Returns:
        tdf: The train set as a pandas dataframe
        vdf: The validation set as a pandas dataframe
    """
    tdf, vdf = train_test_split(df, test_size=split_size, stratify=df['target'], random_state=1)
    tdf, idf = train_test_split(tdf, test_size=split_size/100, stratify=tdf['target'], random_state=1)
    return tdf, vdf, idf

def split_dataset_2(df, split_size):
    """Splits preprocessed data into train and validation sets
    
    The train and validation sets are split depending on the intended size of the validation set

    Args:
        df: The preprocessed dataset as a pandas dataframe
        split_size: size of the validation set as percentage of entire dataset

    Returns:
        tdf: The train set as a pandas dataframe
        vdf: The validation set as a pandas dataframe
    """

    tdf, vdf = train_test_split(df, test_size=split_size, stratify=df['target'], random_state=1)
    tdf, idf = train_test_split(tdf, test_size=split_size, stratify=tdf['target'], random_state=1)
    return tdf, vdf, idf

def tvsplit_main():
    """Command line execution."""
    # Get parameters and read preprocessed dataset file
    args = parse_parameters()
    validate_arguments(args)
    df = pd.read_csv(args.dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

    # Perform splitting
    train_df_small, valid_df_small, infer_df_small = split_dataset_1(df, float(args.valid_size))
    train_df_small_small, valid_df_small_small, infer_df_small_small = split_dataset_2(df, float(args.valid_size)/100)

    # Save the train and validation datasets in csv format
    valid_df_small.to_csv(args.output_dir+'/1.6m_twitts_small.csv', header=False, index=False)
    valid_df_small_small.to_csv(args.output_dir+'/1.6m_twitts_small_small.csv', header=False, index=False)
    infer_df_small.to_csv(args.output_dir+'/1.6m_twitts_small_inference.csv', header=False, index=False)
    infer_df_small_small.to_csv(args.output_dir+'/1.6m_twitts_small_small_inference.csv', header=False, index=False)


if __name__ == "__main__":
    tvsplit_main()
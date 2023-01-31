# Copyright (c) 2023 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

import argparse
import joblib
import os
import pandas as pd
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import unittest

def list_file():
    # The path for listing items
    path = '/input/'
    file_list = []
    for path, folders, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(path, file))
    
    # Loop to print each filename separately
    for filename in file_list:
        print("===All Files in input===", filename)

# Read config file
with open("dataset_split-config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


def parse_parameters():
    """Command line parser."""
    cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
    parser = argparse.ArgumentParser(description="""Train-Validation Split""")
    parser.add_argument('--dataset_path', action='store', dest='dataset_path', required=False, default='/input/dev_fine_tune_s3_connector_2/sentiment_analysis_data/sen1.6m_twitts.csv',
                            help="""--- Path to the preprocessed dataset in csv format ---""")
    parser.add_argument('--valid_size', action='store', dest='valid_size', required=False, default='0.0001',
                            help="""--- size of validation set as percentage of entire dataset ---""")
    parser.add_argument('--output_dir', action='store', dest='output_dir', required=False, default=cnvrg_workdir, 
                            help="""--- The path to save train and validation dataset files to ---""")
    return parser.parse_args()


def test_connection(self):
    actual = self.con.connect()
    self.assertTrue(actual)

def validate_arguments(args):
    """Validates input arguments
    
    Makes sure that validation size lies between 0.0 and 0.4

    Args:
        args: argparse object

    Raises:
        Exception: If validation size does not lie between 0.0 and 0.4
    """
    while True:
        try:
            input = float(args.valid_size)
        except ValueError: # just catch the exceptions you know!
            print ('That\'s not a number!')
        else:
            if 0.0 <= input < 0.4: 
                break
            else:
                # print ('Validation size needs to be a value between 0.0 and 0.4.')
                raise ValueError("Validation size needs to be a value between 0.0 and 0.4.")

    # assert(
    #     float(args.valid_size) > 0 and float(args.valid_size) <= 0.4
    # ), "Validation size needs to be a value between 0.0 and 0.4."


def split_dataset(df, split_size):
    """Splits preprocessed data into train and validation sets
    
    The train and validation sets are split depending on the intended size of the validation set

    Args:
        df: The preprocessed dataset as a pandas dataframe
        split_size: size of the validation set as percentage of entire dataset

    Returns:
        tdf: The train set as a pandas dataframe
        vdf: The validation set as a pandas dataframe
        idf: The inference set as a pandas dataframe
    """
    tdf, vdf = train_test_split(df, test_size=split_size, stratify=df['target'], random_state=1)
    tdf, idf = train_test_split(tdf, test_size=split_size, stratify=tdf['target'], random_state=1)
    return tdf, vdf, idf

def tvsplit_main():
    """Command line execution."""
    DATASET_COLUMNS = config["DATASET_COLUMNS"]
    DATASET_ENCODING = config["DATASET_ENCODING"]
    # Get parameters and read preprocessed dataset file
    # list_file() # check the default files in the converge path
    args = parse_parameters()
    validate_arguments(args)
    df = pd.read_csv(args.dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

    # Perform splitting
    train_df_small, valid_df_small, infer_df_small = split_dataset(df, float(args.valid_size))

    # Save the train and validation datasets in csv format
    # train_df.to_csv(args.output_dir+'/train.csv', index=False)
    valid_df_small.to_csv(args.output_dir+config["SMALL_DATASET"], header=False, index=False)
    infer_df_small.to_csv(args.output_dir+config["INFERENCE_DATASET"], header=False, index=False)


if __name__ == "__main__":
    tvsplit_main()
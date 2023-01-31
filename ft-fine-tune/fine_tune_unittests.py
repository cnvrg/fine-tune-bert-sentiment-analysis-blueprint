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

import pandas as pd
import unittest
from finetune import decode_sentiment
import os
import pathlib as pl

print(os.getcwd())

class TestFT(unittest.TestCase):
    def setUp(self):
        """Overrides setUp from unittest to create a toy pandas dataframe for testing"""
        # Here anomalies and non-anomalies are represented by -1 and 1 respectively.
        self.train_data = [
            [
                0,
                2015361140,
                'Wed Jun 03 04:27:30 PDT 2009',
                'NO_QUERY',
                'BabyHaroldK',
                "@YoungMrFudge Looks like you will be in the top 20 with me my friend....but site won't let me vote today anyway" ,
            ],
            [
                4,
                1677609543,
                'Wed Jun 03 04:27:30 PDT 2009',
                'NO_QUERY',
                'emilymatthews',
                'woke up at 10  going shopping nowwww! xo',
            ],
            [
                0,
                2326013649,
                'Thu Jun 25 06:25:18 PDT 2009',
                'NO_QUERY',
                'nickie_d_2009',
                "here at rita's. gettin ready then i dk what? i miss sammy" ,
            ],
            [
                0,
                1834993130,
                'Mon May 18 04:50:07 PDT 2009',
                'NO_QUERY',
                'hannahsharred',
                'Go away rain',
            ],
            [
                0,
                1883045755,
                'Fri May 22 07:55:09 PDT 2009',
                'NO_QUERY',
                'seregon',
                'My good mood has been completely ruined ',
            ],
         ]
        self.valid_data = [
            [
                0,
                2178473742,
                'Mon Jun 15 07:45:14 PDT 2009',
                'NO_QUERY',
                'debsylou',
                '@imBdW neither...5pm is more like 8pm ',
            ],
            [
                4,
                1882301030,
                'Fri May 22 06:37:15 PDT 2009',
                'NO_QUERY',
                'JulzM',
                '@crackbarbie This is my road kill http://twitpic.com/6m1q .. ',
            ],
            [
                0,
                2055350279,
                'Sat Jun 06 09:09:23 PDT 2009',
                'NO_QUERY',
                'TreylinRae',
                "Stopping to get food and a floatie for brodie and then on the way to raging waters in san dimas!! Woo! I'm so tired  haha",
            ],
        ]
        self.df_columns = [
            "target",
            "timestamp",
            "datetime",
            "query",
            "user",
            "text",
        ]
        self.train_df = pd.DataFrame(self.train_data, columns=self.df_columns)
        self.valid_df = pd.DataFrame(self.valid_data, columns=self.df_columns)

        # Define dummy arguments
        self.input_filename = "./train.csv"

        # Define lists for testing map_labels
        self.trainy = pd.Series([0, 4, 0, 0, 0])
        self.validy = pd.Series([0, 4, 0])

        # Define +/- labels
        self.positive_label = "4"
        self.neutral_label = "2"
        self.negative_label = "0"

        # Expected values
        self.train_mapped = pd.core.series.Series([0, 2, 0, 0, 0])
        self.valid_mapped = pd.core.series.Series([0, 2, 0])

    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

    def test_data_paths(self):
        """Checks if input file provided"""
        file = pl.Path("./1.6m_twitts.csv")
        self.assertIsFile(file)

    def test_return_type(self):
        """Checks if the function returns pd.core.series.Series"""
        returned_lists_trainy = self.trainy.apply(lambda x: decode_sentiment(x))
        returned_lists_validy = self.validy.apply(lambda x: decode_sentiment(x))
        self.assertIsInstance(returned_lists_trainy, pd.core.series.Series)
        self.assertIsInstance(returned_lists_validy, pd.core.series.Series)

    def test_list_values(self):
        """Checks if function returns list with just '2, 1, 0's"""
        returned_lists_trainy = self.trainy.apply(lambda x: decode_sentiment(x))
        # print("returned_lists_trainy", returned_lists_trainy)
        returned_lists_validy = self.validy.apply(lambda x: decode_sentiment(x))
        # print("returned_lists_validy", returned_lists_validy)
        self.assertEqual(all(returned_lists_trainy), all(self.train_mapped))
        self.assertEqual(all(returned_lists_validy), all(self.valid_mapped))

if __name__ == "__main__":
    unittest.main()
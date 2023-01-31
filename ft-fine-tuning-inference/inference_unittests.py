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
from inference import validate_arguments, NoModelError, predict
import os
import json
import pathlib as pl

print(os.getcwd())

class TestFT(unittest.TestCase):
    def setUp(self):
        """Overrides setUp from unittest to create a toy pandas dataframe for testing"""
        # Here anomalies and non-anomalies are represented by -1 and 1 respectively.
        self.train_data = [
            {
                "text": "@YoungMrFudge Looks like you will be in the top 20 with me my friend....but site won't let me vote today anyway" ,
            },
            {
                "text": 'woke up at 10  going shopping nowwww! xo',
            },
            {
                "text": "here at rita's. gettin ready then i dk what? i miss sammy" ,
            },
            {
                "text": 'Go away rain',
            },
            {
                "text": 'My good mood has been completely ruined ',
            },
         ]
        
        # Define dummy arguments
        self.model_path = "checkpoint-50"

        # Expected values
        self.train_result = [{'label': 'negative', 'score': 0.0}, 
                            {'label': 'positive', 'score': 2.0}, 
                            {'label': 'negative', 'score': 0.0}, 
                            {'label': 'negative', 'score': 0.0}, 
                            {'label': 'negative', 'score': 0.0}]

    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

    def assertIsDir(self, path):
        if not pl.Path(path).resolve().is_dir():
            raise AssertionError("Folder does not exist: %s" % str(path))

    def test_model_paths(self):
        """Checks if model folder provided"""
        path = pl.Path("./checkpoint-50")
        self.assertIsDir(path)

    def test_result(self):
        """Checks if the result is as expected"""
        result_all = []
        for data_set in self.train_data:
            json_dump = json.dumps(data_set)
            data = json.loads(json_dump)
            result = predict(data)
            result_all.append(result)
        # print("result_all", result_all)
        self.assertEqual(result_all, self.train_result)



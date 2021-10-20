# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Writes CSV data to SQLite.

Reads from the all_data.csv file created in preprocess.sh. Since the data in
all_data.csv is not sorted by user, this file sorts the data by user. For each
user, it then gathers the user's tweets, does some minimal preprocessing where
punctuation is removed, all text is lowercased, and features are reduced. The
reduced features are: numbers -> <NUMERIC> @... -> <USERNAME> http... -> <URL>.
The preprocessed tweets are then written to a normalized_lines.txt file, which
contains all preprocessed tweets and aids in the creation of vocab.txt. The
preprocessed tweets, alongside other user metadata, are finally converted to
NumPy arrays and written to SQLite.

It is important to note that on the original data set, the training data only
contains positive and negative labels, 4 and 0, respectively. However, the
testing data contains positive, neutral, and negative labels, 4, 2, and 0,
respectively. To maintain consistency with the leaf repository, this file
converts positive labels (4) to 1 and neutral (2) and negative labels (0) to 0.
"""
import csv
import os
import re
import string

from typing import List, Dict
from absl import app
from absl import flags
import fedjax
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('out_dir', '/Users',
                    'Path for the sqlite file and normalized_lines.txt file.')

flags.DEFINE_string('data_dir', '/tmp/sent140',
                    'Path for Sent140 data set.')


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  data_file = os.path.join(FLAGS.data_dir, 'all_data.csv')
  data = {}
  raw_data = []
  with open(data_file, 'rt', encoding='ISO-8859-1') as f:
    reader = csv.reader(f)
    raw_data = list(reader)
    for line in raw_data:
      user = line[4]
      if user not in data:
        data[user] = [line]
      else:
        data[user].append(line)

  with fedjax.SQLiteFederatedDataBuilder(
      os.path.join(FLAGS.out_dir, 'sent140_dataset.sqlite')) as builder:
    builder.add_many(
        (cid.encode(), user_data_to_numpy(cd)) for cid, cd in data.items())

  # Writing normalized lines to file for word count for vocabulary generation.
  with open(
      os.path.join(FLAGS.out_dir, 'normalized_lines.txt'),
      'w') as file:
    for row in raw_data:
      file.write(normalize_tweet(row[5]))
      file.write('\n')


def user_data_to_numpy(user_data: List[List[str]]) -> Dict[str, np.ndarray]:
  """Formats a user's data as NumPy arrays.

  Normalizes tweets with normalize_tweet.

  Args:
    user_data: An array containing a user's data.

  Returns:
    A dict[str, np.ndarray]
  """
  tweet_id = []
  tweet_date = []
  tweet_query = []
  tweet_client = []
  tweet_text = []
  tweet_dataset = []
  tweet_sentiment = []

  for row in user_data:
    # Converts positive sentiment label of 4 to 1. Converts neutral and negative
    # sentiment labels of 2 and 0 to 0. Done to maintain homogeneity with leaf
    # repository.
    y = '1' if row[0] == '4' else '0'
    tweet_id.append(bytes(row[1], 'utf-8'))
    tweet_date.append(bytes(row[2], 'utf-8'))
    tweet_query.append(bytes(row[3], 'utf-8'))
    tweet_client.append(bytes(row[4], 'utf-8'))
    tweet_text.append(bytes(normalize_tweet(row[5]), 'utf-8'))
    tweet_dataset.append(bytes(row[6], 'utf-8'))
    tweet_sentiment.append(bytes(y, 'utf-8'))

  np_data = {
      'tweet_id': np.array(tweet_id, dtype=np.object),
      'tweet_date': np.array(tweet_date, dtype=np.object),
      'tweet_query': np.array(tweet_query, dtype=np.object),
      'tweet_client': np.array(tweet_client, dtype=np.object),
      'tweet_text': np.array(tweet_text, dtype=np.object),
      'tweet_dataset': np.array(tweet_dataset, dtype=np.object),
      'tweet_sentiment': np.array(tweet_sentiment, dtype=np.object)
  }

  return np_data


def normalize_tweet(tweet: str) -> str:
  """Normalizes the tweet.

  1. Removes punctuation and decases the tweet.
  2. Usernames (@...) are replaced with <USERNAME>.
  3. Urls (http...) are replaced with <URL>.
  4. Numeric values are replaced with <NUMERIC>.

  Args:
    tweet: single tweet from sent140 data set.

  Returns:
    The normalized tweet
  """
  decased = tweet.lower()
  no_urls = re.sub(r'http\S+', ' URL ', decased)
  no_usernames = re.sub(r'@\S+', ' USERNAME ', no_urls)
  no_numerics = re.sub(r'\d+(,*)\d*(\.*)\d*(-*)\d*', ' NUMERIC ', no_usernames)
  no_punc = ' '.join([
      # strips every punctuation but apostrophes and hyphens
      w.strip(string.punctuation[:6] + string.punctuation[7:12] +
              string.punctuation[13:])
      for w in re.findall(r'\w+|[^\s\w]+', no_numerics)
  ])
  no_double_hyphen = re.sub(r'-{2,}', ' ', no_punc)
  format_apostrophes = re.sub(r" ' ", "'", no_double_hyphen)
  format_hyphens = re.sub(r' - ', '-', format_apostrophes)
  single_spaces = ' '.join(format_hyphens.split())

  return single_spaces


if __name__ == '__main__':
  app.run(main)

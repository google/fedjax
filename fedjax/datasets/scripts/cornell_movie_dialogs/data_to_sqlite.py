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
# limitations under the License.Let's use the full copyright notice
"""This script is to be run by build_dataset.sh.

It turns the data from txt files to a python dictionary,
and then using SQLiteFederatedDataBuilder it moves the dictionary to SQLite.
This file also outputs the normalized_lines text file to be used for the
vocabulary file with most frequent words.
"""

import os
import re
import string

from absl import app
from absl import flags
import fedjax
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/cornell/cornell_dataset',
                    'Path for cornell dataset.')

flags.DEFINE_string(
    'out_dir', '/dev/cornell',
    'Path for the sqlite file and normalized_lines.txt file.')


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  federated_dataset_dict = {}
  # Get line id and dialogue data from movie_lines.txt file.
  with open(
      os.path.join(FLAGS.data_dir, 'movie_lines.txt'),
      'r',
      encoding='iso-8859-1') as text:
    for line in text:
      line_id, client_id, _, _, dialogue = line.split(' +++$+++ ')
      if ' CONTINUED:' in dialogue:
        continue
      normalized_string = normalize_dialogue_string(dialogue)
      if not normalized_string or normalized_string.isspace():
        continue
      if client_id not in federated_dataset_dict:
        federated_dataset_dict[client_id] = {
            'line_id': [line_id],
            'line': [normalized_string]
        }
      else:
        federated_dataset_dict[client_id]['line_id'].append(line_id)
        federated_dataset_dict[client_id]['line'].append(normalized_string)

  # Writing normalized lines to file for word count for vocabulary generation.
  with open(os.path.join(FLAGS.out_dir, 'normalized_lines.txt'),
            'w') as file:
    for client_id in federated_dataset_dict:
      for line in federated_dataset_dict[client_id]['line']:
        file.write(line)
        file.write('\n')
  # Get name, movie id, and gender data from movie_characters_metadata.txt file.
  with open(
      os.path.join(FLAGS.data_dir,
                   'movie_characters_metadata.txt'),
      'r',
      encoding='iso-8859-1') as text:
    for line in text:
      client_id, name, movie_id, _, gender, _ = line.split(' +++$+++ ')
      if client_id in federated_dataset_dict:
        num_examples = len(federated_dataset_dict[client_id]['line'])

        federated_dataset_dict[client_id]['name'] = np.full(num_examples, name)
        federated_dataset_dict[client_id]['movie'] = np.full(
            num_examples, movie_id)
        federated_dataset_dict[client_id]['gender'] = np.full(
            num_examples, gender)

  for client_id in federated_dataset_dict:
    federated_dataset_dict[client_id]['line_id'] = np.array(
        federated_dataset_dict[client_id]['line_id'])
    federated_dataset_dict[client_id]['line'] = np.array(
        federated_dataset_dict[client_id]['line'])

  with fedjax.SQLiteFederatedDataBuilder(
      os.path.join(FLAGS.out_dir,
                   'cornell_dataset.sqlite')) as builder:
    builder.add_many(
        (cid.encode(), cds) for cid, cds in federated_dataset_dict.items())


def normalize_dialogue_string(dialogue: str) -> str:
  """Normalizes the lines (dialogue).

  1. Removes newlines, punctuation and decases the string.
  2. Removes ellipses.
  3. Removes double hyphens.
  4. Numeric values are replaced with <NUMERIC>.
  3. Repalces any two or more spaces with one space.

  Args:
    dialogue: line of dialogue from the movie_lines.txt file

  Returns:
    The normalized string
  """
  strip_newline = dialogue.strip('\n')
  no_html_tags = re.sub(r'</?.>', '', strip_newline)
  no_punc_decased = ' '.join(
      [w.strip(string.punctuation) for w in no_html_tags.lower().split(' ')])
  no_ellipses = re.sub(r'(\.+)\1{1,}', ' ', no_punc_decased)
  no_double_hyphen = re.sub(r'--', ' ', no_ellipses)
  no_numbers = re.sub(r' \d+(,\d\d\d)*(\.\d+)?', ' <NUMERIC> ',
                      '' + no_double_hyphen + '').strip()
  return re.sub(r'\s{2,}', ' ', no_numbers)


if __name__ == '__main__':
  app.run(main)

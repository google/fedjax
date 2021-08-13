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
#
#!/bin/bash
#
# Downloads the sent140 dataset with minimal preprocessing and writes it to an
# SQLite file using sqlite_federated_data.SQLiteFederatedDataBuilder. Users
# should pass two flags that give the path to where they want the output files
# written to and to what temporary directory they want intermediate steps
# written to. An optional third flag for the fraction of the dataset that the
# user wants is also available (defaults to 0.1 or ten percent).
# input -o output_dir -t temp_dir -f fraction_of_data.
#
# In the end, the user is left with three files in the output_dir (-o):
# normalized_lines.txt, vocab.txt, and sent140_dataset.sqlite.
# normalized_lines.txt is a text file where every line is a normalized tweet
# from the sent140 dataset. vocab.txt is a text file where every line is a
# word from the sent140 dataset, sorted from most occurring to least occurring.
# sent140_dataset.sqlite is an SQLite file that contains the Sent140 dataset.

set -euo pipefail


tmp_dir=/tmp/sent140
size=0.1
while getopts ":o:t:f:" options; do
  case $options in
    o) out_dir="${OPTARG}";;
    t) tmp_dir="${OPTARG}";;
    f) fraction="${OPTARG}";;
    \?) echo "Usage: build_dataset.sh -o output_dir [-t temp_dir] [-f fraction_of_dataset]"; exit 0;;
  esac
done

if [ ! -d "${out_dir}" ]; then
  mkdir -p "${out_dir}"
fi
if [ ! -d "${tmp_dir}" ]; then
  mkdir -p "${tmp_dir}"
fi

# Downloads the original Sent140 data set from Stanford website.
wget --no-check-certificate \
    http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip \
    -O "${tmp_dir}/trainingandtestdata.zip"
unzip "${tmp_dir}/trainingandtestdata.zip" -d "{$tmp_dir}"
mv "${tmp_dir}/training.1600000.processed.noemoticon.csv" \
    "${tmp_dir}/training.csv"
mv "${tmp_dir}/testdata.manual.2009.06.14.csv" "${tmp_dir}/test.csv"
rm "${tmp_dir}/trainingandtestdata.zip"


# Combines testing and training files into a single file.
sed 's/$/,training/' < "${tmp_dir}/training.csv" > "${tmp_dir}/merged_data.csv"
sed 's/$/,test/' < "${tmp_dir}/test.csv" >> "${tmp_dir}/merged_data.csv"

# Reduces the size of the data set
num_lines=$(echo "1600498*${fraction}/1" | bc)
head -n ${num_lines} "${tmp_dir}/merged_data.csv" > "${tmp_dir}/all_data.csv"

python3 data_to_sqlite.py -out_dir ${out_dir} -data_dir ${tmp_dir}

# From normalized_lines.txt, create a file with all words from the dialogue,
# most common to least common.
sed 's/ /\n/g' "${out_dir}/normalized_lines.txt" \
    | sort | uniq -c | sort -k1 -nr > "${tmp_dir}/numbered.txt"
awk '{$1=""; print $0}' "${tmp_dir}/numbered.txt" > "${out_dir}/vocab.txt"


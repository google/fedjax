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

#!/bin/bash

# Downloads cornell dataset and unzip it, input -d data_dir, -o output_dir
# -k keep_intermediate_output


# Calls data_to_sqlite.py to write dataset to SQLite.
# Makes vocabulary file with normalized_lines.txt
# of most to least common words in dialogs.

set -euo pipefail
# Flags for temporary directory and keep intermediate output
data_dir=/tmp/cornell
output_dir=/tmp/cornell
keep_intermediate_output=0
while getopts ":d:o:k" options; do
  case $options in
    d) data_dir=${OPTARG};;
    o) output_dir=${OPTARG};;
    k) keep_intermediate_output=1;;
    \?) echo "Usage: build_dataset.sh [-d data_dir] [-o output_dir]  [-k]"; exit 0;;
  esac
done

mkdir -p "${data_dir}"

# Download and unzip, and rename cornell movie dialogs
wget "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip" \
    -O "${data_dir}/cornell.zip"

unzip "${data_dir}/cornell.zip" -d "${data_dir}"

mv "${data_dir}/cornell movie-dialogs corpus" "${data_dir}/cornell_dataset"


python3 data_to_sqlite.py -data_dir "${data_dir}/cornell_dataset" \
    -output_dir "${output_dir}"


# From normalized_lines.txt, create a file with all words from the dialogue,
# most common to least common.
sed 's/ /\n/g' "${output_dir}/normalized_lines.txt" \
    | sort | uniq -c | sort -k1 -nr > "${output_dir}/numbered.txt"

awk '{$1=""; print $0}' "${output_dir}/numbered.txt" > "${output_dir}/vocab.txt"

if (( ${keep_intermediate_output} != 1 )); then
  rm "${output_dir}/numbered.txt"
  rm "${output_dir}/normalized_lines.txt"
  rm "${data_dir}/cornell.zip"
  rm -r "${data_dir}/cornell_dataset"
  rm -r "${data_dir}/__MACOSX"
fi

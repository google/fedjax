#!/bin/bash
#
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

source gbash.sh || exit
source module gbash_unit.sh

PREFIX=third_party/py/fedjax/experiments/fed_avg

function test::fed_avg() {
  ROOT_DIR="${TEST_TMPDIR}/run_fed_avg"
  rm -R -f "${ROOT_DIR}"
  "${PREFIX}/run_fed_avg.par" \
    -root_dir="${ROOT_DIR}" \
    -data_mode=sstable \
    -alsologtostderr \
    -task=EMNIST_CONV \
    -server_optimizer=adam \
    -num_rounds=5 \
    -num_clients_per_round=3 \
    -eval_frequency=3
  EXPECT_FILE_DIRECTORY "${ROOT_DIR}"
}

gbash::unit::main "$@"

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
"""Shakespeare recurrent models."""

from fedjax.core import metrics
from fedjax.core import models

import haiku as hk
import jax.numpy as jnp


def create_lstm_model(vocab_size: int = 86,
                      embed_size: int = 8,
                      lstm_hidden_size: int = 256,
                      lstm_num_layers: int = 2) -> models.Model:
  """Creates LSTM language model.

  Character-level LSTM for Shakespeare language model.
  Defaults to the model used in:

  Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629

  Args:
    vocab_size: The number of possible output characters. This does not include
      special tokens like PAD, BOS, EOS, or OOV.
    embed_size: Embedding size for each character.
    lstm_hidden_size: Hidden size for LSTM cells.
    lstm_num_layers: Number of LSTM layers.

  Returns:
    Model.
  """
  # TODO(jaero): Replace these with direct references from dataset.
  pad = 0
  bos = vocab_size + 1
  eos = vocab_size + 2
  oov = vocab_size + 3
  full_vocab_size = vocab_size + 4
  # We do not guess EOS, and if we guess OOV, it's treated as a mistake.
  logits_mask = [0. for _ in range(full_vocab_size)]
  for i in (pad, bos, eos, oov):
    logits_mask[i] = jnp.NINF
  logits_mask = tuple(logits_mask)

  def forward_pass(batch):
    x = batch['x']
    # [time_steps, batch_size, ...].
    x = jnp.transpose(x)
    # [time_steps, batch_size, embed_dim].
    embedding_layer = hk.Embed(full_vocab_size, embed_size)
    embeddings = embedding_layer(x)

    lstm_layers = []
    for _ in range(lstm_num_layers):
      lstm_layers.extend([hk.LSTM(hidden_size=lstm_hidden_size), jnp.tanh])
    rnn_core = hk.DeepRNN(lstm_layers)
    initial_state = rnn_core.initial_state(batch_size=embeddings.shape[1])
    # [time_steps, batch_size, hidden_size].
    output, _ = hk.static_unroll(rnn_core, embeddings, initial_state)

    output = hk.Linear(full_vocab_size)(output)
    # [batch_size, time_steps, full_vocab_size].
    output = jnp.transpose(output, axes=(1, 0, 2))
    return output

  def train_loss(batch, preds):
    """Returns average token loss per sequence."""
    targets = batch['y']
    per_token_loss = metrics.unreduced_cross_entropy_loss(targets, preds)
    # Don't count padded values in loss.
    per_token_loss *= targets != pad
    return jnp.mean(per_token_loss, axis=-1)

  transformed_forward_pass = hk.transform(forward_pass)
  return models.create_model_from_haiku(
      transformed_forward_pass=transformed_forward_pass,
      sample_batch={
          'x': jnp.zeros((1, 1), dtype=jnp.int32),
          'y': jnp.zeros((1, 1), dtype=jnp.int32),
      },
      train_loss=train_loss,
      eval_metrics={
          'accuracy_in_vocab':
              metrics.SequenceTokenAccuracy(
                  masked_target_values=(pad, eos), logits_mask=logits_mask),
          'accuracy_no_eos':
              metrics.SequenceTokenAccuracy(masked_target_values=(pad, eos)),
          'num_tokens':
              metrics.SequenceTokenCount(masked_target_values=(pad,)),
          'sequence_length':
              metrics.SequenceLength(masked_target_values=(pad,)),
          'sequence_loss':
              metrics.SequenceCrossEntropyLoss(masked_target_values=(pad,)),
          'token_loss':
              metrics.SequenceTokenCrossEntropyLoss(
                  masked_target_values=(pad,)),
          'token_oov_rate':
              metrics.SequenceTokenOOVRate(
                  oov_target_values=(oov,), masked_target_values=(pad,)),
      })

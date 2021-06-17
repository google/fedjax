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
"""Stack Overflow recurrent models."""

from typing import Optional

from fedjax.core import metrics
from fedjax.core import models

import haiku as hk
import jax.numpy as jnp


def create_lstm_model(vocab_size: int = 10000,
                      embed_size: int = 96,
                      lstm_hidden_size: int = 670,
                      lstm_num_layers: int = 1,
                      share_input_output_embeddings: bool = False,
                      expected_length: Optional[float] = None) -> models.Model:
  """Creates LSTM language model.

  Word-level language model for Stack Overflow.
  Defaults to the model used in:

  Adaptive Federated Optimization
    Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush,
    Jakub Konečný, Sanjiv Kumar, H. Brendan McMahan.
    https://arxiv.org/abs/2003.00295

  Args:
    vocab_size: The number of possible output words. This does not include
      special tokens like PAD, BOS, EOS, or OOV.
    embed_size: Embedding size for each word.
    lstm_hidden_size: Hidden size for LSTM cells.
    lstm_num_layers: Number of LSTM layers.
    share_input_output_embeddings: Whether to share the input embeddings with
      the output logits.
    expected_length: Expected average sentence length used to scale the training
      loss down by `1. / expected_length`. This constant term is used so that
      the total loss over all the words in a sentence can be scaled down to per
      word cross entropy values by a constant factor instead of dividing by
      number of words which can vary across batches. Defaults to no scaling.

  Returns:
    Model.
  """
  # TODO(jaero): Replace these with direct references from dataset.
  pad = 0
  bos = 1
  eos = 2
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
      lstm_layers.extend([
          hk.LSTM(hidden_size=lstm_hidden_size),
          jnp.tanh,
          # Projection changes dimension from lstm_hidden_size to embed_size.
          hk.Linear(embed_size)
      ])
    rnn_core = hk.DeepRNN(lstm_layers)
    initial_state = rnn_core.initial_state(batch_size=embeddings.shape[1])
    # [time_steps, batch_size, hidden_size].
    output, _ = hk.static_unroll(rnn_core, embeddings, initial_state)

    if share_input_output_embeddings:
      output = jnp.dot(output, jnp.transpose(embedding_layer.embeddings))
      output = hk.Bias(bias_dims=[-1])(output)
    else:
      output = hk.Linear(full_vocab_size)(output)
    # [batch_size, time_steps, full_vocab_size].
    output = jnp.transpose(output, axes=(1, 0, 2))
    return output

  def train_loss(batch, preds):
    """Returns total loss per sentence optionally scaled down to token level."""
    targets = batch['y']
    per_token_loss = metrics.unreduced_cross_entropy_loss(targets, preds)
    # Don't count padded values in loss.
    per_token_loss *= targets != pad
    sentence_loss = jnp.sum(per_token_loss, axis=-1)
    if expected_length is not None:
      return sentence_loss * (1. / expected_length)
    return sentence_loss

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
          'truncation_rate':
              metrics.SequenceTruncationRate(
                  eos_target_value=eos, masked_target_values=(pad,)),
      })

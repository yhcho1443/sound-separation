# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a MixIT separation model on FUSS using tf.estimator."""


import argparse
import os
import sys

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import tensorflow.compat.v1 as tf

cur_path = os.path.dirname(os.path.realpath(__file__))
models_path = os.path.dirname(cur_path)
for p in (cur_path, models_path):
  if p not in sys.path:
    sys.path.insert(0, p)

import model
from train import data_io
from train import train_with_estimator


def main():
  tf.disable_v2_behavior()
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser(
      description='Train the MixIT separation model on FUSS.')
  parser.add_argument(
      '-dd', '--data_dir', help='Data directory.',
      required=True)
  parser.add_argument(
      '-md', '--model_dir', help='Directory for checkpoints and summaries.',
      required=True)
  parser.add_argument(
      '-bs', '--batch_size', type=int, default=3,
      help='Effective MoM batch size (actual input batch is 2 * batch_size).')
  parser.add_argument(
      '-ts', '--train_steps', type=int, default=20000000,
      help='Maximum number of training steps.')
  parser.add_argument(
      '-ee', '--eval_examples', type=int, default=800,
      help='Number of validation examples per evaluation.')
  parser.add_argument(
      '-scs', '--save_checkpoints_secs', type=int, default=600,
      help='Save checkpoints every N seconds.')
  parser.add_argument(
      '-sss', '--save_summary_steps', type=int, default=1000,
      help='Save TensorBoard summaries every N steps.')
  parser.add_argument(
      '-pr', '--parallel_readers', type=int, default=16,
      help='Number of parallel dataset prefetch readers (lower avoids Colab OOM).')
  parser.add_argument(
      '--lr', type=float, default=1e-4,
      help='Initial learning rate.')
  args = parser.parse_args()

  hparams = model.get_model_hparams()
  hparams.sr = 16000.0
  hparams.lr = args.lr

  roomsim_params = {
      'num_sources': len(hparams.signal_names),
      'num_receivers': 1,
      'num_samples': int(hparams.sr * 10.0),
  }
  tf.logging.info('Params: %s', roomsim_params.values())

  feature_spec = data_io.get_roomsim_spec(**roomsim_params)
  inference_spec = data_io.get_inference_spec()

  train_list = os.path.join(args.data_dir, 'train_example_list.txt')
  validation_list = os.path.join(args.data_dir, 'validation_example_list.txt')

  params = {
      'feature_spec': feature_spec,
      'inference_spec': inference_spec,
      'hparams': hparams,
      'io_params': {'parallel_readers': args.parallel_readers,
                    'num_samples': int(hparams.sr * 10.0)},
      'input_data_train': train_list,
      'input_data_eval': validation_list,
      'model_dir': args.model_dir,
      # Effective batch size of args.batch_size, since batches split in half to create MoMs.
      'train_batch_size': 2 * args.batch_size,
      'eval_batch_size': 2 * args.batch_size,
      'train_steps': args.train_steps,
      'eval_suffix': 'validation',
      'eval_examples': args.eval_examples,
      'save_checkpoints_secs': args.save_checkpoints_secs,
      'save_summary_steps': args.save_summary_steps,
      'keep_checkpoint_every_n_hours': 4,
      'write_inference_graph': True,
      'randomize_training': True,
  }
  tf.logging.info(params)
  params['input_data_train'] = data_io.read_lines_from_file(
      params['input_data_train'], skip_fields=1)
  params['input_data_eval'] = data_io.read_lines_from_file(
      params['input_data_eval'], skip_fields=1)
  train_with_estimator.execute(model.model_fn, data_io.input_fn, **params)


if __name__ == '__main__':
  main()

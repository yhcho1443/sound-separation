# Copyright 2020 Google LLC
# Licensed under the Apache License, Version 2.0.
"""Generate a small synthetic FUSS-format dataset for quick Colab smoke tests."""

import argparse
import os
import numpy as np
from scipy.io import wavfile


def generate_dummy_fuss(data_dir, num_train=12, num_val=6, num_eval=6, sr=16000, duration=10.0):
  """Creates synthetic FUSS-format wav files and example lists."""
  num_samples = int(sr * duration)
  t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
  freqs = [220.0, 330.0, 440.0, 550.0, 660.0, 880.0]

  splits = {
      'train': num_train,
      'validation': num_val,
      'eval': num_eval,
  }

  for split, count in splits.items():
    split_dir = os.path.join(data_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    list_path = os.path.join(data_dir, f'{split}_example_list.txt')
    lines = []

    for idx in range(count):
      example_id = f'example_{idx:04d}'
      src_dir = os.path.join(split_dir, f'{example_id}_sources')
      os.makedirs(src_dir, exist_ok=True)

      num_sources = 1 + (idx % 4)  # 1 to 4 sources
      mix = np.zeros(num_samples, dtype=np.float32)
      source_rel_paths = []

      for s in range(num_sources):
        freq = freqs[(idx + s) % len(freqs)]
        env = 0.5 * (1.0 + np.sin(2 * np.pi * (0.5 + 0.2 * s) * t))
        sig = (0.15 * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        mix += sig

        src_name = 'background0_sound.wav' if s == 0 else f'foreground{s}_sound.wav'
        src_abs_path = os.path.join(src_dir, src_name)
        wavfile.write(src_abs_path, sr, sig)
        source_rel_paths.append(
            os.path.relpath(src_abs_path, data_dir).replace('\\', '/'))

      mix_abs_path = os.path.join(split_dir, f'{example_id}.wav')
      wavfile.write(mix_abs_path, sr, mix)
      mix_rel_path = os.path.relpath(mix_abs_path, data_dir).replace('\\', '/')

      fields = [mix_rel_path] + source_rel_paths
      lines.append('\t'.join(fields))

    with open(list_path, 'w', encoding='utf-8') as f:
      f.write('\n'.join(lines) + '\n')

  print(f'Dummy FUSS dataset created at: {data_dir}')


def main():
  parser = argparse.ArgumentParser(description='Create dummy FUSS dataset for testing.')
  parser.add_argument('-d', '--data_dir', default='/content/fuss_dummy/ssdata',
                      help='Target ssdata directory.')
  parser.add_argument('--num_train', type=int, default=12)
  parser.add_argument('--num_val', type=int, default=6)
  parser.add_argument('--num_eval', type=int, default=6)
  args = parser.parse_args()
  generate_dummy_fuss(args.data_dir, args.num_train, args.num_val, args.num_eval)


if __name__ == '__main__':
  main()

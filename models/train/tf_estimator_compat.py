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
"""Compatibility layer for tf.compat.v1.estimator on TensorFlow 2.16+ (Colab)."""

import os
import time
import tensorflow.compat.v1 as tf

_NATIVE_ESTIMATOR = None
try:
  from tensorflow.compat.v1 import estimator as _native_est  # pylint: disable=g-import-not-at-top
  _NATIVE_ESTIMATOR = _native_est
except (ImportError, AttributeError):
  try:
    from tensorflow_estimator.python.estimator.api._v1 import estimator as _native_est  # pylint: disable=g-import-not-at-top
    _NATIVE_ESTIMATOR = _native_est
  except (ImportError, AttributeError):
    _NATIVE_ESTIMATOR = None


if _NATIVE_ESTIMATOR is not None:
  ModeKeys = _NATIVE_ESTIMATOR.ModeKeys
  EstimatorSpec = _NATIVE_ESTIMATOR.EstimatorSpec
  TrainSpec = _NATIVE_ESTIMATOR.TrainSpec
  EvalSpec = _NATIVE_ESTIMATOR.EvalSpec
  RunConfig = _NATIVE_ESTIMATOR.RunConfig
  Estimator = _NATIVE_ESTIMATOR.Estimator
  train_and_evaluate = _NATIVE_ESTIMATOR.train_and_evaluate
else:

  class ModeKeys(object):
    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'infer'

  class EstimatorSpec(object):
    """Minimal compatible replacement for tf.estimator.EstimatorSpec."""

    def __init__(self,
                 mode,
                 predictions=None,
                 loss=None,
                 train_op=None,
                 eval_metric_ops=None,
                 training_hooks=None,
                 **kwargs):
      self.mode = mode
      self.predictions = predictions
      self.loss = loss
      self.train_op = train_op
      self.eval_metric_ops = eval_metric_ops or {}
      self.training_hooks = training_hooks or []
      self.extra_kwargs = kwargs

  class RunConfig(object):
    """Minimal compatible replacement for tf.estimator.RunConfig."""

    def __init__(self,
                 model_dir=None,
                 save_summary_steps=100,
                 save_checkpoints_secs=600,
                 save_checkpoints_steps=None,
                 keep_checkpoint_max=5,
                 keep_checkpoint_every_n_hours=10000,
                 **kwargs):
      self.model_dir = model_dir
      self.save_summary_steps = save_summary_steps
      self.save_checkpoints_secs = save_checkpoints_secs
      self.save_checkpoints_steps = save_checkpoints_steps
      self.keep_checkpoint_max = keep_checkpoint_max
      self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
      self.extra_kwargs = kwargs

  class TrainSpec(object):
    """Minimal compatible replacement for tf.estimator.TrainSpec."""

    def __init__(self, input_fn, max_steps=None, hooks=None):
      self.input_fn = input_fn
      self.max_steps = max_steps
      self.hooks = hooks or []

  class EvalSpec(object):
    """Minimal compatible replacement for tf.estimator.EvalSpec."""

    def __init__(self,
                 input_fn,
                 steps=100,
                 name=None,
                 hooks=None,
                 throttle_secs=600):
      self.input_fn = input_fn
      self.steps = steps
      self.name = name
      self.hooks = hooks or []
      self.throttle_secs = throttle_secs

  class Estimator(object):
    """Session-based v1 Estimator fallback for TensorFlow 2.16+."""

    def __init__(self, model_fn, model_dir=None, config=None, params=None):
      self._model_fn = model_fn
      self._config = config or RunConfig(model_dir=model_dir)
      self._model_dir = model_dir or self._config.model_dir
      self._params = params or {}
      if self._model_dir:
        tf.gfile.MakeDirs(self._model_dir)

    @property
    def model_dir(self):
      return self._model_dir

    @property
    def config(self):
      return self._config

    @property
    def params(self):
      return self._params

    def evaluate(self, input_fn, steps=None, checkpoint_path=None, name=None):
      """Runs evaluation for `steps` batches on the latest or given checkpoint."""
      if not checkpoint_path:
        checkpoint_path = tf.train.latest_checkpoint(self._model_dir)
      if not checkpoint_path:
        tf.logging.warning('No checkpoint found in %s for evaluation.',
                           self._model_dir)
        return {}

      eval_dir = os.path.join(
          self._model_dir, 'eval_%s' % name if name else 'eval')
      tf.gfile.MakeDirs(eval_dir)

      with tf.Graph().as_default() as g:
        global_step_tensor = tf.train.get_or_create_global_step()
        features = input_fn()
        labels = None
        if isinstance(features, tuple) and len(features) == 2:
          features, labels = features

        spec = self._model_fn(features, labels, ModeKeys.EVAL, self._params)
        saver = tf.train.Saver()

        update_ops = {}
        value_ops = {}
        for metric_name, metric_tuple in spec.eval_metric_ops.items():
          value_ops[metric_name] = metric_tuple[0]
          update_ops[metric_name] = metric_tuple[1]
        if spec.loss is not None:
          loss_metric = tf.metrics.mean(spec.loss)
          value_ops['loss'] = loss_metric[0]
          update_ops['loss'] = loss_metric[1]

        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True

        with tf.Session(graph=g, config=session_config) as sess:
          sess.run(tf.global_variables_initializer())
          sess.run(tf.local_variables_initializer())
          saver.restore(sess, checkpoint_path)
          step_val = int(sess.run(global_step_tensor))
          tf.logging.info('Starting evaluation at step %d from %s',
                          step_val, checkpoint_path)

          num_steps = steps or 100
          for _ in range(num_steps):
            try:
              sess.run(update_ops)
            except tf.errors.OutOfRangeError:
              break

          results = sess.run(value_ops)
          results['global_step'] = step_val

        summary_writer = tf.summary.FileWriter(eval_dir)
        summary = tf.Summary()
        for k, v in results.items():
          if k == 'global_step':
            continue
          summary.value.add(tag=k, simple_value=float(v))
        summary_writer.add_summary(summary, step_val)
        summary_writer.flush()
        summary_writer.close()

        tf.logging.info('Evaluation results at step %d: %s', step_val, results)
        return results

  def train_and_evaluate(estimator, train_spec, eval_spec):
    """Runs training and periodic evaluation using tf.compat.v1.Session."""
    model_dir = estimator.model_dir
    config = estimator.config
    tf.gfile.MakeDirs(model_dir)

    with tf.Graph().as_default() as g:
      global_step_tensor = tf.train.get_or_create_global_step()
      features = train_spec.input_fn()
      labels = None
      if isinstance(features, tuple) and len(features) == 2:
        features, labels = features

      spec = estimator._model_fn(  # pylint: disable=protected-access
          features, labels, ModeKeys.TRAIN, estimator.params)

      summary_op = tf.summary.merge_all()
      saver = tf.train.Saver(
          max_to_keep=config.keep_checkpoint_max,
          keep_checkpoint_every_n_hours=config.keep_checkpoint_every_n_hours)
      summary_writer = tf.summary.FileWriter(model_dir, g)

      session_config = tf.ConfigProto(allow_soft_placement=True)
      session_config.gpu_options.allow_growth = True

      ckpt_path_prefix = os.path.join(model_dir, 'model.ckpt')

      with tf.Session(graph=g, config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        latest_ckpt = tf.train.latest_checkpoint(model_dir)
        if latest_ckpt:
          tf.logging.info('Restoring checkpoint from %s', latest_ckpt)
          saver.restore(sess, latest_ckpt)
        else:
          step_val = int(sess.run(global_step_tensor))
          saver.save(sess, ckpt_path_prefix, global_step=step_val)
          tf.logging.info('Saved initial checkpoint at step %d', step_val)

        step_val = int(sess.run(global_step_tensor))
        max_steps = train_spec.max_steps
        last_ckpt_time = time.time()
        last_eval_time = 0.0
        last_log_time = time.time()
        last_log_step = step_val

        tf.logging.info('Starting training from step %d to max_steps=%s',
                        step_val, str(max_steps))

        while max_steps is None or step_val < max_steps:
          should_summarize = (
              summary_op is not None and
              config.save_summary_steps and
              (step_val % config.save_summary_steps == 0))

          try:
            if should_summarize:
              _, loss_val, step_val, summary_str = sess.run([
                  spec.train_op, spec.loss, global_step_tensor, summary_op
              ])
              summary_writer.add_summary(summary_str, step_val)
              summary_writer.flush()
            else:
              _, loss_val, step_val = sess.run(
                  [spec.train_op, spec.loss, global_step_tensor])
          except tf.errors.OutOfRangeError:
            tf.logging.info('Input dataset exhausted at step %d.', step_val)
            break

          step_val = int(step_val)
          now = time.time()

          if now - last_log_time >= 10.0 or step_val == 1:
            elapsed = max(now - last_log_time, 1e-6)
            steps_per_sec = (step_val - last_log_step) / elapsed
            tf.logging.info('step = %d, loss = %.4f (%.2f steps/sec)',
                            step_val, float(loss_val), steps_per_sec)
            print(f'[Train] step = {step_val}, loss = {float(loss_val):.4f} '
                  f'({steps_per_sec:.2f} steps/sec)', flush=True)
            last_log_time = now
            last_log_step = step_val

          should_save_ckpt = False
          if (config.save_checkpoints_secs and
              (now - last_ckpt_time >= config.save_checkpoints_secs)):
            should_save_ckpt = True
          elif (config.save_checkpoints_steps and
                (step_val % config.save_checkpoints_steps == 0)):
            should_save_ckpt = True

          if should_save_ckpt:
            saved_path = saver.save(
                sess, ckpt_path_prefix, global_step=step_val)
            last_ckpt_time = time.time()
            tf.logging.info('Saved checkpoint: %s', saved_path)
            print(f'[Checkpoint] Saved: {saved_path}', flush=True)

            if (eval_spec is not None and
                (now - last_eval_time >= eval_spec.throttle_secs)):
              last_eval_time = time.time()
              try:
                estimator.evaluate(
                    input_fn=eval_spec.input_fn,
                    steps=eval_spec.steps,
                    checkpoint_path=saved_path,
                    name=eval_spec.name)
              except Exception as e:  # pylint: disable=broad-except
                tf.logging.warning('Evaluation failed at step %d: %s',
                                   step_val, str(e))

        saved_path = saver.save(sess, ckpt_path_prefix, global_step=step_val)
        summary_writer.flush()
        summary_writer.close()
        tf.logging.info('Training finished at step %d. Final checkpoint: %s',
                        step_val, saved_path)
        print(f'[Done] Training finished at step {step_val}. '
              f'Final checkpoint: {saved_path}', flush=True)

        if eval_spec is not None:
          try:
            estimator.evaluate(
                input_fn=eval_spec.input_fn,
                steps=eval_spec.steps,
                checkpoint_path=saved_path,
                name=eval_spec.name)
          except Exception as e:  # pylint: disable=broad-except
            tf.logging.warning('Final evaluation failed: %s', str(e))


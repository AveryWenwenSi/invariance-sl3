import tensorflow as tf
from tensorflow.python.util import nest
import os

os.environ["CUDA_VISIBLE_DEVICES"]='3'

def slicing_where(condition, full_input, true_branch, false_branch):
  """Split `full_input` between `true_branch` and `false_branch` on `condition`.

  Args:
    condition: A boolean Tensor with shape [B_1, ..., B_N].
    full_input: A Tensor or nested tuple of Tensors of any dtype, each with
      shape [B_1, ..., B_N, ...], to be split between `true_branch` and
      `false_branch` based on `condition`.
    true_branch: A function taking a single argument, that argument having the
      same structure and number of batch dimensions as `full_input`. Receives
      slices of `full_input` corresponding to the True entries of
      `condition`. Returns a Tensor or nested tuple of Tensors, each with batch
      dimensions matching its inputs.
    false_branch: Like `true_branch`, but receives inputs corresponding to the
      false elements of `condition`. Returns a Tensor or nested tuple of Tensors
      (with the same structure as the return value of `true_branch`), but with
      batch dimensions matching its inputs.
  Returns:
    Interleaved outputs from `true_branch` and `false_branch`, each Tensor
    having shape [B_1, ..., B_N, ...].
  """
  full_input_flat = nest.flatten(full_input)
  true_indices = tf.where(condition)
  false_indices = tf.where(tf.logical_not(condition))
  true_branch_inputs = nest.pack_sequence_as(
      structure=full_input,
      flat_sequence=[tf.gather_nd(params=input_tensor, indices=true_indices)
                     for input_tensor in full_input_flat])
  false_branch_inputs = nest.pack_sequence_as(
      structure=full_input,
      flat_sequence=[tf.gather_nd(params=input_tensor, indices=false_indices)
                     for input_tensor in full_input_flat])
  true_outputs = true_branch(true_branch_inputs)
  false_outputs = false_branch(false_branch_inputs)
  nest.assert_same_structure(true_outputs, false_outputs)
  def scatter_outputs(true_output, false_output):
    batch_shape = tf.shape(condition)
    scattered_shape = tf.concat(
        [batch_shape, tf.shape(true_output)[tf.rank(batch_shape):]],
        0)
    true_scatter = tf.scatter_nd(
        indices=tf.cast(true_indices, tf.int32),
        updates=true_output,
        shape=scattered_shape)
    false_scatter = tf.scatter_nd(
        indices=tf.cast(false_indices, tf.int32),
        updates=false_output,
        shape=scattered_shape)
    return true_scatter + false_scatter
  result = nest.pack_sequence_as(
      structure=true_outputs,
      flat_sequence=[
          scatter_outputs(true_single_output, false_single_output)
          for true_single_output, false_single_output
          in zip(nest.flatten(true_outputs), nest.flatten(false_outputs))])
  return result

vector_test = slicing_where(
    condition=tf.equal(tf.range(10) % 2, 0),
    full_input=tf.range(10, dtype=tf.float32),
    true_branch=lambda x: 0.2 + x,
    false_branch=lambda x: 0.1 + x)

cross_range = (tf.range(10, dtype=tf.float32)[:, None]
               * tf.range(10, dtype=tf.float32)[None, :])
matrix_test = slicing_where(
    condition=tf.equal(tf.range(10) % 3, 0),
    full_input=cross_range,
    true_branch=lambda x: -x,
    false_branch=lambda x: x + 0.1)

with tf.Session():
  print(vector_test.eval())
  print(matrix_test.eval())
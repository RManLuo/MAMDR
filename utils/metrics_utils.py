# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=protected-access
"""Utils related to keras metrics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import weakref
from enum import Enum

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.util import tf_decorator

NEG_INF = -1e10


class Reduction(Enum):
    """Types of metrics reduction.

    Contains the following values:

    * `SUM`: Scalar sum of weighted values.
    * `SUM_OVER_BATCH_SIZE`: Scalar sum of weighted values divided by
          number of elements.
    * `WEIGHTED_MEAN`: Scalar sum of weighted values divided by sum of weights.
    """
    SUM = 'sum'
    SUM_OVER_BATCH_SIZE = 'sum_over_batch_size'
    WEIGHTED_MEAN = 'weighted_mean'


def update_state_wrapper(update_state_fn):
    """Decorator to wrap metric `update_state()` with `add_update()`.

    Args:
      update_state_fn: function that accumulates metric statistics.

    Returns:
      Decorated function that wraps `update_state_fn()` with `add_update()`.
    """

    def decorated(metric_obj, *args, **kwargs):
        """Decorated function with `add_update()`."""

        with tf_utils.graph_context_for_symbolic_tensors(*args, **kwargs):
            update_op = update_state_fn(*args, **kwargs)
        if update_op is not None:  # update_op will be None in eager execution.
            metric_obj.add_update(update_op)
        return update_op

    return tf_decorator.make_decorator(update_state_fn, decorated)


def result_wrapper(result_fn):
    """Decorator to wrap metric `result()` function in `merge_call()`.

    Result computation is an idempotent operation that simply calculates the
    metric value using the state variables.

    If metric state variables are distributed across replicas/devices and
    `result()` is requested from the context of one device - This function wraps
    `result()` in a distribution strategy `merge_call()`. With this,
    the metric state variables will be aggregated across devices.

    Args:
      result_fn: function that computes the metric result.

    Returns:
      Decorated function that wraps `result_fn()` in distribution strategy
      `merge_call()`.
    """

    def decorated(metric_obj, *args):
        """Decorated function with merge_call."""
        result_t = array_ops.identity(result_fn(*args))

        # We are saving the result op here to be used in train/test execution
        # functions. This basically gives the result op that was generated with a
        # control dep to the updates for these workflows.
        metric_obj._call_result = result_t
        return result_t

    return tf_decorator.make_decorator(result_fn, decorated)


def weakmethod(method):
    """Creates a weak reference to the bound method."""

    cls = method.im_class
    func = method.im_func
    instance_ref = weakref.ref(method.im_self)

    @functools.wraps(method)
    def inner(*args, **kwargs):
        return func.__get__(instance_ref(), cls)(*args, **kwargs)

    del method
    return inner


def assert_thresholds_range(thresholds):
    if thresholds is not None:
        invalid_thresholds = [t for t in thresholds if t is None or t < 0 or t > 1]
        if invalid_thresholds:
            raise ValueError(
                'Threshold values must be in [0, 1]. Invalid values: {}'.format(
                    invalid_thresholds))


def parse_init_thresholds(thresholds, default_threshold=0.5):
    if thresholds is not None:
        assert_thresholds_range(to_list(thresholds))
    thresholds = to_list(default_threshold if thresholds is None else thresholds)
    return thresholds


class ConfusionMatrix(Enum):
    TRUE_POSITIVES = 'tp'
    FALSE_POSITIVES = 'fp'
    TRUE_NEGATIVES = 'tn'
    FALSE_NEGATIVES = 'fn'


class AUCCurve(Enum):
    """Type of AUC Curve (ROC or PR)."""
    ROC = 'ROC'
    PR = 'PR'

    @staticmethod
    def from_str(key):
        if key in ('pr', 'PR'):
            return AUCCurve.PR
        elif key in ('roc', 'ROC'):
            return AUCCurve.ROC
        else:
            raise ValueError('Invalid AUC curve value "%s".' % key)


class AUCSummationMethod(Enum):
    """Type of AUC summation method.

    https://en.wikipedia.org/wiki/Riemann_sum)

    Contains the following values:
    * 'interpolation': Applies mid-point summation scheme for `ROC` curve. For
      `PR` curve, interpolates (true/false) positives but not the ratio that is
      precision (see Davis & Goadrich 2006 for details).
    * 'minoring': Applies left summation for increasing intervals and right
      summation for decreasing intervals.
    * 'majoring': Applies right summation for increasing intervals and left
      summation for decreasing intervals.
    """
    INTERPOLATION = 'interpolation'
    MAJORING = 'majoring'
    MINORING = 'minoring'

    @staticmethod
    def from_str(key):
        if key in ('interpolation', 'Interpolation'):
            return AUCSummationMethod.INTERPOLATION
        elif key in ('majoring', 'Majoring'):
            return AUCSummationMethod.MAJORING
        elif key in ('minoring', 'Minoring'):
            return AUCSummationMethod.MINORING
        else:
            raise ValueError('Invalid AUC summation method value "%s".' % key)


def update_confusion_matrix_variables(variables_to_update,
                                      y_true,
                                      y_pred,
                                      thresholds,
                                      top_k=None,
                                      class_id=None,
                                      sample_weight=None):
    """Returns op to update the given confusion matrix variables.

    For every pair of values in y_true and y_pred:

    true_positive: y_true == True and y_pred > thresholds
    false_negatives: y_true == True and y_pred <= thresholds
    true_negatives: y_true == False and y_pred <= thresholds
    false_positive: y_true == False and y_pred > thresholds

    The results will be weighted and added together. When multiple thresholds are
    provided, we will repeat the same for every threshold.

    For estimation of these metrics over a stream of data, the function creates an
    `update_op` operation that updates the given variables.

    If `sample_weight` is `None`, weights default to 1.
    Use weights of 0 to mask values.

    Args:
      variables_to_update: Dictionary with 'tp', 'fn', 'tn', 'fp' as valid keys
        and corresponding variables to update as values.
      y_true: A `Tensor` whose shape matches `y_pred`. Will be cast to `bool`.
      y_pred: A floating point `Tensor` of arbitrary shape and whose values are in
        the range `[0, 1]`.
      thresholds: A float value or a python list or tuple of float thresholds in
        `[0, 1]`, or NEG_INF (used when top_k is set).
      top_k: Optional int, indicates that the positive labels should be limited to
        the top k predictions.
      class_id: Optional int, limits the prediction and labels to the class
        specified by this argument.
      sample_weight: Optional `Tensor` whose rank is either 0, or the same rank as
        `y_true`, and must be broadcastable to `y_true` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `y_true` dimension).

    Returns:
      Update op.

    Raises:
      ValueError: If `y_pred` and `y_true` have mismatched shapes, or if
        `sample_weight` is not `None` and its shape doesn't match `y_pred`, or if
        `variables_to_update` contains invalid keys.
    """
    if variables_to_update is None:
        return
    y_true = math_ops.cast(y_true, dtype=dtypes.float32)
    y_pred = math_ops.cast(y_pred, dtype=dtypes.float32)
    [y_pred,
     y_true], _ = ragged_assert_compatible_and_get_flat_values([y_pred, y_true],
                                                               sample_weight)
    y_pred.shape.assert_is_compatible_with(y_true.shape)

    if not any(
            key for key in variables_to_update if key in list(ConfusionMatrix)):
        raise ValueError(
            'Please provide at least one valid confusion matrix '
            'variable to update. Valid variable key options are: "{}". '
            'Received: "{}"'.format(
                list(ConfusionMatrix), variables_to_update.keys()))

    invalid_keys = [
        key for key in variables_to_update if key not in list(ConfusionMatrix)
    ]
    if invalid_keys:
        raise ValueError(
            'Invalid keys: {}. Valid variable key options are: "{}"'.format(
                invalid_keys, list(ConfusionMatrix)))

    with ops.control_dependencies([
        check_ops.assert_greater_equal(
            y_pred,
            math_ops.cast(0.0, dtype=y_pred.dtype),
            message='predictions must be >= 0'),
        check_ops.assert_less_equal(
            y_pred,
            math_ops.cast(1.0, dtype=y_pred.dtype),
            message='predictions must be <= 1')
    ]):
        if sample_weight is None:
            y_pred, y_true = squeeze_or_expand_dimensions(
                y_pred, y_true)
        else:
            y_pred, y_true, sample_weight = (
                squeeze_or_expand_dimensions(
                    y_pred, y_true, sample_weight=sample_weight))

    if top_k is not None:
        y_pred = _filter_top_k(y_pred, top_k)
    if class_id is not None:
        y_true = y_true[..., class_id]
        y_pred = y_pred[..., class_id]

    thresholds = to_list(thresholds)
    num_thresholds = len(thresholds)
    num_predictions = array_ops.size(y_pred)

    # Reshape predictions and labels.
    predictions_2d = array_ops.reshape(y_pred, [1, -1])
    labels_2d = array_ops.reshape(
        math_ops.cast(y_true, dtype=dtypes.bool), [1, -1])

    # Tile the thresholds for every prediction.
    thresh_tiled = array_ops.tile(
        array_ops.expand_dims(array_ops.constant(thresholds), 1),
        array_ops.stack([1, num_predictions]))

    # Tile the predictions for every threshold.
    preds_tiled = array_ops.tile(predictions_2d, [num_thresholds, 1])

    # Compare predictions and threshold.
    pred_is_pos = math_ops.greater(preds_tiled, thresh_tiled)

    # Tile labels by number of thresholds
    label_is_pos = array_ops.tile(labels_2d, [num_thresholds, 1])

    if sample_weight is not None:
        weights = weights_broadcast_ops.broadcast_weights(
            math_ops.cast(sample_weight, dtype=dtypes.float32), y_pred)
        weights_tiled = array_ops.tile(
            array_ops.reshape(weights, [1, -1]), [num_thresholds, 1])
    else:
        weights_tiled = None

    update_ops = []

    def weighted_assign_add(label, pred, weights, var):
        label_and_pred = math_ops.cast(
            math_ops.logical_and(label, pred), dtype=dtypes.float32)
        if weights is not None:
            label_and_pred *= weights
        return var.assign_add(math_ops.reduce_sum(label_and_pred, 1))

    loop_vars = {
        ConfusionMatrix.TRUE_POSITIVES: (label_is_pos, pred_is_pos),
    }
    update_tn = ConfusionMatrix.TRUE_NEGATIVES in variables_to_update
    update_fp = ConfusionMatrix.FALSE_POSITIVES in variables_to_update
    update_fn = ConfusionMatrix.FALSE_NEGATIVES in variables_to_update

    if update_fn or update_tn:
        pred_is_neg = math_ops.logical_not(pred_is_pos)
        loop_vars[ConfusionMatrix.FALSE_NEGATIVES] = (label_is_pos, pred_is_neg)

    if update_fp or update_tn:
        label_is_neg = math_ops.logical_not(label_is_pos)
        loop_vars[ConfusionMatrix.FALSE_POSITIVES] = (label_is_neg, pred_is_pos)
        if update_tn:
            loop_vars[ConfusionMatrix.TRUE_NEGATIVES] = (label_is_neg, pred_is_neg)

    for matrix_cond, (label, pred) in loop_vars.items():
        if matrix_cond in variables_to_update:
            update_ops.append(
                weighted_assign_add(label, pred, weights_tiled,
                                    variables_to_update[matrix_cond]))
    return control_flow_ops.group(update_ops)


def _filter_top_k(x, k):
    """Filters top-k values in the last dim of x and set the rest to NEG_INF.

    Used for computing top-k prediction values in dense labels (which has the same
    shape as predictions) for recall and precision top-k metrics.

    Args:
      x: tensor with any dimensions.
      k: the number of values to keep.

    Returns:
      tensor with same shape and dtype as x.
    """
    _, top_k_idx = nn_ops.top_k(x, k, sorted=False)
    top_k_mask = math_ops.reduce_sum(
        array_ops.one_hot(top_k_idx, x.shape[-1], axis=-1), axis=-2)
    return x * top_k_mask + NEG_INF * (1 - top_k_mask)


def ragged_assert_compatible_and_get_flat_values(values, mask=None):
    """If ragged, it checks the compatibility and then returns the flat_values.

       Note: If two tensors are dense, it does not check their compatibility.
       Note: Although two ragged tensors with different ragged ranks could have
             identical overall rank and dimension sizes and hence be compatible,
             we do not support those cases.
    Args:
       values: A list of potentially ragged tensor of the same ragged_rank.
       mask: A potentially ragged tensor of the same ragged_rank as elements in
         Values.

    Returns:
       A tuple in which the first element is the list of tensors and the second
       is the mask tensor. ([Values], mask). Mask and the element in Values
       are equal to the flat_values of the input arguments (if they were ragged).
    """
    return values, mask


def squeeze_or_expand_dimensions(y_pred, y_true=None, sample_weight=None):
    """Squeeze or expand last dimension if needed.
    1. Squeezes last dim of `y_pred` or `y_true` if their rank differs by 1
    (using `confusion_matrix.remove_squeezable_dimensions`).
    2. Squeezes or expands last dim of `sample_weight` if its rank differs by 1
    from the new rank of `y_pred`.
    If `sample_weight` is scalar, it is kept scalar.
    This will use static shape if available. Otherwise, it will add graph
    operations, which could result in a performance hit.
    Args:
      y_pred: Predicted values, a `Tensor` of arbitrary dimensions.
      y_true: Optional label `Tensor` whose dimensions match `y_pred`.
      sample_weight: Optional weight scalar or `Tensor` whose dimensions match
        `y_pred`.
    Returns:
      Tuple of `y_pred`, `y_true` and `sample_weight`. Each of them possibly has
      the last dimension squeezed,
      `sample_weight` could be extended by one dimension.
      If `sample_weight` is None, (y_pred, y_true) is returned.
    """
    y_pred_shape = y_pred.shape
    y_pred_rank = y_pred_shape.ndims
    if y_true is not None:

        # If sparse matrix is provided as `y_true`, the last dimension in `y_pred`
        # may be > 1. Eg: y_true = [0, 1, 2] (shape=(3,)),
        # y_pred = [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]] (shape=(3, 3))
        # In this case, we should not try to remove squeezable dimension.
        y_true_shape = y_true.shape
        y_true_rank = y_true_shape.ndims
        if (y_true_rank is not None) and (y_pred_rank is not None):
            # Use static rank for `y_true` and `y_pred`.
            if (y_pred_rank - y_true_rank != 1) or y_pred_shape[-1] == 1:
                y_true, y_pred = confusion_matrix.remove_squeezable_dimensions(
                    y_true, y_pred)
        else:
            # Use dynamic rank.
            rank_diff = array_ops.rank(y_pred) - array_ops.rank(y_true)
            squeeze_dims = lambda: confusion_matrix.remove_squeezable_dimensions(  # pylint: disable=g-long-lambda
                y_true, y_pred)
            is_last_dim_1 = math_ops.equal(1, array_ops.shape(y_pred)[-1])
            maybe_squeeze_dims = lambda: control_flow_ops.cond(  # pylint: disable=g-long-lambda
                is_last_dim_1, squeeze_dims, lambda: (y_true, y_pred))
            y_true, y_pred = control_flow_ops.cond(
                math_ops.equal(1, rank_diff), maybe_squeeze_dims, squeeze_dims)

    if sample_weight is None:
        return y_pred, y_true

    sample_weight = ops.convert_to_tensor(sample_weight)
    weights_shape = sample_weight.shape
    weights_rank = weights_shape.ndims
    if weights_rank == 0:  # If weights is scalar, do nothing.
        return y_pred, y_true, sample_weight

    if (y_pred_rank is not None) and (weights_rank is not None):
        # Use static rank.
        if weights_rank - y_pred_rank == 1:
            sample_weight = array_ops.squeeze(sample_weight, [-1])
        elif y_pred_rank - weights_rank == 1:
            sample_weight = array_ops.expand_dims(sample_weight, [-1])
        return y_pred, y_true, sample_weight

    # Use dynamic rank.
    weights_rank_tensor = array_ops.rank(sample_weight)
    rank_diff = weights_rank_tensor - array_ops.rank(y_pred)
    maybe_squeeze_weights = lambda: array_ops.squeeze(sample_weight, [-1])

    def _maybe_expand_weights():
        expand_weights = lambda: array_ops.expand_dims(sample_weight, [-1])
        return control_flow_ops.cond(
            math_ops.equal(rank_diff, -1), expand_weights, lambda: sample_weight)

    def _maybe_adjust_weights():
        return control_flow_ops.cond(
            math_ops.equal(rank_diff, 1), maybe_squeeze_weights,
            _maybe_expand_weights)

    # squeeze or expand last dim of `sample_weight` if its rank differs by 1
    # from the new rank of `y_pred`.
    sample_weight = control_flow_ops.cond(
        math_ops.equal(weights_rank_tensor, 0), lambda: sample_weight,
        _maybe_adjust_weights)
    return y_pred, y_true, sample_weight

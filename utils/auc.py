from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops

from . import metrics_utils

# This class is a copy from tf.keras.AUC when tf version >=1.12.
# If you tf version >= 1.12, you can just use tf.keras.AUC
class AUC(Metric):
    """Computes the approximate AUC (Area under the curve) via a Riemann sum.

    This metric creates four local variables, `true_positives`, `true_negatives`,
    `false_positives` and `false_negatives` that are used to compute the AUC.
    To discretize the AUC curve, a linearly spaced set of thresholds is used to
    compute pairs of recall and precision values. The area under the ROC-curve is
    therefore computed using the height of the recall values by the false positive
    rate, while the area under the PR-curve is the computed using the height of
    the precision values by the recall.

    This value is ultimately returned as `auc`, an idempotent operation that
    computes the area under a discretized curve of precision versus recall values
    (computed using the aforementioned variables). The `num_thresholds` variable
    controls the degree of discretization with larger numbers of thresholds more
    closely approximating the true AUC. The quality of the approximation may vary
    dramatically depending on `num_thresholds`. The `thresholds` parameter can be
    used to manually specify thresholds which split the predictions more evenly.

    For best results, `predictions` should be distributed approximately uniformly
    in the range [0, 1] and not peaked around 0 or 1. The quality of the AUC
    approximation may be poor if this is not the case. Setting `summation_method`
    to 'minoring' or 'majoring' can help quantify the error in the approximation
    by providing lower or upper bound estimate of the AUC.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage:

    ```python
    m = tf.keras.metrics.AUC(num_thresholds=3)
    m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])

    # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
    # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
    # recall = [1, 0.5, 0], fp_rate = [1, 0, 0]
    # auc = ((((1+0.5)/2)*(1-0))+ (((0.5+0)/2)*(0-0))) = 0.75

    print('Final result: ', m.result().numpy())  # Final result: 0.75
    ```

    Usage with tf.keras API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.AUC()])
    ```
    """

    def __init__(self,
                 num_thresholds=200,
                 curve='ROC',
                 summation_method='interpolation',
                 name=None,
                 dtype=None,
                 thresholds=None):
        """Creates an `AUC` instance.

        Args:
          num_thresholds: (Optional) Defaults to 200. The number of thresholds to
            use when discretizing the roc curve. Values must be > 1.
          curve: (Optional) Specifies the name of the curve to be computed, 'ROC'
            [default] or 'PR' for the Precision-Recall-curve.
          summation_method: (Optional) Specifies the Riemann summation method used
            (https://en.wikipedia.org/wiki/Riemann_sum): 'interpolation' [default],
              applies mid-point summation scheme for `ROC`. For PR-AUC, interpolates
              (true/false) positives but not the ratio that is precision (see Davis
              & Goadrich 2006 for details); 'minoring' that applies left summation
              for increasing intervals and right summation for decreasing intervals;
              'majoring' that does the opposite.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
          thresholds: (Optional) A list of floating point values to use as the
            thresholds for discretizing the curve. If set, the `num_thresholds`
            parameter is ignored. Values should be in [0, 1]. Endpoint thresholds
            equal to {-epsilon, 1+epsilon} for a small positive epsilon value will
            be automatically included with these to correctly handle predictions
            equal to exactly 0 or 1.
        """
        # Validate configurations.
        if isinstance(curve, metrics_utils.AUCCurve) and curve not in list(
                metrics_utils.AUCCurve):
            raise ValueError('Invalid curve: "{}". Valid options are: "{}"'.format(
                curve, list(metrics_utils.AUCCurve)))
        if isinstance(
                summation_method,
                metrics_utils.AUCSummationMethod) and summation_method not in list(
            metrics_utils.AUCSummationMethod):
            raise ValueError(
                'Invalid summation method: "{}". Valid options are: "{}"'.format(
                    summation_method, list(metrics_utils.AUCSummationMethod)))

        # Update properties.
        if thresholds is not None:
            # If specified, use the supplied thresholds.
            self.num_thresholds = len(thresholds) + 2
            thresholds = sorted(thresholds)
        else:
            if num_thresholds <= 1:
                raise ValueError('`num_thresholds` must be > 1.')

            # Otherwise, linearly interpolate (num_thresholds - 2) thresholds in
            # (0, 1).
            self.num_thresholds = num_thresholds
            thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                          for i in range(num_thresholds - 2)]

        # Add an endpoint "threshold" below zero and above one for either
        # threshold method to account for floating point imprecisions.
        self.thresholds = [0.0 - K.epsilon()] + thresholds + [1.0 + K.epsilon()]

        if isinstance(curve, metrics_utils.AUCCurve):
            self.curve = curve
        else:
            self.curve = metrics_utils.AUCCurve.from_str(curve)
        if isinstance(summation_method, metrics_utils.AUCSummationMethod):
            self.summation_method = summation_method
        else:
            self.summation_method = metrics_utils.AUCSummationMethod.from_str(
                summation_method)
        super(AUC, self).__init__(name=name, dtype=dtype)

        # Create metric variables
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(self.num_thresholds,),
            initializer=init_ops.zeros_initializer)
        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=(self.num_thresholds,),
            initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(self.num_thresholds,),
            initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(self.num_thresholds,),
            initializer=init_ops.zeros_initializer)

        self.reset_op = [tf.assign(v, tf.zeros_like(v)) for v in self.variables]

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        return metrics_utils.update_confusion_matrix_variables({
            metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
            metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
            metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
            metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
        }, y_true, y_pred, self.thresholds, sample_weight=sample_weight)

    def interpolate_pr_auc(self):
        """Interpolation formula inspired by section 4 of Davis & Goadrich 2006.

        https://www.biostat.wisc.edu/~page/rocpr.pdf

        Note here we derive & use a closed formula not present in the paper
        as follows:

          Precision = TP / (TP + FP) = TP / P

        Modeling all of TP (true positive), FP (false positive) and their sum
        P = TP + FP (predicted positive) as varying linearly within each interval
        [A, B] between successive thresholds, we get

          Precision slope = dTP / dP
                          = (TP_B - TP_A) / (P_B - P_A)
                          = (TP - TP_A) / (P - P_A)
          Precision = (TP_A + slope * (P - P_A)) / P

        The area within the interval is (slope / total_pos_weight) times

          int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}
          int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}

        where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in

          int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)

        Bringing back the factor (slope / total_pos_weight) we'd put aside, we get

          slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight

        where dTP == TP_B - TP_A.

        Note that when P_A == 0 the above calculation simplifies into

          int_A^B{Precision.dTP} = int_A^B{slope * dTP} = slope * (TP_B - TP_A)

        which is really equivalent to imputing constant precision throughout the
        first bucket having >0 true positives.

        Returns:
          pr_auc: an approximation of the area under the P-R curve.
        """
        dtp = self.true_positives[:self.num_thresholds -
                                   1] - self.true_positives[1:]
        p = self.true_positives + self.false_positives
        dp = p[:self.num_thresholds - 1] - p[1:]

        prec_slope = math_ops.div_no_nan(
            dtp, math_ops.maximum(dp, 0), name='prec_slope')
        intercept = self.true_positives[1:] - math_ops.multiply(prec_slope, p[1:])

        safe_p_ratio = array_ops.where(
            math_ops.logical_and(p[:self.num_thresholds - 1] > 0, p[1:] > 0),
            math_ops.div_no_nan(
                p[:self.num_thresholds - 1],
                math_ops.maximum(p[1:], 0),
                name='recall_relative_ratio'),
            array_ops.ones_like(p[1:]))

        return math_ops.reduce_sum(
            math_ops.div_no_nan(
                prec_slope * (dtp + intercept * math_ops.log(safe_p_ratio)),
                math_ops.maximum(self.true_positives[1:] + self.false_negatives[1:],
                                 0),
                name='pr_auc_increment'),
            name='interpolate_pr_auc')

    def result(self):
        if (self.curve == metrics_utils.AUCCurve.PR and
                self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION
        ):
            # This use case is different and is handled separately.
            return self.interpolate_pr_auc()

        # Set `x` and `y` values for the curves based on `curve` config.
        recall = math_ops.div_no_nan(self.true_positives,
                                     self.true_positives + self.false_negatives)
        if self.curve == metrics_utils.AUCCurve.ROC:
            fp_rate = math_ops.div_no_nan(self.false_positives,
                                          self.false_positives + self.true_negatives)
            x = fp_rate
            y = recall
        else:  # curve == 'PR'.
            precision = math_ops.div_no_nan(
                self.true_positives, self.true_positives + self.false_positives)
            x = recall
            y = precision

        # Find the rectangle heights based on `summation_method`.
        if self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION:
            # Note: the case ('PR', 'interpolation') has been handled above.
            heights = (y[:self.num_thresholds - 1] + y[1:]) / 2.
        elif self.summation_method == metrics_utils.AUCSummationMethod.MINORING:
            heights = math_ops.minimum(y[:self.num_thresholds - 1], y[1:])
        else:  # self.summation_method = metrics_utils.AUCSummationMethod.MAJORING:
            heights = math_ops.maximum(y[:self.num_thresholds - 1], y[1:])

        # Sum up the areas of all the rectangles.
        return math_ops.reduce_sum(
            math_ops.multiply(x[:self.num_thresholds - 1] - x[1:], heights),
            name=self.name)

    def reset_states(self):
        K.get_session().run(self.reset_op)

    def get_config(self):
        config = {
            'num_thresholds': self.num_thresholds,
            'curve': self.curve.value,
            'summation_method': self.summation_method.value,
            # We remove the endpoint thresholds as an inverse of how the thresholds
            # were initialized. This ensures that a metric initialized from this
            # config has the same thresholds.
            'thresholds': self.thresholds[1:-1],
        }
        base_config = super(AUC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

"""Bagging meta-estimator for PU learning."""

from __future__ import division
import itertools
import numbers
import numpy as np
from abc import ABCMeta, abstractmethod

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.base import ClassifierMixin, RegressorMixin
from joblib import Parallel, delayed
from six import with_metaclass
from six.moves import zip
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state, check_X_y, check_array, column_or_1d
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from sklearn.utils import indices_to_mask, check_consistent_length
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.multiclass import check_classification_targets
from sklearn.ensemble.base import BaseEnsemble, _partition_estimators

__all__ = ["BaggingClassifierPU"]
MAX_INT = np.iinfo(np.int32).max

flag = 0

def _generate_indices(random_state, bootstrap, n_population, n_samples):
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(n_population, n_samples,
                                             random_state=random_state)
    return indices


def _generate_bagging_indices(random_state, bootstrap_features,
                              bootstrap_samples, n_features, n_samples,
                              max_features, max_samples):
    random_state = check_random_state(random_state)

    feature_indices = _generate_indices(random_state, bootstrap_features,
                                        n_features, max_features)
    sample_indices = _generate_indices(random_state, bootstrap_samples,
                                       n_samples, max_samples)

    return feature_indices, sample_indices


def _parallel_build_estimators(n_estimators, ensemble, X, y, sample_weight,
                               seeds, total_n_estimators, verbose):

    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_,
                                              "sample_weight")
    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    estimators = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        random_state = np.random.RandomState(seeds[i])
        estimator = ensemble._make_estimator(append=False,
                                             random_state=random_state)

        iP = [pair[0] for pair in enumerate(y) if pair[1] == 1]
        iU = [pair[0] for pair in enumerate(y) if pair[1] < 1]
        features, indices = _generate_bagging_indices(random_state,
                                                      bootstrap_features,
                                                      bootstrap, n_features,
                                                      len(iU), max_features,
                                                      max_samples)
        indices = [iU[i] for i in indices] + iP

        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0

            estimator.fit(X[:, features], y, sample_weight=curr_sample_weight)

        else:
            estimator.fit((X[indices])[:, features], y[indices])

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features


def _parallel_predict_proba(estimators, estimators_features, X, n_classes):
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_classes))
    c = 1.0
    for estimator, features in zip(estimators, estimators_features):
        if hasattr(estimator, "predict_proba"):
            proba_estimator = estimator.predict_proba(X[:, features])
            c = np.mean(proba_estimator[:, 1])
            if n_classes == len(estimator.classes_):
                proba += proba_estimator
            else:
                proba[:, estimator.classes_] += proba_estimator[:, range(len(estimator.classes_))]
        else:
            predictions = estimator.predict(X[:, features])

            for i in range(n_samples):
                proba[i, predictions[i]] += 1

    return proba / c


def _parallel_predict_log_proba(estimators, estimators_features, X, n_classes):
    n_samples = X.shape[0]
    log_proba = np.empty((n_samples, n_classes))
    log_proba.fill(-np.inf)
    all_classes = np.arange(n_classes, dtype=np.int)

    for estimator, features in zip(estimators, estimators_features):
        log_proba_estimator = estimator.predict_log_proba(X[:, features])
        c = np.mean(log_proba_estimator[:, 1])
        if n_classes == len(estimator.classes_):
            log_proba = np.logaddexp(log_proba, log_proba_estimator)

        else:
            log_proba[:, estimator.classes_] = np.logaddexp(
                log_proba[:, estimator.classes_],
                log_proba_estimator[:, range(len(estimator.classes_))])

            missing = np.setdiff1d(all_classes, estimator.classes_)
            log_proba[:, missing] = np.logaddexp(log_proba[:, missing],
                                                 -np.inf)

    return log_proba / c


def _parallel_decision_function(estimators, estimators_features, X):
    return sum(estimator.decision_function(X[:, features])
               for estimator, features in zip(estimators,
                                              estimators_features))


class BaseBaggingPU(with_metaclass(ABCMeta, BaseEnsemble)):
    """Base class for Bagging PU meta-estimator."""

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=True,
                 warm_start=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(BaseBaggingPU, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)

        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        return self._fit(X, y, self.max_samples, sample_weight=sample_weight)

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        random_state = check_random_state(self.random_state)

        self.y = y

        X, y = check_X_y(X, y, ['csr', 'csc'])
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        n_samples, self.n_features_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)

        self._validate_estimator()

        if max_depth is not None:
            self.base_estimator_.max_depth = max_depth

        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, (numbers.Integral, np.integer)):
            max_samples = int(max_samples * sum(y < 1))

        if not (0 < max_samples <= sum(y < 1)):
            raise ValueError("max_samples must be positive"
                             " and no larger than the number of unlabeled points")

        self._max_samples = max_samples

        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        self._max_features = max_features

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available"
                             " if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, 'estimators_'):
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
            return self

        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators,
                                                             self.n_jobs)
        total_n_estimators = sum(n_estimators)

        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i]:starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose)
            for i in range(n_jobs))

        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self

    @abstractmethod
    def _set_oob_score(self, X, y):
        """Calculate out of bag predictions and score."""

    def _validate_y(self, y):
        return column_or_1d(y, warn=True)

    def _get_estimators_indices(self):
        for seed in self._seeds:
            random_state = np.random.RandomState(seed)

            iP = [pair[0] for pair in enumerate(self.y) if pair[1] == 1]
            iU = [pair[0] for pair in enumerate(self.y) if pair[1] < 1]

            feature_indices, sample_indices = _generate_bagging_indices(
                random_state, self.bootstrap_features, self.bootstrap,
                self.n_features_, len(iU), self._max_features,
                self._max_samples)

            sample_indices = [iU[i] for i in sample_indices] + iP
            '''
            output = open('sample.xls', 'a+', encoding='gbk')
            for i in range(len(sample_indices)):
                    output.write(str(sample_indices[i]))  
                    output.write('\t') 
            output.write('\n')
            output.close()
            '''
            yield feature_indices, sample_indices

    @property
    def estimators_samples_(self):
        sample_masks = []
        for _, sample_indices in self._get_estimators_indices():
            mask = indices_to_mask(sample_indices, self._n_samples)
            sample_masks.append(mask)

        return sample_masks


class BaggingClassifierPU(BaseBaggingPU, ClassifierMixin):
    """A Bagging PU classifier."""
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=True,
                 warm_start=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):

        super(BaggingClassifierPU, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def _validate_estimator(self):
        super(BaggingClassifierPU, self)._validate_estimator(
            default=DecisionTreeClassifier())

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]
        n_classes_ = self.n_classes_
        classes_ = self.classes_
        
        predictions = np.zeros((n_samples, n_classes_))

        for estimator, samples, features in zip(self.estimators_,
                                                self.estimators_samples_,
                                                self.estimators_features_):

            mask = ~samples
            if hasattr(estimator, "predict_proba"):
                
                predictions[mask, :] += estimator.predict_proba(
                    (X[mask, :])[:, features])

            else:           
                p = estimator.predict((X[mask, :])[:, features])
                j = 0

                for i in range(n_samples):
                    if mask[i]:
                        predictions[i, p[j]] += 1
                        j += 1

        with np.errstate(invalid='ignore'):
            oob_decision_function = (predictions /
                                     predictions.sum(axis=1)[:, np.newaxis])
            oob_score = accuracy_score(y, np.argmax(predictions, axis=1))

        self.oob_decision_function_ = oob_decision_function
        self.oob_score_ = oob_score

    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        return y

    def predict(self, X):
        """Predict class for X."""
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse=['csr', 'csc'])

        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X,
                self.n_classes_)
            for i in range(n_jobs))

        proba = sum(all_proba) / self.n_estimators
        return proba

    def predict_log_proba(self, X):
        check_is_fitted(self, "classes_")
        if hasattr(self.base_estimator_, "predict_log_proba"):
            X = check_array(X, accept_sparse=['csr', 'csc'])

            if self.n_features_ != X.shape[1]:
                raise ValueError("Number of features of the model must "
                                 "match the input. Model n_features is {0} "
                                 "and input n_features is {1} "
                                 "".format(self.n_features_, X.shape[1]))

            n_jobs, n_estimators, starts = _partition_estimators(
                self.n_estimators, self.n_jobs)

            all_log_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                delayed(_parallel_predict_log_proba)(
                    self.estimators_[starts[i]:starts[i + 1]],
                    self.estimators_features_[starts[i]:starts[i + 1]],
                    X,
                    self.n_classes_)
                for i in range(n_jobs))

            log_proba = all_log_proba[0]

            for j in range(1, len(all_log_proba)):
                log_proba = np.logaddexp(log_proba, all_log_proba[j])

            log_proba -= np.log(self.n_estimators)

            return log_proba

        else:
            return np.log(self.predict_proba(X))

    @if_delegate_has_method(delegate='base_estimator')
    def decision_function(self, X):
        """Average of the decision functions of the base classifiers."""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse=['csr', 'csc'])

        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1} "
                             "".format(self.n_features_, X.shape[1]))

        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_decisions = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_decision_function)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X)
            for i in range(n_jobs))

        decisions = sum(all_decisions) / self.n_estimators

        return decisions
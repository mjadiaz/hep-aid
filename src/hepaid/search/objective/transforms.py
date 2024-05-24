'''
Adapted from scikit-optimize https://github.com/scikit-optimize
'''

from __future__ import division
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import column_or_1d


class Transformer(object):
    """Base class for all 1-D transformers.
    """
    def fit(self, X):
        return self

    def transform(self, X):
        raise NotImplementedError

    def inverse_transform(self, X):
        raise NotImplementedError


class Identity(Transformer):
    """Identity transform.
    """

    def transform(self, X):
        return X

    def inverse_transform(self, Xt):
        return Xt
    

class LogN(Transformer):
    """Base N logarithm transform."""

    def __init__(self, base):
        self._base = base

    def transform(self, X):
        return np.log10(np.asarray(X, dtype=np.float)) / np.log10(self._base)

    def inverse_transform(self, Xt):
        return self._base ** np.asarray(Xt, dtype=np.float)

class Normalize(Transformer):
    """
    Scales each dimension into the interval [0, 1].

    Parameters
    ----------
    low : float
        Lower bound.

    high : float
        Higher bound.

    is_int : bool, default=True
        Round and cast the return value of `inverse_transform` to integer. Set
        to `True` when applying this transform to integers.
    """
    def __init__(self, low, high, is_int=False):
        self.low = float(low)
        self.high = float(high)
        self.is_int = is_int
        self._eps = 1e-8

    def transform(self, X):
        X = np.asarray(X)
        if self.is_int:
            if np.any(np.round(X) > self.high):
                raise ValueError("All integer values should"
                                 "be less than %f" % self.high)
            if np.any(np.round(X) < self.low):
                raise ValueError("All integer values should"
                                 "be greater than %f" % self.low)
        else:
            if np.any(X > self.high + self._eps):
                raise ValueError("All values should"
                                 "be less than %f" % self.high)
            if np.any(X < self.low - self._eps):
                raise ValueError("All values should"
                                 "be greater than %f" % self.low)
        if (self.high - self.low) == 0.:
            return X * 0.
        if self.is_int:
            return (np.round(X).astype(np.int) - self.low) /\
                   (self.high - self.low)
        else:
            return (X - self.low) / (self.high - self.low)

    def inverse_transform(self, X):
        X = np.asarray(X)
        if np.any(X > 1.0 + self._eps):
            raise ValueError("All values should be less than 1.0")
        if np.any(X < 0.0 - self._eps):
            raise ValueError("All values should be greater than 0.0")
        X_orig = X * (self.high - self.low) + self.low
        if self.is_int:
            return np.round(X_orig).astype(np.int)
        return X_orig


class Pipeline(Transformer):
    """
    A lightweight pipeline to chain transformers.

    Parameters
    ----------
    transformers : list
        A list of Transformer instances.
    """
    def __init__(self, transformers):
        self.transformers = list(transformers)
        for transformer in self.transformers:
            if not isinstance(transformer, Transformer):
                raise ValueError(
                    "Provided transformers should be a Transformer "
                    "instance. Got %s" % transformer
                )

    def fit(self, X):
        for transformer in self.transformers:
            transformer.fit(X)
        return self

    def transform(self, X):
        for transformer in self.transformers:
            X = transformer.transform(X)
        return X

    def inverse_transform(self, X):
        for transformer in self.transformers[::-1]:
            X = transformer.inverse_transform(X)
        return X
    

class Rescale(Transformer):
    """
    Scales each dimension into the interval [a, b].

    Parameters
    ----------
    low : float
        Lower bound.
    high : float
        Higher bound.
    a : float
        New lower bound after transformation.
    b : float
        New upper bound after transformation.
    is_int : bool, default=True
        Round and cast the return value of `inverse_transform` to integer. Set
        to `True` when applying this transform to integers.
    """
    def __init__(self, low, high, a, b, is_int=False):
        self.low = float(low)
        self.high = float(high)
        self.a = float(a)
        self.b = float(b)
        self.is_int = is_int
        self._eps = 1e-8

    def transform(self, X):
        X = np.asarray(X)
        # Check bounds for input values
        if self.is_int:
            if np.any(np.round(X) > self.high):
                raise ValueError("All integer values should be less than %f" % self.high)
            if np.any(np.round(X) < self.low):
                raise ValueError("All integer values should be greater than %f" % self.low)
        else:
            if np.any(X > self.high + self._eps):
                raise ValueError("All values should be less than %f" % self.high)
            if np.any(X < self.low - self._eps):
                raise ValueError("All values should be greater than %f" % self.low)

        # Scale the values to the interval [a, b]
        if (self.high - self.low) == 0.:
            return X * 0. + self.a
        scale = (X - self.low) / (self.high - self.low)
        X_rescaled = self.a + scale * (self.b - self.a)
        if self.is_int:
            return np.round(X_rescaled).astype(np.int)
        return X_rescaled

    def inverse_transform(self, X):
        X = np.asarray(X)
        # Check bounds for scaled values
        if np.any(X > self.b + self._eps):
            raise ValueError("All values should be less than %f" % self.b)
        if np.any(X < self.a - self._eps):
            raise ValueError("All values should be greater than %f" % self.a)

        # Inverse scale the values to the original range
        scale = (X - self.a) / (self.b - self.a)
        X_orig = self.low + scale * (self.high - self.low)
        if self.is_int:
            return np.round(X_orig).astype(np.int)
        return X_orig


class Standardize(Transformer):
    """
    Standardizes each dimension to have zero mean and unit variance.

    Parameters
    ----------
    epsilon : float, default=1e-8
        A small constant added to the standard deviation to prevent division by zero.

    is_int : bool, default=False
        Round and cast the return value of `inverse_transform` to integer. Set
        to `True` when applying this transform to integers.
    """
    def __init__(self, epsilon=1e-8, is_int=False):
        self.mean = None
        self.std = None
        self.epsilon = epsilon
        self.is_int = is_int

    def fit(self, X):
        """
        Computes the mean and standard deviation of the data.
        
        Parameters
        ----------
        X : array-like
            The data to compute the statistics on.
        """
        X = np.asarray(X)
        self.mean = np.mean(X)
        self.std = np.std(X) + self.epsilon

    def transform(self, X):
        """
        Transforms the data using the computed mean and standard deviation.

        Parameters
        ----------
        X : array-like
            The data to transform.
        """
        X = np.asarray(X)
        X_standardized = (X - self.mean) / self.std
        if self.is_int:
            return np.round(X_standardized).astype(np.int)
        return X_standardized

    def inverse_transform(self, X):
        """
        Reverses the standardization transformation.

        Parameters
        ----------
        X : array-like
            The data to inverse transform.
        """
        X = np.asarray(X)
        X_original = X * self.std + self.mean
        if self.is_int:
            return np.round(X_original).astype(np.int)
        return X_original

from typing import Union, Optional, List, Tuple, Mapping
from pandas import Index
import numpy as np
import pandas as pd
from tsl import logger
from . import checks
from . import casting
from tsl.ops.dataframe import to_numpy
from tsl.typing import FrameArray
from tsl.utils.python_utils import ensure_list
from tsl.ops.framearray import framearray_shape, framearray_to_numpy
from tsl.ops.pattern import check_pattern, infer_pattern


class TabularParsingMixin:
    def _parse_target(self, obj: FrameArray) -> FrameArray:
        # if target is DataFrame
        if isinstance(obj, pd.DataFrame):
            casting.to_nodes_channels_columns(obj)
            obj = casting.convert_precision_df(obj, precision=self.precision)
        # if target is array-like
        else:
            obj = np.asarray(obj)
            # reshape to [time, nodes, features]
            while obj.ndim < 3:
                obj = obj[..., None]
            assert obj.ndim == 3, \
                "Target signal must be 3-dimensional with pattern 't n f'."
            obj = casting.convert_precision_numpy(obj, precision=self.precision)
        return obj

    def _parse_covariate(self, obj: FrameArray, pattern: Optional[str] = None) \
            -> Tuple[FrameArray, str]:
        # compute object shape
        shape = framearray_shape(obj)

        # infer pattern if it is None, otherwise sanity check
        if pattern is None:
            pattern = infer_pattern(shape, t=self.length, n=self.n_nodes)
        else:
            # check that pattern and shape match
            pattern = check_pattern(pattern, ndim=len(shape))

        dims = pattern.split(' ')  # 't n f' -> ['t', 'n', 'f']

        if isinstance(obj, pd.DataFrame):
            assert self.is_target_dataframe, \
                "Cannot add DataFrame covariates if target is ndarray."
            # check covariate index matches steps or nodes in the dataset
            # according to the dim token
            index = self._token_to_index(dims[0], obj.index)
            obj = obj.reindex(index=index)

            # todo check when columns is not multiindex
            #  add 1 dummy feature dim always?
            for lvl, tkn in enumerate(dims[1:]):
                columns = self._token_to_index(tkn, obj.columns.unique(lvl))
                if isinstance(obj.columns, pd.MultiIndex):
                    obj.reindex(columns=columns, level=lvl)
                else:
                    obj.reindex(columns=columns)
            obj = casting.convert_precision_df(obj, precision=self.precision)
        else:
            obj = np.asarray(obj)
            # check shape
            for d, s in zip(dims, obj.shape):
                self._token_to_index(d, s)
            obj = casting.convert_precision_numpy(obj, precision=self.precision)

        return obj, pattern

    def _token_to_index(self, token, index_or_size: Union[int, Index]):
        no_index = isinstance(index_or_size, int)
        if token == 't':
            if no_index:
                assert index_or_size == len(self.index)
            return self.index if self.force_synchronization else index_or_size
        if token == 'n':
            if no_index:
                assert index_or_size == len(self.nodes)
            else:
                assert set(index_or_size).issubset(self.nodes), \
                    "You are trying to add a covariate dataframe with " \
                    "nodes that are not in the dataset."
            return self.nodes
        if token in ['c', 'f'] and not no_index:
            return index_or_size

    def _columns_multiindex(self, nodes=None, channels=None):
        nodes = nodes if nodes is not None else self.nodes
        channels = channels if channels is not None else self.channels
        return pd.MultiIndex.from_product([nodes, channels],
                                          names=['nodes', 'channels'])

    def _value_to_kwargs(self, value: Union[FrameArray, List, Tuple, Mapping]):
        keys = ['value', 'pattern']
        if isinstance(value, (pd.DataFrame, np.ndarray)):
            return dict(value=value)
        if isinstance(value, (list, tuple)):
            return dict(zip(keys, value))
        elif isinstance(value, Mapping):
            assert set(value.keys()).issubset(keys)
            return value
        else:
            raise TypeError('Invalid type for value "{}"'.format(type(value)))




class PandasParsingMixin:

    def _parse_dataframe(self, df: pd.DataFrame, node_level: bool = True):
        assert checks.is_datetime_like_index(df.index)
        if node_level:
            df = checks.to_nodes_channels_columns(df)
        else:
            df = checks.to_channels_columns(df)
        df = checks.cast_df(df, precision=self.precision)
        return df

    def _to_indexed_df(self, array: np.ndarray):
        if array.ndim == 1:
            array = array[..., None]
        # check shape equivalence
        time, channels = array.shape
        if time != self.length:
            raise ValueError("Cannot match temporal dimensions {} and {}"
                             .format(time, self.length))
        return pd.DataFrame(array, self.index)

    def _to_primary_df_schema(self, array: np.ndarray):
        array = np.asarray(array)
        while array.ndim < 3:
            array = array[..., None]
        # check shape equivalence
        time, nodes, channels = array.shape
        if time != self.length:
            raise ValueError("Cannot match temporal dimensions {} and {}"
                             .format(time, self.length))
        if nodes != self.n_nodes:
            raise ValueError("Cannot match nodes dimensions {} and {}"
                             .format(nodes, self.n_nodes))
        array = array.reshape(time, nodes * channels)
        columns = self.columns(channels=pd.RangeIndex(channels))
        return pd.DataFrame(array, self.index, columns)

    def _synch_with_primary(self, df: pd.DataFrame):
        assert hasattr(self, 'df'), \
            "Cannot call this method before setting primary dataframe."
        if df.columns.nlevels == 2:
            nodes = set(df.columns.unique(0))
            channels = list(df.columns.unique(1))
            assert nodes.issubset(self.nodes), \
                "You are trying to add an exogenous dataframe with nodes that" \
                " are not in the dataset."
            columns = self.columns(channels=channels)
            df = df.reindex(index=self.index, columns=columns)
        elif df.columns.nlevels == 1:
            df = df.reindex(index=self.index)
        else:
            raise ValueError("Input dataframe must have either 1 ('nodes' or "
                             "'channels') or 2 ('nodes', 'channels') column "
                             "levels.")
        return df

    def _check_name(self, name: str, check_type: str):
        assert check_type in ['exogenous', 'attribute']
        invalid_names = set(dir(self))
        if check_type == 'exogenous':
            invalid_names.update(self._attributes)
        else:
            invalid_names.update(self._exogenous)
        if name in invalid_names:
            raise ValueError(f"Cannot set {check_type} with name '{name}', "
                             f"{self.__class__.__name__} contains already an "
                             f"attribute named '{name}'.")


class TemporalFeaturesMixin:

    def datetime_encoded(self, units):
        units = ensure_list(units)
        mapping = {un: pd.to_timedelta('1' + un).delta
                   for un in ['day', 'hour', 'minute', 'second',
                              'millisecond', 'microsecond', 'nanosecond']}
        mapping['week'] = pd.to_timedelta('1W').delta
        mapping['year'] = 365.2425 * 24 * 60 * 60 * 10 ** 9
        index_nano = self.index.view(np.int64)
        datetime = dict()
        for unit in units:
            if unit not in mapping:
                raise ValueError()
            nano_sec = index_nano * (2 * np.pi / mapping[unit])
            datetime[unit + '_sin'] = np.sin(nano_sec)
            datetime[unit + '_cos'] = np.cos(nano_sec)
        return pd.DataFrame(datetime, index=self.index, dtype=np.float32)

    def datetime_onehot(self, units):
        units = ensure_list(units)
        datetime = dict()
        for unit in units:
            if hasattr(self.index.__dict__, unit):
                raise ValueError()
            datetime[unit] = getattr(self.index, unit)
        dummies = pd.get_dummies(pd.DataFrame(datetime, index=self.index),
                                 columns=units)
        return dummies


class MissingValuesMixin:
    eval_mask: np.ndarray

    def set_eval_mask(self, eval_mask: FrameArray):
        if isinstance(eval_mask, pd.DataFrame):
            eval_mask = to_numpy(self._parse_dataframe(eval_mask))
        if eval_mask.ndim == 2:
            eval_mask = eval_mask[..., None]
        assert eval_mask.shape == self.shape
        eval_mask = eval_mask.astype(self.mask.dtype) & self.mask
        self.eval_mask = eval_mask

    @property
    def training_mask(self):
        if hasattr(self, 'eval_mask') and self.eval_mask is not None:
            return self.mask & (1 - self.eval_mask)
        return self.mask

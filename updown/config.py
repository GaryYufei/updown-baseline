r"""This module provides package-wide configuration management."""
from typing import Any, List

from yacs.config import CfgNode as CN


class Config(object):
    r"""
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be overriden by (first) through a YAML file and (second) through
    a list of attributes and values.

    Extended Summary
    ----------------
    This class definition contains default values corresponding to ``joint_training`` phase, as it
    is the final training phase and uses almost all the configuration parameters. Modification of
    any parameter after instantiating this class is not possible, so you must override required
    parameter values in either through ``config_yaml`` file or ``config_override`` list.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override. This happens after
        overriding from YAML file.

    Attributes
    ----------
    RANDOM_SEED: 0
        Random seed for NumPy and PyTorch, important for reproducibility.

    todo (kd): document other attributes.
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()

        self._C.MODEL = CN()
        self._C.MODEL.INPUT_SIZE = 1000
        self._C.MODEL.HIDDEN_SIZE = 1200
        self._C.MODEL.IMAGE_FEATURE_SIZE = 2048
        self._C.MODEL.ATTENTION_PROJECTION_SIZE = 768

        # Override parameter values from YAML file first, then from override list.
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __str__(self):
        return _config_str(self)

    def __repr__(self):
        return self._C.__repr__()


def _config_str(config: Config) -> str:
    r"""
    Collect a subset of config in sensible order (not alphabetical) according to phase. Used by
    :func:`Config.__str__()`.

    Parameters
    ----------
    config: Config
        A :class:`Config` object which is to be printed.
    """
    _C = config

    __C: CN = CN({"RANDOM_SEED": _C.RANDOM_SEED})
    common_string: str = str(__C) + "\n"
    common_string += str(_C.MODEL) + "\n"

    return common_string

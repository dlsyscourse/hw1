"""Computational engine interface."""
from enum import IntEnum
from typing import Union, List
import numpy as np


class DLDeviceType(IntEnum):
    CPU = 1
    CUDA = 2


class BackendNDArrayBase:
    """Base class of all on-device ndarray without AD."""


# The type of underlying data array
CachedData = Union[np.ndarray, BackendNDArrayBase]


class Device:
    """Device is the interface of the underlying computational backend."""

    def __dlpack_device__(self):
        """Get the dlpack encoding of the device."""
        raise NotImplementedError()

    def __eq__(self, other):
        return (
            isinstance(other, Device)
            and self.__dlpack_device__() == other.__dlpack_device__()
        )

    def enabled(self):
        return True

    def array(self, array, dtype):
        """Create a new device array for a given python array."""
        raise NotImplementedError()

    def empty(self, shape, dtype):
        """Create an empty device array for a given python array."""
        raise NotImplementedError()

    def fill(self, device_array, fill_value):
        """Fill the device array with corresponding element value."""
        raise NotImplementedError()

    def to_numpy(self, device_array):
        """Convert device_array to numpy array."""
        raise NotImplementedError()

    def compute(
        self, op: "Op", input_data: List[CachedData], attrs: object
    ) -> CachedData:
        """Execute the underlying computation of op on inputs."""
        raise NotImplementedError()


def default_device():
    """The default computational device."""
    return _DEFAULT_DEVICE()


# Note: to be assigned to NumpyDevice when we import
# keep this here to avoid cyclic dependency
_DEFAULT_DEVICE = None

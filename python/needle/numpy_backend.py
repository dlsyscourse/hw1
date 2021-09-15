"""Numpy computation backend.

This backend uses numpy for cached data and redirects
all computations to corresponding numpy functions.
"""
import numpy as np
import needle.device
from needle.device import Device, DLDeviceType
from needle.ops import register_op_attr


class NumpyDevice(Device):
    def __dlpack_device__(self):
        return (DLDeviceType.CPU, 0)

    def __repr__(self):
        return "numpy_device()"

    def __str__(self):
        return self.__repr__()

    def array(self, array, dtype):
        return np.array(array, dtype=dtype)

    def empty(self, shape, dtype):
        return np.empty(shape, dtype=dtype)

    def fill(self, arr, fill_value):
        arr.fill(fill_value)
        return arr

    def to_numpy(self, data):
        return data

    def compute(self, op, inputs, attrs):
        """Dispatch device specific computation"""
        # dispatch device specific compute to op.numpy_compute
        # these computation are registered below.
        return op.numpy_compute(inputs, attrs)


# set default device to be numpy device.
needle.device._DEFAULT_DEVICE = NumpyDevice


def numpy_device() -> NumpyDevice:
    return NumpyDevice()


def register_numpy_compute(name, value=None):
    """Register the numpy compute property"""
    return register_op_attr(name, "numpy_compute", value)


# device specific computations
@register_numpy_compute("EWiseAdd")
def add(inputs, attrs):
    assert len(inputs) == 2
    return np.add(inputs[0], inputs[1])


@register_numpy_compute("AddScalar")
def add_scalar(inputs, attrs):
    assert len(inputs) == 1
    return np.add(inputs[0], attrs["scalar"])


@register_numpy_compute("EWiseMul")
def multiply(inputs, attrs):
    return np.multiply(inputs[0], inputs[1])


@register_numpy_compute("MulScalar")
def multiply_scalar(inputs, attrs):
    return np.multiply(inputs[0], attrs["scalar"])


@register_numpy_compute("EWiseDiv")
def divide(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


@register_numpy_compute("DivScalar")
def divide_scalar(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


@register_numpy_compute("MatMul")
def matmul(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


@register_numpy_compute("Summation")
def summation(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


@register_numpy_compute("BroadcastTo")
def broadcast_to(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


@register_numpy_compute("Reshape")
def reshape(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


@register_numpy_compute("Negate")
def negate(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


@register_numpy_compute("Transpose")
def transpose(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


@register_numpy_compute("Log")
def log(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


@register_numpy_compute("Exp")
def exp(inputs, attrs):
    return np.exp(inputs[0])


@register_numpy_compute("ReLU")
def relu(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION

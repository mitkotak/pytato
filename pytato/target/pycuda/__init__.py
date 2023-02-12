from __future__ import annotations

__copyright__ = """Copyright (C) 2022 Mit Kotak"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

__doc__ = """
.. currentmodule:: pytato

.. autofunction:: generate_cudagraph

.. currentmodule:: pytato.target.python

.. autoclass:: PythonTarget
.. autoclass:: BoundPythonProgram
.. autoclass:: CUDAGraphTarget
.. autoclass:: BoundCUDAGraphProgram
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Mapping, FrozenSet, Callable, Dict, Set

from pytato.target import Target, BoundProgram


# {{{ abstract types

class PythonTarget(Target, ABC):
    """
    A target that generates code for a python program, typically by invoking
    some :mod:`numpy`-like for the array operations.
    """

    @abstractmethod
    def bind_program(self,
                     program: str,
                     entrypoint: str,
                     expected_arguments: FrozenSet[str],
                     bound_arguments: Mapping[str, Any]) -> BoundProgram:
        """
        :arg program: The python code containing the compiled routine.
        :arg entrypoint: Name of the entrypoint
        """
        pass


@dataclass(repr=False, eq=False)
class BoundCUDAGraphProgram(BoundProgram):
    """
    A wrapper for executing python programs with bound arguments.

    .. automethod:: __call__
    .. automethod:: copy
    .. automethod:: with_transformed_program
    """
    expected_arguments: FrozenSet[str]
    entrypoint: str

    @cached_property
    def _compiled_function(self) -> Callable[..., Any]:
        variables_after_execution: Dict[str, Any] = {
            "_MODULE_SOURCE_CODE": self.program,  # helps pudb
            "cuda_allocator": self.cuda_allocator,
            "cuda_dev": self.cuda_dev
        }
        exec(self.program, variables_after_execution)
        assert callable(variables_after_execution[self.entrypoint])
        return variables_after_execution[  # type: ignore[no-any-return]
            self.entrypoint]

    @cached_property
    def _bound_argment_names(self) -> Set[str]:
        return set(self.bound_arguments.keys())

    @cached_property
    def _processed_bound_args(self) -> Mapping[str, Any]:
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray
        import numpy as np

        processed_bound_args = {}
        for name, arg in zip(self.bound_arguments.keys(),
        self.bound_arguments.values()):
            if isinstance(arg, np.ndarray):
                processed_bound_args[name] = gpuarray.to_gpu(arg)
            elif isinstance(arg, gpuarray.GPUArray):
                processed_bound_args[name] = arg
            else:
                raise ValueError("Array format not supported")
        return processed_bound_args

    def __call__(self, cuda_allocator=None, *args: Any, **kwargs: Any) -> Any:
        if bool(cuda_allocator):
            self.cuda_allocator = cuda_allocator
        else:
            import pycuda
            self.cuda_allocator = pycuda.tools.DeviceMemoryPool().allocate
        import pycuda.driver as drv
        self.cuda_dev = drv.Context.get_device()
        if args:
            raise ValueError(f"'{type(self).__call__}' does not take positional"
                             " arguments.")

        if set(kwargs.keys()) & self._bound_argment_names:
            raise ValueError("Got arguments that were previously bound: "
                    f"'{set(kwargs.keys()) & set(self.bound_arguments.keys())}'.")
        updated_kwargs = dict(self._processed_bound_args)
        updated_kwargs.update(kwargs)
        updated_kwargs = {kw: arg
                          for kw, arg in updated_kwargs.items()
                          if kw in self.expected_arguments}
        return self._compiled_function(**updated_kwargs)

    def copy(self, **kwargs: Any) -> BoundCUDAGraphProgram:
        from dataclasses import replace
        return replace(self, **kwargs)

    def with_transformed_program(self, *args: Any, **kwargs: Any
                                 ) -> BoundCUDAGraphProgram:
        raise ValueError("Cannot transform python program.")

# }}}


# {{{ cudagraph-numpy target


class CUDAGraphTarget(PythonTarget):
    """
    A target that generates code for a python program by offloading array
    operations to :mod:`cudagraph.cudagraph`.
    """

    @property
    def numpy_like_module_name(self) -> str:
        return "pycuda.driver"

    @property
    def numpy_like_module_name_shorthand(self) -> str:
        return "_pt_drv"

    def bind_program(self,
                     program: str,
                     entrypoint: str,
                     expected_arguments: FrozenSet[str],
                     bound_arguments: Mapping[str, Any]) -> BoundCUDAGraphProgram:
        return BoundCUDAGraphProgram(target=self, program=program,
                                     entrypoint=entrypoint,
                                     expected_arguments=expected_arguments,
                                     bound_arguments=bound_arguments)

# }}}

# vim: foldmethod=marker

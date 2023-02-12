__copyright__ = """
Copyright (C) 2022 Mit Kotak
"""

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

from typing import Union, Optional, Mapping

from pytato.array import Array, DictOfNamedArrays
from pytato.target.pycuda import CUDAGraphTarget, BoundCUDAGraphProgram
from pytato.target.pycuda.numpy_like import generate_cudagraph_numpy_like

__doc__ = """
.. autofunction:: generate_cudagraph
"""


def generate_cudagraph(expr: Union[Array, Mapping[str, Array], DictOfNamedArrays],
                 *,
                 target: Optional[CUDAGraphTarget] = None,
                 function_name: str = "_pt_kernel",
                 show_code: bool = False,
                 colorize_show_code: bool = True,
                 dot_graph_path: Optional[str] = "") -> BoundCUDAGraphProgram:
    """
    Returns a :class:`pytato.target.python.BoundCUDAGraphProgram` for the array
    expressions in *expr*.
    :arg function: Name of the entrypoint function in the generated code.
    :arg show_code: If *True*, the generated code is printed to ``stdout``.
    """
    if target is None:
        target = CUDAGraphTarget()
    assert isinstance(target, CUDAGraphTarget)
    return generate_cudagraph_numpy_like(expr,
                               target=target,
                               function_name=function_name,
                               show_code=show_code,
                               colorize_show_code=colorize_show_code,
                               dot_graph_path=dot_graph_path)

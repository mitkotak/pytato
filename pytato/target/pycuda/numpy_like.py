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

import ast
import sys
import os
import numpy as np
import loopy as lp

from typing import (Callable, Union, Optional, Mapping, Dict, TypeVar,
                    List, Set, Tuple, Any)

import pymbolic.primitives as prim
from pytools import (UniqueNameGenerator, memoize_method)
from pytato.transform import CachedMapper, ArrayOrNames
from pytato.array import (IndexLambda, DataWrapper,
                          Placeholder, SizeParam,
                          Array, DictOfNamedArrays,
                          DataInterface)
from pytato.scalar_expr import (IdentityMapper, Reduce,
                                WalkMapper as ScalarWalkMapper)
from pytato.target.pycuda import (CUDAGraphTarget,
                                  BoundCUDAGraphProgram)
from pyrsistent import pmap
from loopy.symbolic import Reduction as LoopyReduction
from pytato.target.loopy.codegen import PYTATO_REDUCTION_TO_LOOPY_REDUCTION
from dataclasses import dataclass

T = TypeVar("T")


def _can_colorize_output() -> bool:
    try:
        import pygments  # noqa: F401
        return True
    except ImportError:
        return False


def _get_default_colorize_code() -> bool:
    return ((not sys.stdout.isatty())
            # https://no-color.org/
            and "NO_COLOR" not in os.environ)


def fill_tuple(dim_tuple: Tuple[int, ...]) -> Tuple[int, ...]:
    while len(dim_tuple) < 3:
        dim_tuple += (1,)
    return dim_tuple


def _splay(n: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    # heavily modified from cublas
    min_threads = 32
    max_threads = 128
    max_blocks = (
        4
        * 8
        * 64
    )

    if n < min_threads:
        block_count = 1
        threads_per_block = min_threads
    elif n < (max_blocks * min_threads):
        block_count = (n + min_threads - 1) // min_threads
        threads_per_block = min_threads
    elif n < (max_blocks * max_threads):
        block_count = max_blocks
        grp = (n + min_threads - 1) // min_threads
        threads_per_block = ((grp + max_blocks - 1) // max_blocks) * min_threads
    else:
        block_count = max_blocks
        threads_per_block = max_threads

    # print "n:%d bc:%d tpb:%d" % (n, block_count, threads_per_block)
    return (block_count, 1), (threads_per_block, 1, 1)


class _RednVarToBoundRecorder(ScalarWalkMapper):
    def __init__(self) -> None:
        self.redn_var_to_bounds: Mapping[
                                    Any, Tuple[int, Union[int, Any, Array]]
                                        ] = {}
        super().__init__()

    def map_reduce(self, expr: Reduce) -> None:
        for redn_var, (lbound, ubound) in expr.bounds.items():
            assert redn_var not in self.redn_var_to_bounds
            self.redn_var_to_bounds[redn_var] = (lbound, ubound)


def _get_dim_to_bounds(
                    expression,
                    shape
                    ) -> Mapping[str, Tuple[int, Union[int, Any, Array]]]:
    redn_var_to_bound_recorder = _RednVarToBoundRecorder()
    redn_var_to_bound_recorder(expression)
    return {**redn_var_to_bound_recorder.redn_var_to_bounds,
            **{f"_{i}": (0, dim) for i, dim in enumerate(shape)}}


def _get_knl_name(expr: IndexLambda) -> str:
    from pytato.raising import (index_lambda_to_high_level_op,
                                    BinaryOp, WhereOp,
                                    BroadcastOp,
                                    C99CallOp,
                                    FullOp)
    from pytato.diagnostic import (UnknownIndexLambdaExpr)
    try:
        hlo = index_lambda_to_high_level_op(expr)
        if isinstance(hlo, BinaryOp):
            knl_string = str(hlo.binary_op)[13:] + \
                "_x1_" + str(hlo.x1.size) + \
                "_x2_" + str(hlo.x2.size)
        elif isinstance(hlo, C99CallOp):
            from pytato.target.python.numpy_like import _c99_callop_numpy_name
            knl_string = str(_c99_callop_numpy_name(hlo))
        elif isinstance(hlo, WhereOp):
            knl_string = "where"
        elif isinstance(hlo, FullOp):
            knl_string = "fill_" + str(int(hlo.fill_value))
        elif isinstance(hlo, BroadcastOp):
            knl_string = "broadcast"
        else:
            knl_string = str(type(hlo)).split(".")[-1][:-2]
    except UnknownIndexLambdaExpr:
        if isinstance(expr.expr, Reduce):
            knl_string = str(type(expr.expr.op)).split(".")[-1][:-2]
        elif isinstance(expr.expr, prim.Subscript):
            knl_string = "subscript"
        elif isinstance(expr.expr, prim.Product):
            knl_string = "product"
        elif isinstance(expr.expr, prim.If):
            knl_string = "if"
        else:
            knl_string = str(type(expr)).split(".")[-1][:-2]
    except Exception:
        knl_string = str(type(expr)).split(".")[-1][:-2]
    return "knl_" + knl_string.lower()


@dataclass(frozen=True, eq=True)
class LoopyArg:
    expr: prim.Expression
    shape: Tuple[Any, ...]
    dtype: np.dtype[Any]

    @property
    def size(self):
        from pytools import product
        return product(self.shape)

    @property
    def ndim(self):
        return len(self.shape)


class CgenMapperAccumulator:
    def __init__(self,
                buffer_name: str,
                dep_nodes: Optional[Tuple[str, ...]] = tuple(),
                buffers: Optional[Tuple[str, ...]] = tuple(),
                is_allocated: Optional[bool] = False):

        self.buffer_name = buffer_name
        self.dep_nodes = dep_nodes
        self.is_allocated = is_allocated
        self.buffers = buffers


class BufferAccumulator:
    def __init__(self,
                array_name: str,
                mapped_array_name: str,
                stmt: ast.stmt):
        self.array_name = array_name
        self.mapped_array_name = mapped_array_name
        self.stmt = stmt


class NumpyCodegenMapper(CachedMapper[ArrayOrNames]):
    """
    .. note::
        - This mapper stores mutable state for building the program. The same
          mapper instance must be re-used with care.
    """
    def __init__(self,
                graph_name: str,
                buffer_accumulator: str,
                node_accumulator: str,
                numpy_backend: str,
                vng: Callable[[str], str],
                bound_arguments: Dict[str, DataInterface],
                extra_preambles: List[ast.stmt],
                dot_graph_path: Optional[str] = ""):
        super().__init__()
        self.graph_name = graph_name
        self.buffer_accumulator = buffer_accumulator
        self.node_accumulator = node_accumulator
        self.numpy_backend = numpy_backend
        self.vng = vng
        self.bound_arguments = bound_arguments
        self.extra_preambles = extra_preambles
        self.dot_graph_path = dot_graph_path
        self.lines_graph_exec: List[ast.stmt] = []
        self.lines_pt_kernel: List[ast.stmt] = []
        self.lines_output_allocation: List[ast.stmt] = []
        self.allocated_buffers_to_users: Mapping[str, Set[str]] = {}
        self.input_args: Mapping[str, Placeholder] = {}
        self.kernel_to_kernel_args: Mapping[str, List[str]] = {}
        self.temp_input_arrays_to_output_stmt_mapper = {}
        self.dict_line: List[ast.stmt] = []

    @memoize_method
    def _add_source_module_line(self, knl_name, lp_arg_acc, bindings) -> Tuple[Any,
                                                Tuple[int, ...],
                                                Tuple[int, ...],
                                                str]:
        pt2lp_mapper = PytatoExprToLoopyExprMapper()
        lp_expr = pt2lp_mapper(lp_arg_acc.expr)
        dim_to_bounds = _get_dim_to_bounds(lp_arg_acc.expr, lp_arg_acc.shape)
        import loopy as lp
        from pymbolic import var
        all_dims = ", ".join(list(dim_to_bounds))
        bounds = " and ".join([f"{lbound} <= {dim} < {ubound}"
                            for dim, (lbound, ubound)
                            in list(dim_to_bounds.items())])
        out_var = var("out")[tuple(var(f"_{i}") for i in range(lp_arg_acc.ndim))]
        knl = lp.make_kernel(
            #FIXME : Need to remove the if condition for null domains
            name=knl_name,
            domains="{ [%s]: %s }" % (all_dims, bounds) if bool(
                dim_to_bounds) else "{:}",
            instructions=[lp.Assignment(
                                out_var,
                                lp_expr,
                                within_inames=frozenset(
                                    {f"_{i}" for i in range(lp_arg_acc.ndim)}
                                    ))],
            kernel_data=[
                lp.GlobalArg("out", shape=lp_arg_acc.shape, dtype=lp_arg_acc.dtype,
                offset=0),
                *[lp.GlobalArg(name, shape=bnd[0], dtype=bnd[1],
                    offset=0)
                  for name, bnd in list(sorted(bindings.items()))]],
            target=lp.PyCudaTarget(),
            lang_version=(2018, 2))

        grid, block = _splay(lp_arg_acc.size * lp_arg_acc.dtype.itemsize)
        knl = lp.join_inames(knl, [f"_{i}" for i in range(lp_arg_acc.ndim)], "iout")
        knl = lp.split_iname(knl, "iout", np.prod(
            grid) * np.prod(block), outer_iname="ibatch")
        knl = lp.split_iname(knl, "iout_inner", np.prod(
            block), inner_iname="ithread", outer_iname="ithreadblock")
        knl = lp.split_iname(
            knl, "ithread", block[0],
            inner_iname="ithread_x", outer_iname="ithread_y")
        knl = lp.split_iname(knl, "ithreadblock",
                            grid[1],
                            outer_iname="iblock_x", inner_iname="iblock_y")
        knl = lp.tag_inames(
            knl, {
                "iblock_x": "g.0",
                "iblock_y": "g.1",
                "ithread_x": "l.0",
                "ithread_y": "l.1"
                })
        grid, block = knl.default_entrypoint.get_grid_size_upper_bounds_as_exprs(
            knl.callables_table)
        grid, block = fill_tuple(grid), fill_tuple(block)
        knl_string = lp.generate_code_v2(knl).device_code()
        mod = self.vng("_pt_mod")
        source_mod_line = ast.Assign(targets=[ast.Name(mod)],
                                    value=ast.Call(ast.Name("_pt_SourceModule"),
                                    args=[ast.Constant(knl_string)],
                                    keywords=[]))
        self.extra_preambles.append(source_mod_line)
        return mod, grid, block

    def map_index_lambda(self, expr: IndexLambda) -> CgenMapperAccumulator:
        array_name = self.vng("_pt_array")
        array_malloc_node_name = self.vng("_pt_memalloc")
        mod = self.vng("_pt_mod")
        kernel_call_node_name = self.vng("_pt_kernel")
        if expr.size == 0:
            return CgenMapperAccumulator(buffer_name=array_name)

        def _get_dependencies(
                            array_name: str,
                            kernel_call_node_name: str
                            ) -> Tuple[Tuple[str, ...], Mapping[str, ast.Name]]:
            dependencies = ()
            buffer_accumulator_lines = []
            buffers_kernel = [array_name]
            buffers = [array_name]
            buffer_accumulator_lines.append(
                        ast.Assign(
                            targets=[ast.Subscript(ast.Name(self.buffer_accumulator),
                                                   ast.Constant(array_name))],
                            value=ast.Name(array_name)
                        )
                    )
            for _, bnd in list(expr.bindings.items()):
                rec_bnd = self.rec(bnd)
                if rec_bnd.is_allocated:
                    self.allocated_buffers_to_users[rec_bnd.buffer_name].add(
                        kernel_call_node_name)
                    buffer_accumulator_lines.append(
                        ast.Assign(
                            targets=[
                                ast.Subscript(ast.Name(self.buffer_accumulator),
                                              ast.Constant(rec_bnd.buffer_name))],
                            value=ast.Name(rec_bnd.buffer_name)
                        )
                    )
                    buffers_kernel.append(rec_bnd.buffer_name)
                else:
                    buffers_kernel.append("139712027164672")
                buffers.append(rec_bnd.buffer_name)
                dependencies += rec_bnd.dep_nodes

            self.kernel_to_kernel_args[kernel_call_node_name] = buffers
            return (tuple(set(dependencies)),
                    buffers, buffers_kernel,
                    buffer_accumulator_lines)

        output_tuple = _get_dependencies(array_name, kernel_call_node_name)
        dependencies = output_tuple[0]
        buffers = output_tuple[1]
        buffers_kernel = output_tuple[2]
        buffer_accumulator_lines = output_tuple[3]
        array_memalloc_line = ast.Assign(targets=[ast.Tuple(
                                        elts=[ast.Name(array_malloc_node_name),
                                        ast.Name(array_name)])],
                                        value=ast.Call(
                                                    ast.Attribute(
                                                        ast.Name(self.graph_name),
                                                        "add_memalloc_node"),
                                                    args=[],
                                                    keywords=[
                                                        ast.keyword(
                                                            "size",
                                                            ast.Constant(
                                                                expr.size
                                                                *
                                                                expr.dtype.itemsize
                                                                )),
                                                        ast.keyword(
                                                            "dependencies",
                                                            ast.List(
                                                                elts=[
                                                                    ast.Name(dep)
                                                                    for
                                                                    dep in
                                                                    dependencies]))
                                                                    ]))
        self.lines_graph_exec.append(array_memalloc_line)
        self.allocated_buffers_to_users[array_name] = set()
        self.allocated_buffers_to_users[array_name].add(kernel_call_node_name)
        knl_name = _get_knl_name(expr)
        lp_arg_acc = LoopyArg(expr.expr,
                              expr.shape,
                              expr.dtype)

        import immutables
        mod, grid, block = self._add_source_module_line(knl_name,
                                                        lp_arg_acc,
                                                        immutables.Map(
                                                            {name: (
                                                                bnd.shape, bnd.dtype
                                                                )
                                                             for name, bnd
                                                             in expr.bindings.items()
                                                             }))
        kernel_call_line = ast.Assign(targets=[
                    ast.Name(kernel_call_node_name)],
                    value=ast.Call(ast.Attribute(ast.Name(self.graph_name),
                                "add_kernel_node"),
                                args=[
                                    ast.Name(buffer_name)
                                    for buffer_name in buffers_kernel
                                    ],
                                keywords=[ast.keyword(
                                                arg="func",
                                                value=ast.Call(
                                                    ast.Attribute(
                                                        ast.Name(mod),
                                                        "get_function"),
                                                    args=[
                                                        ast.Constant(knl_name)
                                                        ],
                                                    keywords=[])),
                                          ast.keyword(
                                                arg="block",
                                                value=ast.Tuple(
                                                    elts=[
                                                        ast.Constant(d)
                                                        for d in block
                                                        ])),
                                          ast.keyword(
                                                arg="grid",
                                                value=ast.Tuple(
                                                    elts=[
                                                        ast.Constant(d)
                                                        for d in grid
                                                        ])),
                                          ast.keyword(
                                                arg="dependencies",
                                                value=ast.List(
                                                    elts=[
                                                        ast.Name(
                                                            array_malloc_node_name
                                                            )] + [
                                                            ast.Name(
                                                                dep
                                                                )
                                                            for dep in dependencies
                                                            ]))]))
        self.lines_graph_exec.append(kernel_call_line)
        self.lines_graph_exec += buffer_accumulator_lines
        self.lines_graph_exec.append(
            ast.Assign(
                targets=[ast.Subscript(ast.Name(self.node_accumulator),
                                        ast.Constant(kernel_call_node_name))],
                value=ast.Name(kernel_call_node_name)
                )
            )
        return CgenMapperAccumulator(
            buffer_name=array_name,
            dep_nodes=(kernel_call_node_name,),
            buffers=buffers,
            is_allocated=True)

    def map_placeholder(self, expr: Placeholder) -> CgenMapperAccumulator:
        self.input_args[expr.name] = expr
        input_buffer_name = self.vng("_pt_buffer")
        self.temp_input_arrays_to_output_stmt_mapper[
            input_buffer_name] = ast.Attribute(ast.Name(expr.name), "gpudata")
        return CgenMapperAccumulator(buffer_name=input_buffer_name)

    def map_size_param(self, expr: SizeParam) -> str:
        # would demand a more complicated BoundProgram implementation.
        raise NotImplementedError("SizeParams not yet supported  in numpy-targets.")

    def map_dict_of_named_arrays(self,
                                expr: DictOfNamedArrays) -> CgenMapperAccumulator:
        dict_name = self.vng("_pt_tmp")
        keys = []
        values = []
        buffer_to_result_mapper = {}
        for name, subexpr in list(expr._data.items()):
            keys.append(ast.Constant(name))
            rec = self.rec(subexpr)
            result_array_name = self.vng("_pt_result")
            if bool(rec.buffers):
                if rec.buffer_name in buffer_to_result_mapper:
                    values.append(ast.Name(buffer_to_result_mapper[rec.buffer_name]))
                else:
                    buffer_to_result_mapper[rec.buffer_name] = result_array_name
                    self.lines_output_allocation.append(
                        ast.Assign(
                            targets=[ast.Name(result_array_name)],
                            value=ast.Call(
                                ast.Attribute(ast.Name("_pt_gpuarray"), "GPUArray"),
                                args=[ast.Tuple(elts=[
                                    ast.Constant(dim) for dim in subexpr.shape
                                    ])],
                                keywords=[
                                    ast.keyword(
                                        arg="dtype",
                                        value=ast.Constant(subexpr.dtype.name)),
                                    ast.keyword(
                                        arg="allocator",
                                        value=ast.Name("allocator")),
                                    ast.keyword(
                                        arg="dev",
                                        value=ast.Name("dev"))])))
                    values.append(ast.Name(result_array_name))
                    self.temp_input_arrays_to_output_stmt_mapper[
                        rec.buffers[0]] = ast.Attribute(
                            ast.Name(result_array_name), "gpudata")
            else:
                if subexpr.size != 0:
                    values.append(ast.Name(subexpr.name))
                else:
                    self.lines_output_allocation.append(
                        ast.Assign(
                            targets=[ast.Name(result_array_name)],
                            value=ast.Call(
                                ast.Attribute(
                                    ast.Name("_pt_gpuarray"),
                                    "GPUArray"
                                    ),
                                args=[
                                    ast.Tuple(
                                        elts=[
                                            ast.Constant(dim)
                                            for dim in subexpr.shape
                                            ])],
                                keywords=[
                                    ast.keyword(
                                        arg="dtype",
                                        value=ast.Constant(subexpr.dtype.name)),
                                    ast.keyword(
                                        arg="allocator",
                                        value=ast.Name("allocator"))])))
                    values.append(ast.Name(result_array_name))
        for array_name, dependencies in self.allocated_buffers_to_users.items():
            self.lines_graph_exec.append(
                ast.Expr(
                    ast.Call(
                        ast.Attribute(ast.Name("_pt_g"),
                        "add_memfree_node"),
                        args=[
                            ast.Name(array_name),
                            ast.List(
                                elts=[ast.Name(dep) for dep in dependencies]
                                )],
                        keywords=[])))
        if bool(self.dot_graph_path):
            self.lines_graph_exec.append(
                ast.Expr(
                    ast.Call(
                        ast.Attribute(
                            ast.Name(self.graph_name),
                            "show_dot_graph"),
                        args=[ast.Constant(self.dot_graph_path)],
                        keywords=[])))

        self.dict_line = [ast.Assign(targets=[ast.Name(dict_name)],
                                value=ast.Dict(keys=keys, values=values))]

        return CgenMapperAccumulator(dict_name, ())


class PytatoExprToLoopyExprMapper(IdentityMapper):

    @memoize_method
    def call(self, expr):
        return super().call(expr)

    def map_reduce(self, expr: Reduce) -> LoopyReduction:
        try:
            loopy_redn = PYTATO_REDUCTION_TO_LOOPY_REDUCTION[type(expr.op)]
        except KeyError:
            raise NotImplementedError(expr.op)
        return LoopyReduction(loopy_redn,
                              tuple(list(expr.bounds)),
                              self.rec(expr.inner_expr))

    def map_call(self, expr: prim.Call) -> prim.Call:
        if isinstance(expr.function, prim.Variable) and (
                expr.function.name.startswith("pytato.c99.")):
            name_in_loopy = expr.function.name[11:]
            return prim.Call(prim.Variable(name_in_loopy),
                             self.rec(expr.parameters))

        return super().map_call(expr)

    def map_subscript(self, expr: prim.Subscript) -> prim.Subscript:
        aggregate = self.rec(expr.aggregate)
        index = self.rec(expr.index)
        if aggregate is expr.aggregate and index is expr.index:
            if not bool(expr.index):
                return aggregate
            else:
                return expr
        return type(expr)(aggregate, index)


def generate_cudagraph_numpy_like(expr: Union[
                                        Array,
                                        Mapping[str, Array],
                                        DictOfNamedArrays],
                                  target: CUDAGraphTarget,
                                  function_name: str,
                                  show_code: bool,
                                  colorize_show_code: Optional[bool] = None,
                                  dot_graph_path: Optional[str] = ""
                                  ) -> BoundCUDAGraphProgram:

    extra_preambles = [ast.ImportFrom(module="pycuda.driver",
                                              names=[ast.alias(
                                                  "KernelNodeParams",
                                                  asname="_pt_KernelNodeParams")],
                                              level=0),
                      ast.ImportFrom(module="pycuda.compiler",
                                              names=[ast.alias(
                                                  "SourceModule",
                                                  asname="_pt_SourceModule")],
                                              level=0),
                      ast.ImportFrom(module="pycuda",
                                            names=[ast.alias(
                                                "gpuarray",
                                                asname="_pt_gpuarray")],
                                            level=0),
                      ast.ImportFrom(module="functools",
                                            names=[ast.alias(
                                                "cache",
                                                asname="")],
                                            level=0)]
    import collections
    from pytato.transform import InputGatherer

    if ((not isinstance(expr, DictOfNamedArrays))
            and isinstance(expr, collections.abc.Mapping)):
        from pytato.array import make_dict_of_named_arrays
        expr = make_dict_of_named_arrays(dict(expr))

    assert isinstance(expr, (Array, DictOfNamedArrays))

    var_name_gen = UniqueNameGenerator()

    var_name_gen.add_names({input_expr.name
                            for input_expr in InputGatherer()(expr)
                            if isinstance(input_expr,
                                          (Placeholder, SizeParam, DataWrapper))
                            if input_expr.name is not None})

    if isinstance(expr, DictOfNamedArrays):
        var_name_gen.add_names(expr)

    var_name_gen.add_names({target.numpy_like_module_name_shorthand,
                            "np",
                            function_name})
    dict_flag = False
    if isinstance(expr, DictOfNamedArrays):
        dict_flag = True
    from pytato.codegen import normalize_outputs as normalize_outputs
    from pytato.target.loopy.codegen import preprocess as preprocess_loopy
    result = normalize_outputs(expr)
    preproc_result = preprocess_loopy(result, target=lp.CudaTarget())
    graph_name = var_name_gen("_pt_g")
    buffer_accumulator = var_name_gen("_pt_buffer_acc")
    node_accumulator = var_name_gen("_pt_node_acc")
    cgen_mapper = NumpyCodegenMapper(
        graph_name=graph_name,
        buffer_accumulator=buffer_accumulator,
        node_accumulator=node_accumulator,
        numpy_backend=target.numpy_like_module_name_shorthand,
        vng=var_name_gen,
        bound_arguments=preproc_result.bound_arguments,
        extra_preambles=extra_preambles,
        dot_graph_path=dot_graph_path)
    # type-ignore-reason: https://github.com/inducer/pytato/issues/236
    result_var = cgen_mapper(preproc_result.outputs)
    lines_graph_exec: List[Any] = []
    lines_graph_exec.append(
        ast.Module(
                    body=[
                        ast.Assign(targets=[
                            ast.Name(id=graph_name, ctx=ast.Store())
                            ],
                            value=ast.Call(
                                func=ast.Attribute(
                                    ast.Name(
                                        cgen_mapper.numpy_backend, ctx=ast.Load()),
                                    "Graph"),
                                args=[],
                                keywords=[]))
                        ], type_ignores=[]))
    lines_graph_exec.append(
        ast.Module(
                body=[
                    ast.Assign(targets=[
                        ast.Name(id=buffer_accumulator, ctx=ast.Store())
                        ],
                        value=ast.Dict(keys=[], values=[]))
                ], type_ignores=[]
        )
    )
    lines_graph_exec.append(
        ast.Module(
                body=[
                    ast.Assign(targets=[
                        ast.Name(id=node_accumulator, ctx=ast.Store())
                        ],
                        value=ast.Dict(keys=[], values=[]))
                ], type_ignores=[]
        )
    )
    lines_graph_exec += cgen_mapper.lines_graph_exec
    lines_graph_exec.append(ast.Return(
                            ast.Tuple(elts=[
                                ast.Call(
                                    ast.Attribute(ast.Name(graph_name),
                                        "get_exec_graph"),
                                    args=[],
                                    keywords=[]),
                                ast.Name(graph_name),
                                ast.Name(node_accumulator),
                                ast.Name(buffer_accumulator)])))

    lines_pt_kernel: List[Any] = []
    lines_pt_kernel += cgen_mapper.lines_output_allocation
    g_exec = cgen_mapper.vng("_pt_exec_g")
    lines_pt_kernel.append(
        ast.Assign(targets=[ast.Tuple(
            elts=[ast.Name(g_exec),
                  ast.Name(graph_name),
                  ast.Name(node_accumulator),
                  ast.Name(buffer_accumulator)
                  ])],
                   value=ast.Call(ast.Name("exec_graph_builder"),
                                  args=[],
                                  keywords=[]
                                  )))
    keys = []
    values = []
    for kernel_node, kernel_args in cgen_mapper.kernel_to_kernel_args.items():
        keys.append(
            ast.Subscript(
                ast.Name(node_accumulator),
                ast.Constant(kernel_node)))
        values.append(
            ast.Call(func=ast.Name("_pt_drv.KernelNodeParams"),
                     args=[],
                     keywords=[ast.keyword("args",
                                           ast.List(elts=[cgen_mapper.temp_input_arrays_to_output_stmt_mapper[array_name] if array_name in cgen_mapper.temp_input_arrays_to_output_stmt_mapper else ast.Subscript(ast.Name(buffer_accumulator), ast.Constant(array_name)) for array_name in kernel_args]))])) # noqa : 501
    lines_pt_kernel.append(
        ast.Expr(
            ast.Call(
                ast.Attribute(ast.Name(g_exec), "batched_set_kernel_node_arguments"),
                args=[ast.Dict(
                    keys=keys,
                    values=values)],
                keywords=[])))
    lines_pt_kernel.append(ast.Expr(
                            ast.Call(ast.Attribute(
                                ast.Name(g_exec), "launch"),
                                args=[],
                                keywords=[])))
    if bool(cgen_mapper.dict_line):
        lines_pt_kernel += cgen_mapper.dict_line
    if dict_flag:
        lines_pt_kernel.append(ast.Return(ast.Name(result_var.buffer_name)))
    else:
        lines_pt_kernel.append(
            ast.Return(
                ast.Call(
                    ast.Attribute(
                        ast.Subscript(
                            ast.Name(result_var.buffer_name),
                            ast.Constant("_pt_out")), "get"), args=[], keywords=[])))
    module = ast.Module(
        body=[ast.Import(names=[ast.alias(name=target.numpy_like_module_name,
                                          asname=(
                                              target
                                              .numpy_like_module_name_shorthand
                                          ))]),
              ast.Import(names=[ast.alias(name="numpy", asname="np")]),
              *tuple(extra_preambles),
              ast.FunctionDef(
                  name="exec_graph_builder",
                  posonlyargs=[],
                  args=ast.arguments(
                      args=[],
                      posonlyargs=[],
                      kwonlyargs=[],
                      kw_defaults=[],
                      defaults=[]),
                  body=lines_graph_exec,
                  decorator_list=[ast.Name("cache")]),
              ast.FunctionDef(
                  name=function_name,
                  posonlyargs=[],
                  args=ast.arguments(
                      args=[ast.arg(arg="allocator"), ast.arg("dev")],
                      posonlyargs=[],
                      kwonlyargs=[ast.arg(arg=name)
                                  for name in list(cgen_mapper.input_args.keys())],
                      kw_defaults=[
                          None for _ in list(cgen_mapper.input_args.keys())
                          ],
                      defaults=[ast.Name("cuda_allocator"), ast.Name("cuda_dev")]),
                  body=lines_pt_kernel,
                  decorator_list=[])
              ],
        type_ignores=[])

    program = ast.unparse(ast.fix_missing_locations(module))

    if show_code:
        if colorize_show_code is None:
            colorize_show_code = _get_default_colorize_code()
        assert isinstance(colorize_show_code, bool)

        if _can_colorize_output() and colorize_show_code:
            from pygments import highlight
            from pygments.lexers import PythonLexer
            from pygments.formatters import TerminalTrueColorFormatter
            print(highlight(program,
                            formatter=TerminalTrueColorFormatter(),
                            lexer=PythonLexer()))
        else:
            print(program)
    return target.bind_program(
        program,
        function_name,
        expected_arguments=frozenset(list(cgen_mapper.input_args.keys())),
        bound_arguments=pmap(cgen_mapper.bound_arguments))

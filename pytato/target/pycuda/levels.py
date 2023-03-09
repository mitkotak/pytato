import ast
import sys
import os
import numpy as np
import loopy as lp

from typing import (Callable, Union, Optional, Mapping, Dict, TypeVar,
                    List, Set, Tuple, Any)

import pymbolic.primitives as prim
from pytools import (UniqueNameGenerator, memoize_method)
from pytools.graph import reverse_graph
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


class StringMapper(CachedMapper[ArrayOrNames]):
    
    def __init__(self):
        super().__init__()
        from bidict import bidict
        self.string_map = bidict()
        self.dep_map = {}
        self.vng = UniqueNameGenerator()

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays):
        from pytato.codegen import normalize_outputs as normalize_outputs
        from pytato.target.loopy.codegen import preprocess as preprocess_loopy
        result = normalize_outputs(expr)
        preproc_result = preprocess_loopy(result, target=lp.CudaTarget())
        for _, subexpr in list(preproc_result.outputs._data.items()):
            if not isinstance(subexpr, Placeholder):
                expression = self.rec(subexpr)
                self.string_map[subexpr] = expression
    

    def map_placeholder(self, expr: Placeholder):
        placeholder = self.vng("_pt_N")
        return placeholder
    
    def map_index_lambda(self, expr: IndexLambda):
        operation = self.vng("_pt_N")
        self.string_map[expr] = self.vng("_pt_N")
        for _, bnd in list(expr.bindings.items()):
            if not isinstance(bnd, Placeholder):
                index_lambda = self.rec(bnd)
                self.string_map[bnd] = index_lambda
                if not index_lambda in self.dep_map:
                    self.dep_map[index_lambda] = set()
                self.dep_map[index_lambda].add(operation)
        return operation

    def topological_sort(self):
        graph_predecessor = reverse_graph(self.dep_map)
        graph_successor = reverse_graph(graph_predecessor)
        levels = []
        levels_ops = []
        while (len(graph_predecessor)) > 0:
            level_nodes = []
            level_nodes_ops = []
            for key,values in list(graph_predecessor.items()):
                if len(values) == 0:
                    level_nodes.append(key)
                    if key in self.string_map.inverse:
                        level_nodes_ops.append(self.string_map.inverse[key])
                    nodes_to_remove = graph_successor[key]
                    for node in nodes_to_remove:
                        if key in graph_predecessor[node]:
                            tmp_set = set(graph_predecessor[node])
                            tmp_set.remove(key)
                            graph_predecessor[node] = frozenset(tmp_set)
                    del graph_predecessor[key]
            levels.append(len(level_nodes))
            levels_ops.append(level_nodes_ops)
        return levels, levels_ops
    
def weight_calculator(levels_ops):
    levels_weights = np.zeros(shape=(len(levels_ops)))
    for level_i,level_op in enumerate(levels_ops):
        level_bytes = 0
        for node in level_op:
            assert isinstance(node, IndexLambda)
            level_bytes += node.size * np.dtype(node.dtype).itemsize
            for key,bnd in node.bindings.items():
                level_bytes += bnd.size * np.dtype(bnd.dtype).itemsize
        levels_weights[level_i] = level_bytes
    return levels_weights/sum(levels_weights)
            

from . import base
from typing import Tuple, Union
import uuid
from sympy import symbols, simplify, Symbol
from enum import Enum
from functools import reduce
from collections import namedtuple
import torch


class OpBase:
    def __init__(self, name, **kwargs):
        self.name = name
        self.config = kwargs

    def apply(self, input, name=""):
        raise NotImplementedError

    def getName(self, name=""):
        if name == "":
            name = f"_{uuid.uuid4().hex[:8]}"
        return name

    def applyList(self, inputs, names=[]):
        if len(names) < len(inputs):
            names += [self.getName() for _ in range(len(inputs) - len(names))]
        elif len(names) > len(inputs):
            raise ValueError(
                "Number of names must be less than or equal to the number of inputs"
            )
        return [self.apply(i, n) for i, n in zip(inputs, names)]

class Map(OpBase):
    def __init__(self, **kwargs):
        super().__init__("Map", **kwargs)

    def apply(self, input: base.Stream, name=""):
        ###
        # Put Fn in the config to simplify the applyList of Map. This adds another generic type to the Map definition.
        ###
        fn: base.Fn = self.config["fn"]
        assert isinstance(fn, base.Fn), f"Map should take one of provided fns as input, but get {type(fn)}"
        assert fn.input == input.dtype, f"Map should take {fn.input} as input, but get {input.dtype}"

        result = base.Stream(
            self.getName(name), fn.output, input.rank, input.shape
        )

        if input.data is not None:
            result.data = fn.apply(input.data)
            result.ctx = input.ctx
        return result

class Repeat(OpBase):
    def __init__(self, **kwargs):
        super().__init__("Repeat", **kwargs)
    
    def apply(self, input: base.Stream, name=""):
        new_size = self.config["n"]
        new_shape = [new_size] + input.shape
        result = base.Stream(self.getName(name), input.dtype, input.rank + 1, new_shape)
        if input.data is not None:
            result.data = input.data
            result.ctx = input.ctx
            def data_fn(shape, dtype, data, symbols):
                new_shape = base.subsFullShape(dtype, shape, symbols)[::-1]
                return data.unsqueeze(result.rank).expand(new_shape).contiguous()
            result.update(data_fn)
        return result

class Bufferize(OpBase):
    def __init__(self, **kwargs):
        super().__init__("Bufferize", **kwargs)

    def apply(self, input: base.Stream, name=""):
        buf_rank = self.config["a"]
        assert buf_rank <= input.rank, f"Bufferize rank should be less than or equal to input rank, but get buf_rank: {buf_rank} and input rank: {input.rank}"
        assert buf_rank > 0, f"Bufferize rank should be greater than 0, but get buf_rank: {buf_rank}"

        new_shape = input.shape[buf_rank:]
        buf_shape = input.shape[:buf_rank]
        if isinstance(input.dtype, base.Buffer):
            new_buf = base.Buffer(input.dtype.dtype, input.dtype.shape + buf_shape)
        else:
            new_buf = base.Buffer(input.dtype, buf_shape)
        result = base.Stream(
            self.getName(name), new_buf, input.rank - buf_rank, new_shape
        )
        if input.data is not None:
            result.data = input.data
            result.ctx = input.ctx
        return result

def get_full_indices(shape):
    total_elements = reduce(lambda x, y: x * y, shape)
    indices = []
    # For example, shape [2,3,4]
    # Output: [(0,0,0), (0,0,1), (0,0,2), (0,0,3), (0,1,0), (0,1,1), (0,1,2), (0,1,3), ..., (1,0,0), (1,0,1), (1,0,2), (1,0,3), ...]
    for i in range(total_elements):
        idx = []
        for j in range(len(shape)):
            idx.append(i % shape[j])
            i //= shape[j]
        indices.append(tuple(idx[::-1]))
    return indices

class Accum(OpBase):
    def __init__(self, **kwargs):
        super().__init__("Accum", **kwargs)

    def apply(self, input: base.Stream, name=""):
        b = self.config["b"]
        fn: base.Fn = self.config["fn"]
        assert isinstance(fn, base.Fn), f"Accum should take one of provided fns as input, but get {type(fn)}"
        assert fn.input == input.dtype, f"Accum should take {fn.input} as input, but get {input.dtype}"
        assert b > 0 and b <= input.rank, f"Accum should take a positive integer b less than or equal to the rank of the input, but get b: {b} and input rank: {input.rank}"

        result = base.Stream(
            self.getName(name), fn.output, input.rank - b, input.shape[b:]
        )
        if input.data is not None:
            result.ctx = input.ctx
            # TODO: Construct a general application function here
            output_indices = get_full_indices(base.subsOuterShape(result.shape, result.ctx))
            input_indices = get_full_indices(base.subsOuterShape(input.shape[:b], input.ctx))
            if isinstance(result.dtype, base.Scalar) or isinstance(result.dtype, base.Buffer):
                output_shapes = [base.subsFullShape(result.dtype, result.shape, result.ctx)[::-1]]
            elif isinstance(result.dtype, base.STuple):
                output_shapes = [base.subsFullShape(r, result.shape, result.ctx)[::-1] for r in result.dtype]
            else:
                raise ValueError("Invalid dtype")
            result.data = [torch.zeros(shape) for shape in output_shapes]
            for idx in output_indices:
                state = fn.getInit()
                for i in input_indices:
                    full_idx = idx + i
                    partial_data = [d[full_idx + (...,)] for d in input.data]
                    state = fn.apply(state, partial_data)
                for n, s in enumerate(state):
                    result.data[n][idx] = s
        return result

class Streamify(OpBase):
    def __init__(self, **kwargs):
        ###
        # This "Streamify" is FnBlock(Streamify)
        ###
        super().__init__("Streamify", **kwargs)

    def apply(self, input: base.Stream, name=""):
        idtype = input.dtype
        result = base.Stream(
            self.getName(name), idtype.dtype, input.rank + idtype.rank, idtype.shape + input.shape
        )
        if input.data is not None:
            result.data = input.data
            result.ctx = input.ctx
        return result


class Promote(OpBase):
    def __init__(self, **kwargs):
        super().__init__("Promote", **kwargs)

    def apply(self, input: base.Stream, name=""):
        a = input.rank
        b = self.config["b"]
        assert 0 <= b and b <= a + 1, f"Promote should take a non-negative integer b less than or equal to the rank of the input plus one, but get b: {b} and input rank: {a}"
        ishape = input.shape
        result = base.Stream(
            self.getName(name), input.dtype, a + 1, ishape[:b] + [1] + ishape[b:]
        )
        if input.data is not None:
            result.data = input.data
            result.ctx = input.ctx
            def data_fn(shape, dtype, data, symbols):
                return data.unsqueeze(result.rank - b).contiguous()
            result.update(data_fn)
        return result


class Zip(OpBase):
    def __init__(self, **kwargs):
        super().__init__("Zip", **kwargs)

    def apply(self, input: Tuple[base.Stream, base.Stream], name=""):
        ls = input[0]
        rs = input[1]
        assert ls.rank == rs.rank, f"Zip should take two streams with the same rank, but get first rank: {ls.rank} and second rank: {rs.rank}"
        assert ls.shape == rs.shape, f"Zip should take two streams with the same shape, but get first shape: {ls.shape} and second shape: {rs.shape}"
        if isinstance(ls.dtype, base.STuple) and isinstance(rs.dtype, base.STuple):
            new_dtype = base.STuple((*ls.dtype, *rs.dtype))
        elif isinstance(ls.dtype, base.STuple) and not isinstance(rs.dtype, base.STuple):
            new_dtype = base.STuple((*ls.dtype, rs.dtype))
        elif not isinstance(ls.dtype, base.STuple) and isinstance(rs.dtype, base.STuple):
            new_dtype = base.STuple((ls.dtype, *rs.dtype))
        else:
            new_dtype = base.STuple((ls.dtype, rs.dtype))

        result = base.Stream(
            self.getName(name), new_dtype, ls.rank, ls.shape
        )

        if ls.data is not None and rs.data is not None:
            result.ctx = ls.ctx | rs.ctx
            if isinstance(ls.data, list) and isinstance(rs.data, list):
                result.data = ls.data + rs.data
            elif isinstance(ls.data, list) and not isinstance(rs.data, list):
                result.data = ls.data + [rs.data,]
            elif not isinstance(ls.data, list) and isinstance(rs.data, list):
                result.data = [ls.data,] + rs.data
            else:
                result.data = [ls.data, rs.data]
        
        return result


class Flatten(OpBase):
    def __init__(self, **kwargs):
        super().__init__("Flatten", **kwargs)

    def apply(self, input: base.Stream, name=""):
        L = self.config["L"]
        ishape = input.shape
        new_shape = ishape[: (L[0] - 1)]
        reduce_shape = ishape[L[0] - 1]
        for i in L:
            reduce_shape *= ishape[i]
        new_shape.append(reduce_shape)
        new_shape += ishape[(L[-1] + 1) :]

        result = base.Stream(self.getName(name), input.dtype, input.rank - len(L), new_shape)
        if input.data is not None:
            result.data = input.data
            result.ctx = input.ctx
            def data_fn(shape, dtype, data, symbols):
                new_shape = base.subsFullShape(dtype, shape, symbols)[::-1]
                return data.view(*new_shape).contiguous()
            result.update(data_fn)
        return result

class Copy(OpBase):
    def __init__(self, **kwargs):
        super().__init__("Copy", **kwargs)

    def apply(self, input: base.Stream, name=""):
        l = base.Stream(self.getName(name), input.dtype, input.rank, input.shape)
        r = base.Stream(self.getName(name), input.dtype, input.rank, input.shape)
        if input.data is not None:
            l.data = input.data
            l.ctx = input.ctx
            r.data = input.data
            r.ctx = input.ctx
        return l, r

class Scan(OpBase):
    def __init__(self, **kwargs):
        super().__init__("Scan", **kwargs)

    def apply(self, input: base.Stream, name=""):
        b = self.config["b"]
        fn: base.Fn = self.config["fn"]
        assert isinstance(fn, base.Fn), f"Scan should take one of provided fns as input, but get {type(fn)}"
        assert 0 < b and b <= input.rank, f"Scan should take a positive integer b less than or equal to the rank of the input, but get b: {b} and input rank: {input.rank}"
        assert fn.input == input.dtype, f"Scan shoud take {fn.input} as input, but get {input.dtype}"

        result = base.Stream(
            self.getName(name), fn.output, input.rank, input.shape
        )

        if input.data is not None:
            result.ctx = input.ctx
            # 1. Check the prefix shape
            output_indices = get_full_indices(base.subsOuterShape(input.shape[b:], result.ctx))
            input_indices = get_full_indices(base.subsOuterShape(input.shape[:b], input.ctx))
            if isinstance(result.dtype, base.Scalar) or isinstance(result.dtype, base.Buffer):
                output_shapes = [base.subsFullShape(result.dtype, result.shape, result.ctx)[::-1]]
            elif isinstance(result.dtype, base.STuple):
                output_shapes = [base.subsFullShape(r, result.shape, result.ctx)[::-1] for r in result.dtype]
            else:
                raise ValueError("Invalid dtype")
            result.data = [torch.zeros(shape) for shape in output_shapes]
            for idx in output_indices:
                state = fn.getInit()
                for i in input_indices:
                    full_idx = idx + i
                    partial_data = [d[full_idx + (...,)] for d in input.data]
                    state = fn.apply(state, partial_data)
                    for n, s in enumerate(state):
                        result.data[n][full_idx] = s
        return result
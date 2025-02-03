from . import base
from typing import Tuple
import uuid
from sympy import symbols
from functools import reduce
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
        if isinstance(input, tuple):
            # apply Zip until we have a single stream
            while len(input) > 1:
                new_single = Zip().apply((input[0], input[1]))
                input = (new_single,) + input[2:]
            input = input[0]
        fn: base.Fn = self.config["fn"]
        assert isinstance(fn, base.Fn), f"Map should take one of provided fns as input, but get {type(fn)}"
        assert fn.input == input.dtype, f"Map should take {fn.input} as input, but get {input.dtype}"

        result = base.Stream(
            self.getName(name), fn.output, input.rank, input.shape
        )

        if input.data is not None:
            # result.data = fn.apply(input.data)
            shape_value = base.subsOuterShape(input.shape, input.ctx)
            indices = get_full_indices(shape_value)
            records = {}
            for idx in indices:
                partial_data = [d[idx + (...,)] for d in input.data]
                result.data = fn.apply(partial_data)
                for n, s in enumerate(result.data):
                    if n not in records:
                        records[n] = [s]
                    else:
                        records[n].append(s)
            # result.data = [torch.stack(records[n]).view(shape_value) for n in range(len(records))]
            result.data = []
            outer_data_shape = shape_value[::-1]
            for n in range(len(records)):
                record_data = records[n]
                record_data_shape = record_data[0].shape
                result_data = torch.stack(record_data).view(tuple(outer_data_shape) + record_data_shape)
                result.data.append(result_data)
            result.ctx = input.ctx
        return result


class Reshape(OpBase):
    def __init__(self, **kwargs):
        super().__init__("Reshape", **kwargs)

    def apply(self, input: base.Stream, name=""):
        new_shape = []
        for i, s in enumerate(input.shape):
            if i in self.config["L"]:
                new_shape.append(self.config["S"])
                new_shape.append(s // self.config["S"])
            else:
                new_shape.append(s)

        result = base.Stream(self.getName(name), input.dtype, input.rank, new_shape)
        if input.data is not None:
            # view as the reverse of new_shape
            result.data = input.data
            result.ctx = input.ctx
            def data_fn(shape, dtype, data, symbols):
                new_shape = base.subsFullShape(dtype, shape, symbols)[::-1]
                return data.view(*new_shape).contiguous()
            result.update(data_fn)
        return result


class RepeatRef(OpBase):
    def __init__(self, **kwargs):
        super().__init__("RepeatRef", **kwargs)

    def apply(self, input: Tuple[base.Stream, base.Stream], name=""):
        data_stream = input[0]
        ref_stream = input[1]
        assert ref_stream.rank == data_stream.rank + 1, f"RepeatRef reference stream's rank should be data stream's rank + 1, but get ref_stream rank: {ref_stream.rank} and data_stream rank: {data_stream.rank}"
        assert ref_stream.shape[1:] == data_stream.shape, f"RepearRef reference stream's shape should be data stream's shape with an additional leftmost dimension, but get ref_stream shape: {ref_stream.shape} and data_stream shape: {data_stream.shape}"

        new_shape = ref_stream.shape
        result = base.Stream(
            self.getName(name), data_stream.dtype, data_stream.rank + 1, new_shape
        )
        if data_stream.data is not None:
            result.data = data_stream.data
            result.ctx = data_stream.ctx
            def data_fn(shape, dtype, data, symbols):
                new_shape = base.subsFullShape(dtype, shape, symbols)[::-1]
                return data.unsqueeze(result.rank).expand(new_shape).contiguous()
            result.update(data_fn)
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

RepeatLast = Repeat

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
    if len(shape) == 0:
        return [()]
    total_elements = reduce(lambda x, y: x * y, shape)
    indices = []
    # For example, shape [2,3,4]
    # Output: [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1), (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1), (2, 2, 0), (2, 2, 1), (3, 0, 0), (3, 0, 1), (3, 1, 0), (3, 1, 1), (3, 2, 0), (3, 2, 1)]
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
            if isinstance(result.dtype, base.Element) or isinstance(result.dtype, base.Buffer):
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
                new_shape = base.subsFlattenFullShape(dtype, shape, symbols)[::-1]
                return data.view(*new_shape).contiguous()
            result.update(data_fn)
        return result

class Identity(OpBase):
    def __init__(self, **kwargs):
        super().__init__("Identity", **kwargs)

    def apply(self, input: base.Stream, name=""):
        return base.Stream(self.getName(name), input.dtype, input.rank, input.shape)

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
            if isinstance(result.dtype, base.Element) or isinstance(result.dtype, base.Buffer):
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

class Merge(OpBase):
    def __init__(self, **kwargs):
        super().__init__("Merge", **kwargs)
    
    def apply(self, input: Tuple[list[base.Stream], base.Stream], name=""):
        data_streams = input[0]
        idx_stream = input[1]
        fn: base.Fn = self.config["fn"]
        a = data_streams[0].rank
        b = idx_stream.rank
        A_dtype = data_streams[0].dtype
        common_shape = data_streams[0].shape[:-1]
        data_stream_ctx = data_streams[0].ctx
        common_indices = get_full_indices(base.subsOuterShape(common_shape, data_stream_ctx))
        assert all([s.dtype == A_dtype for s in data_streams]), f"Merge should take data streams with the same dtype, but get different dtypes"
        assert all([s.rank == a for s in data_streams]), f"Merge should take data streams with the same rank, but get different ranks"
        assert isinstance(fn, base.Fn), f"Merge should take one of provided fns as input, but get {type(fn)}"
        assert fn.input == A_dtype, f"Merge shoud take {fn.input} as input, but get {A_dtype}"
        # assert isinstance(idx_stream.dtype, base.Buffer) and idx_stream.dtype.rank == 1, f"Merge should take an index stream with rank-1 Buffer dtype, but get {idx_stream.dtype}"
        assert all([s.shape[:-1] == common_shape for s in data_streams]), f"Merge should take data streams with the same shape except the last dimension, but get different shapes"

        result = base.Stream(
            self.getName(name), fn.output, a + b, common_shape + idx_stream.shape
        )
        result.ctx = data_stream_ctx | idx_stream.ctx

        if idx_stream.data is not None:
            # Generate correct index
            # 1. Store the current outermost index for each data stream
            cur_outermost_idx = [0 for _ in range(len(data_streams))]
            # 2. Iterate over the index stream's shape (The dtype is guaranteed to be Buffer with rank 1)
            index_shape_value = base.subsOuterShape(idx_stream.shape, idx_stream.ctx)
            index_indices = get_full_indices(index_shape_value)

            if isinstance(result.dtype, base.Element) or isinstance(result.dtype, base.Buffer):
                output_shapes = [base.subsFullShape(result.dtype, result.shape, result.ctx)[::-1]]
            elif isinstance(result.dtype, base.STuple):
                output_shapes = [base.subsFullShape(r, result.shape, result.ctx)[::-1] for r in result.dtype]
            else:
                raise ValueError("Invalid dtype")
            result.data = [torch.zeros(shape) for shape in output_shapes]
            for idx in index_indices:
                # 3. Get multi-hot index
                multi_hot_idx = idx_stream.data[0][idx]
                # 4. If the multi-hot index is all zeros, skip
                if torch.sum(multi_hot_idx) == 0:
                    continue
                # 5. Find the data stream that has the current outermost index
                partial_data_stream_ids = torch.nonzero(multi_hot_idx).flatten().tolist()
                partial_data_list = [data_streams[i].data for i in partial_data_stream_ids]
                partial_outermost_idx = [cur_outermost_idx[i] for i in partial_data_stream_ids]
                # 6. Accum the chosen data streams at the innermost level
                for i in common_indices:
                    state = fn.getInit()
                    for (outermost_idx, partial_data) in zip(partial_outermost_idx, partial_data_list):
                        full_idx = (outermost_idx,) + i
                        partial_data = [d[full_idx + (...,)] for d in partial_data]
                        state = fn.apply(state, partial_data)
                    for n, s in enumerate(state):
                        result.data[n][idx + i] = s
                # 7. Update the current outermost index
                for i, p in zip(partial_data_stream_ids, partial_outermost_idx):
                    cur_outermost_idx[i] = p + 1
        return result

class Partition(OpBase):
    def __init__(self, **kwargs):
        super().__init__("Partition", **kwargs)    

    def apply(self, input: Tuple[base.Stream, base.Stream], name=""):
        data_stream = input[0]
        idx_stream = input[1]
        out_rank = data_stream.rank - idx_stream.rank
        base_shape = data_stream.shape[:out_rank]
        num_experts = self.config["N"]
        prefix = self.getName(name)
        new_names = symbols(f"{prefix}_0:{num_experts}")
        data_list = [[] for _ in range(num_experts)]
        indices = get_full_indices(base.subsOuterShape(idx_stream.shape, idx_stream.ctx))
        # 1. Scatter the data to data_list in AoS
        for idx in indices:
            multi_hot_idx = idx_stream.data[0][idx]
            partial_data = [d[idx + (...,)] for d in data_stream.data]
            for i, m in enumerate(multi_hot_idx):
                if m == 1:
                    data_list[i].append(partial_data)
        # 2. Convert data_list to SoA
        final_data = []
        new_ctx = {}
        for i in range(num_experts):
            if len(data_list[i]) == 0:
                final_data.append(None)
                new_ctx[new_names[i]] = 0
                continue
            cur_data_list = data_list[i]
            data_dict = {}
            num_fields = len(cur_data_list[0])
            for j in range(num_fields):
                data_dict[j] = [d[j] for d in cur_data_list]
            cur_data = [torch.stack(data_dict[j]) for j in range(num_fields)]
            final_data.append(cur_data)
            new_ctx[new_names[i]] = len(cur_data_list)

        results = [
            base.Stream(
                f"{prefix}_{i}",
                data_stream.dtype,
                out_rank,
                base_shape + [new_names[i]],
            ) for i in range(num_experts)
        ]
        for i, r in enumerate(results):
            r.data = final_data[i]
            r.ctx = {**data_stream.ctx, **new_ctx}
        
        return results
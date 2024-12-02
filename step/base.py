from sympy import Symbol
from typing import Union, Tuple, Optional
import torch

def subsFullShape(dtype, shape, symbols):
    if isinstance(dtype, Buffer):
        assert isinstance(dtype.dtype, Scalar)
        shape = dtype.shape + shape
    else:
        shape = shape
    return [s.subs(symbols) if not isinstance(s, int) else s for s in shape]

def subsOuterShape(shape, symbols):
    return [s.subs(symbols) if not isinstance(s, int) else s for s in shape]

class Element:
    def __init__(self, dtype: str):
        self.dtype = dtype

class Scalar(Element):
    def __init__(self, dtype: str):
        super().__init__(dtype)

    def __str__(self):
        return f"Scalar: {self.dtype}"
    
    def __eq__(self, other):
        if not isinstance(other, Scalar):
            return False
        return self.dtype == other.dtype

class Tile(Element):
    def __init__(self, dtype, shape: list[Symbol]):
        super().__init__(dtype)
        self.shape = shape
        self.rank = len(shape)
    
    def __str__(self):
        return f"Tile: {self.dtype}, [{', '.join([str(s) for s in self.shape])}]"
    
    def __eq__(self, other):
        if not isinstance(other, Tile):
            return False
        return self.dtype == other.dtype and self.shape == other.shape

class Buffer:
    def __init__(self, dtype, shape: list[Symbol]):
        self.dtype = dtype
        self.shape = shape
        self.rank = len(shape)

    def __str__(self):
        return f"Buffer: ([{', '.join([str(s) for s in self.shape])}], {self.dtype}, {self.rank})"
    
    def __eq__(self, other):
        if not isinstance(other, Buffer):
            return False
        return self.dtype == other.dtype and self.rank == other.rank and self.shape == other.shape


class STuple:
    def __init__(self, dtype: Tuple[Union[Element, Buffer], Union[Element, Buffer]]):
        self.dtype = dtype

    def __str__(self):
        result = []
        for d in self.dtype:
            result.append(str(d))
        return "STuple: (" + ",".join(result) + ")"

    def __eq__(self, other):
        if not isinstance(other, STuple):
            return False
        
        if len(self.dtype) != len(other.dtype):
            return False
        
        for (d1, d2) in zip(self.dtype, other.dtype):
            if d1 != d2:
                return False
        return True

    def __iter__(self):
        return iter(self.dtype)
    
    def __getitem__(self, key):
        return self.dtype[key]


class Stream:
    def __init__(
        self, name, dtype: Union[Element, Buffer, STuple], rank, shape: list[Symbol]
    ):
        self.name = name
        self.dtype = dtype
        # Shape is in the opposite order of the Python Array API: [innermost, ..., outermost]
        self.shape = shape
        self.rank = rank
        self.data: Optional[list[torch.Tensor]] = None
        self.ctx = {}

    def __str__(self):
        return f"Stream: {self.name}, {self.dtype}, {self.rank}, [{', '.join([str(s) for s in self.shape])}]"

    def setData(self, data: list[torch.Tensor]):
        self.data = data
    
    def addCtx(self, symbol, value):
        self.ctx[symbol] = value

    def update(self, data_fn):
        new_data = []
        dtype_list = [self.dtype] if not isinstance(self.dtype, STuple) else self.dtype
        for (dtype, data) in zip(dtype_list, self.data):
            new_data.append(data_fn(self.shape, dtype, data, self.ctx))
        self.data = new_data

class Fn:
    def __init__(
        self,
        name,
        input: Union[Element, Buffer, STuple],
        output: Union[Element, Buffer, STuple],
    ):
        self.name = name
        self.input = input
        self.output = output

    def __str__(self):
        return f"Fn: {self.name}, {self.input} -> {self.output}"

    def apply(self, *args):
        pass

    def getInit(self):
        pass
#|code-fold: true
#|code-summary: "Import utilities and define `Matrix` (click to show/hide)"

from Benchmark import Benchmark
from DType import DType
from Intrinsics import strided_load
from List import VariadicList
from Math import div_ceil, min
from Memory import memset_zero
from Object import object, Attr
from Pointer import DTypePointer
from Random import rand, random_f64
from TargetInfo import dtype_sizeof, dtype_simd_width

struct Matrix:
    var data: DTypePointer[DType.f32]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[DType.f32].alloc(rows * cols)
        rand(self.data, rows*cols)
        self.rows = rows
        self.cols = cols

    fn __del__(owned self):
        self.data.free()

    fn zero(inout self):
        memset_zero(self.data, self.rows * self.cols)

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> F32:
        return self.load[1](y, x)

    @always_inline
    fn load[nelts:Int](self, y: Int, x: Int) -> SIMD[DType.f32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: F32):
        return self.store[1](y, x, val)

    @always_inline
    fn store[nelts:Int](self, y: Int, x: Int, val: SIMD[DType.f32, nelts]):
        self.data.simd_store[nelts](y * self.cols + x, val)

fn matmul_naive(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]

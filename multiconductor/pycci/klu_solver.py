"""
Ctypes wrapper for KLUSolve.dll — a sparse complex matrix solver based on KLU.

Provides a KLUSolver class that mimics the scipy.sparse.linalg.splu interface,
accepting a scipy.sparse.csc_matrix and exposing a .solve(rhs) method.

Performance notes:
  - The DLL is loaded once at module import and stays in memory.
  - solve() uses zero-copy numpy↔ctypes conversion (no Python loops).
  - Matrix construction uses vectorised numpy filtering for the upper triangle.
"""

import ctypes
import os
import numpy as np
from scipy.sparse import csc_matrix

# ── Locate and load KLUSolve.dll ──────────────────────────────────────────────

_DLL_SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", ".."),   # project root
    os.path.dirname(__file__),                              # pycci directory
    os.path.join(os.path.dirname(__file__), ".."),          # multiconductor directory
]

_klu_dll = None
for _dir in _DLL_SEARCH_PATHS:
    _path = os.path.join(os.path.abspath(_dir), "KLUSolve.dll")
    if os.path.isfile(_path):
        _klu_dll = ctypes.CDLL(_path)
        break

if _klu_dll is None:
    raise FileNotFoundError(
        "KLUSolve.dll not found. Searched: " +
        ", ".join(os.path.abspath(d) for d in _DLL_SEARCH_PATHS)
    )


# ── Complex type for KLUSolve (real, imag pair of doubles) ────────────────────

class _KLUComplex(ctypes.Structure):
    _fields_ = [("re", ctypes.c_double), ("im", ctypes.c_double)]

_pKLUComplex = ctypes.POINTER(_KLUComplex)


# ── Declare function signatures ───────────────────────────────────────────────

_klu_dll.NewSparseSet.argtypes = [ctypes.c_uint32]
_klu_dll.NewSparseSet.restype = ctypes.c_size_t

_klu_dll.ZeroSparseSet.argtypes = [ctypes.c_size_t]
_klu_dll.ZeroSparseSet.restype = ctypes.c_uint32

_klu_dll.AddMatrixElement.argtypes = [
    ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32, _pKLUComplex,
]
_klu_dll.AddMatrixElement.restype = ctypes.c_uint32

_klu_dll.FactorSparseMatrix.argtypes = [ctypes.c_size_t]
_klu_dll.FactorSparseMatrix.restype = ctypes.c_uint32

_klu_dll.SolveSparseSet.argtypes = [ctypes.c_size_t, _pKLUComplex, _pKLUComplex]
_klu_dll.SolveSparseSet.restype = ctypes.c_uint32

# FUNCTION AddPrimitiveMatrix(id: NativeUInt; nOrder: LongWord;
#                             Nodes: pLongWordArray; Mat: pComplexArray): LongWord
_klu_dll.AddPrimitiveMatrix.argtypes = [
    ctypes.c_size_t, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32), _pKLUComplex,
]
_klu_dll.AddPrimitiveMatrix.restype = ctypes.c_uint32

_klu_dll.DeleteSparseSet.argtypes = [ctypes.c_size_t]
_klu_dll.DeleteSparseSet.restype = ctypes.c_uint32


# ── Zero-copy helpers ─────────────────────────────────────────────────────────

def _np_as_klu_ptr(arr):
    """
    Return a ctypes pointer to a contiguous complex128 numpy array's data.

    numpy complex128 stores each element as two contiguous float64 values
    (real, imag) — the exact memory layout KLUSolve expects.  No copy needed.
    """
    arr = np.ascontiguousarray(arr, dtype=np.complex128).ravel()
    return arr, arr.ctypes.data_as(_pKLUComplex)


# ── KLUSolver class ───────────────────────────────────────────────────────────

class KLUSolver:
    """
    Drop-in replacement for scipy.sparse.linalg.splu for complex sparse matrices.

    Usage::

        solver = KLUSolver(csc_matrix(A))
        x = solver.solve(b)
    """

    def __init__(self, A):
        """
        Create a KLU factorization of sparse matrix *A*.

        Parameters
        ----------
        A : scipy.sparse.csc_matrix
            Square sparse matrix (symmetric).  Converted to CSC if needed.
        """
        if not isinstance(A, csc_matrix):
            A = csc_matrix(A)

        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("KLUSolver requires a square matrix")

        self._n = n
        self._handle = _klu_dll.NewSparseSet(n)
        if self._handle == 0:
            raise RuntimeError("KLUSolve: NewSparseSet failed")

        # ── Single-call matrix construction via AddPrimitiveMatrix ──
        # AddPrimitiveMatrix(id, nOrder, Nodes, Mat) adds a dense nOrder×nOrder
        # matrix in one DLL call — no Python loop over non-zeros.
        # Nodes is a 1-based index array; Mat is row-major complex data.
        A_dense = np.ascontiguousarray(A.toarray(), dtype=np.complex128)
        nodes = np.arange(1, n + 1, dtype=np.uint32)

        rc = _klu_dll.AddPrimitiveMatrix(
            self._handle,
            n,
            nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            A_dense.ctypes.data_as(_pKLUComplex),
        )
        if rc == 0:
            self._cleanup()
            raise RuntimeError("KLUSolve: AddPrimitiveMatrix failed")

        # Factor
        rc = _klu_dll.FactorSparseMatrix(self._handle)
        if rc == 0:
            self._cleanup()
            raise RuntimeError("KLUSolve: FactorSparseMatrix failed (singular?)")

        # Pre-allocate solution buffer (reused across solve calls)
        self._x_buf = np.empty(n, dtype=np.complex128)

    def solve(self, b):
        """
        Solve A x = b using the pre-computed KLU factorization.

        Uses zero-copy numpy↔ctypes conversion — no Python-level element loops.

        Parameters
        ----------
        b : numpy.ndarray
            Right-hand-side vector, shape (n,1) or (n,).

        Returns
        -------
        x : numpy.ndarray, shape (n,1), complex128
        """
        b_flat = np.ascontiguousarray(b, dtype=np.complex128).ravel()
        if len(b_flat) != self._n:
            raise ValueError(
                f"RHS size {len(b_flat)} does not match matrix size {self._n}"
            )

        # Zero-copy pointers into numpy memory
        b_ptr = b_flat.ctypes.data_as(_pKLUComplex)
        x_ptr = self._x_buf.ctypes.data_as(_pKLUComplex)

        rc = _klu_dll.SolveSparseSet(self._handle, x_ptr, b_ptr)
        if rc == 0:
            raise RuntimeError("KLUSolve: SolveSparseSet failed (invalid handle)")
        if rc == 2:
            raise RuntimeError("KLUSolve: SolveSparseSet reports singular matrix")

        return self._x_buf.reshape(-1, 1).copy()

    def _cleanup(self):
        if hasattr(self, '_handle') and self._handle and self._handle != 0:
            _klu_dll.DeleteSparseSet(self._handle)
            self._handle = 0

    def __del__(self):
        self._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._cleanup()

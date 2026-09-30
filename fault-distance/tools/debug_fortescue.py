"""
Minimal verification of the Fortescue matrix.
Synthetic data with a known result, without CSV or FFT.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from symseq.core import abc_to_seq

print("=" * 60)
print("TEST 1: Pure POSITIVE sequence")
print("  Ia = 1<0deg,  Ib = 1<-120deg,  Ic = 1<+120deg")
print("  Expected: I1=1, I2~0, I0~0")
Va = 1.0 + 0j
Vb = np.exp(-1j * 2 * np.pi / 3)
Vc = np.exp(+1j * 2 * np.pi / 3)
V0, V1, V2 = abc_to_seq(Va, Vb, Vc)
print(f"  I0 = {abs(V0):.6f}  (expected ~ 0)")
print(f"  I1 = {abs(V1):.6f}  (expected ~ 1)")
print(f"  I2 = {abs(V2):.6f}  (expected ~ 0)")

print()
print("=" * 60)
print("TEST 2: Pure NEGATIVE sequence")
print("  Ia = 1<0deg,  Ib = 1<+120deg,  Ic = 1<-120deg")
print("  Expected: I2=1, I1~0, I0~0")
Va = 1.0 + 0j
Vb = np.exp(+1j * 2 * np.pi / 3)
Vc = np.exp(-1j * 2 * np.pi / 3)
V0, V1, V2 = abc_to_seq(Va, Vb, Vc)
print(f"  I0 = {abs(V0):.6f}  (expected ~ 0)")
print(f"  I1 = {abs(V1):.6f}  (expected ~ 0)")
print(f"  I2 = {abs(V2):.6f}  (expected ~ 1)")

print()
print("=" * 60)
print("TEST 3: Pure ZERO sequence")
print("  Ia = Ib = Ic = 1<0deg")
print("  Expected: I0=1, I1~0, I2~0")
Va = Vb = Vc = 1.0 + 0j
V0, V1, V2 = abc_to_seq(Va, Vb, Vc)
print(f"  I0 = {abs(V0):.6f}  (expected ~ 1)")
print(f"  I1 = {abs(V1):.6f}  (expected ~ 0)")
print(f"  I2 = {abs(V2):.6f}  (expected ~ 0)")

print()
print("=" * 60)
print("TEST 4: Phase-to-phase fault AB (2AB)")
print("  For 2AB: Ia!=0, Ib!=0, Ic=0 -> I1~I2, I0~0")
Ia = 1.0 + 0j
Ib = np.exp(1j * np.pi)   # opposite phase to Ia
Ic = 0.0 + 0j
V0, V1, V2 = abc_to_seq(Ia, Ib, Ic)
print(f"  I0 = {abs(V0):.6f}  (expected ~ 0)")
print(f"  I1 = {abs(V1):.6f}  (expected = I2)")
print(f"  I2 = {abs(V2):.6f}  (expected = I1)")

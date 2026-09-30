# Multiconductor protection

The protection package evaluates phase-preserving switch currents from
`res_switch_sc` / `res_switch`, or from the associated line result when a switch
result is unavailable.

Supported devices are:

- `Fuse`: log-log interpolation of an explicit manufacturer curve, with a
  clearly identified generic curve available for exploratory studies.
- `DTOC`: definite-time pickup plus optional instantaneous high-set stage.
- `IDMT`: IEC standard, very, extremely, and long inverse curves.
- `IDTOC`: coordinated inverse, definite-time, and instantaneous stages; the
  fastest asserted stage operates.

```python
from multiconductor.protection import DTOC, run_protection

DTOC(
    net,
    switch_index=4,
    pickup_ka=0.4,
    delay_s=0.3,
    instantaneous_ka=2.5,
    instantaneous_delay_s=0.02,
)

# Requires current results from calc_sc(..., branch_results=True).
trip_table = run_protection(net, scenario="sc", trip_switches=True)
```

`calculate_protection_times()` is read-only with respect to switch positions.
`run_protection(..., trip_switches=True)` opens every conductor row belonging to
a tripped grouped switch. Passing `fault_spec=FaultSpec(...)` makes the runner
calculate its own phase-domain fault. With `ordered=True`, only the fastest
device operates at each step and the fault is re-solved after each topology
change until the fault is isolated, no device asserts, or `max_operations` is
reached.

The `net.protection` table stores normalized, JSON-compatible settings rather
than live Python objects. Runtime device objects are rebuilt from this table
after network serialization/deserialization.

Protection timing is deterministic steady-state coordination logic. CT
saturation, relay filtering/memory, breaker travel and interrupting duty,
reclosing, communication schemes, directional polarization, distance zones,
and transient waveform simulation are not modeled. Ordered recalculation starts
a fresh timing evaluation after each topology change; it does not accumulate
partially elapsed relay timing or thermal memory. Generic fuse curves must not be
used for equipment approval; supply the manufacturer's current/time points.

# opendss

Python package for running OpenDSS power-system analyses through `py_dss_interface`. The package mirrors the multiconductor analysis workflow — power flow, short-circuit, load allocation, hosting capacity — while adding control modules that implement the same algorithms as `multiconductor.control` on top of the OpenDSS solver.

## Package Structure

```
opendss/
├── __init__.py              # Re-exports: run_pf, calc_sc, run_load_allocation, run_hosting_capacity
├── pf/
│   └── powerflow.py         # Power flow solver wrapper
├── scd/
│   └── short_circuit.py     # Short-circuit / fault study
├── la/
│   └── load_allocation.py   # Iterative load allocation to match substation target
├── hc/
│   └── hosting_capacity.py  # DER hosting capacity (ICA) via binary search
├── control/
│   ├── __init__.py           # Re-exports all control classes and DSS-script writers
│   ├── load_tap_changer_control.py   # Discrete tap-changer (gang / phase)
│   ├── line_drop_control.py          # Line-drop compensation regulator
│   ├── shunt_controller.py           # Binary capacitor-bank switching
│   ├── volt_var_control.py           # Volt-var Q(V) curve controller
│   ├── regulator.py          # DSS script writer: RegControl from mc controllers
│   ├── capacitor.py          # DSS script writer: Capacitor / CapControl from mc shunts
│   └── regulator_control.py  # Reference pyControl implementation (Davis Montenegro)
└── tools/
    └── dss_converter.py      # Multiconductor pandapowerNet → .dss script converter
```

## Requirements

- Python ≥ 3.9
- `py_dss_interface`
- `numpy`
- `pandas`

## Quick Start

```python
import py_dss_interface
from opendss import run_pf, calc_sc, run_load_allocation, run_hosting_capacity
```

### Power Flow

```python
results = run_pf("path/to/circuit.dss")
bus_df   = results["res_bus"]
line_df  = results["res_line"]
trafo_df = results["res_trafo"]
```

`run_pf` accepts either a `.dss` file path or an already-initialised `py_dss_interface.DSS` object.

### Short-Circuit Study

```python
sc = calc_sc("circuit.dss", bus="bus_650", fault="3ph")
```

Fault types: `"3ph"`, `"1ph"` / `"lg"`, `"ll"` / `"2ph"`, `"llg"`.

### Load Allocation

```python
result = run_load_allocation("circuit.dss", target_kw=5000)
```

Iteratively scales individual load kW/kvar so the total feeder demand at the substation head matches the target.

### Hosting Capacity

```python
hc = run_hosting_capacity("circuit.dss")
hc_df = hc["res_hc"]
```

Determines the maximum DER injection at each bus before voltage, thermal, or reverse-power violations occur.

## Controls

The `opendss.control` sub-package provides four runtime controllers that interact with an active `py_dss_interface.DSS` session. Each controller implements the same algorithm as its multiconductor counterpart.

### Load Tap Changer Control

Discrete voltage regulator — same algorithm as `multiconductor.control.load_tap_changer_control.LoadTapChangerControl`.

```python
from opendss.control import LoadTapChangerControl

ctrl = LoadTapChangerControl(
    dss,
    transformer_names=["Reg1a", "Reg1b", "Reg1c"],
    vm_lower_pu=0.98,
    vm_upper_pu=1.02,
    mode="gang",              # "gang" or "phase"
    detect_oscillation=True,
)
result = ctrl.run()
# result = {"converged": True, "iterations": 5, "taps": [1.025, 1.025, 1.025]}
```

### Line Drop Compensation Control

Voltage regulator with remote load-centre voltage estimation — same algorithm as `multiconductor.control.line_drop_control.LineDropControlExtended`.

```python
from opendss.control import LineDropControl

ctrl = LineDropControl(
    dss,
    transformer_names=["Reg1a", "Reg1b", "Reg1c"],
    mode="bidirectional",     # "bidirectional", "locked_forward", "locked_reverse"
    v_set_secondary_v=122.0,
    bandwidth_secondary_v=2.0,
    pt_ratio=20.0,
    ct_primary_rating_a=700.0,
    r_ldc_v=3.0,
    x_ldc_v=9.0,
)
result = ctrl.run()
```

### Binary Shunt (Capacitor Bank) Controller

Voltage-switched capacitor bank — same algorithm as `multiconductor.control.shunt_controller.MulticonductorBinaryShuntController`.

```python
from opendss.control import BinaryShuntController

ctrl = BinaryShuntController(
    dss,
    capacitor_names=["Cap1"],
    bus_names=["bus_650"],
    v_threshold_on=0.95,
    v_threshold_off=1.05,
    control_mode="switched",  # "switched" or "fixed"
)
result = ctrl.run()
```

### Volt-Var Controller

Q(V) curve-based reactive-power control for generators / PV systems — same algorithm as `multiconductor.control.volt_var_control.VoltVarController`.

```python
from opendss.control import VoltVarController, QVCurve

curve = QVCurve(
    vm_points_pu=[0.90, 0.95, 0.98, 1.02, 1.05, 1.10],
    q_points_pu=[ 0.44, 0.44, 0.00, 0.00,-0.44,-0.44],
)
ctrl = VoltVarController(
    dss,
    element_name="PVSystem.PV1",
    bus_name="bus_680",
    phases=[1, 2, 3],
    sn_kva=500.0,
    qv_curve=curve,
    damping_coef=2.0,
)
result = ctrl.run()
```

## DSS Script Generation

The converter translates a multiconductor `pandapowerNet` into an OpenDSS `.dss` script file, including control elements.

```python
from opendss.tools.dss_converter import mc_net_to_opendss

dss_script = mc_net_to_opendss(net, filename="output.dss")
```

The converter writes: Circuit, LineCode, Line, Switch, Load, Generator, Transformer, Capacitor, RegControl, and CapControl elements.

## Multiconductor ↔ OpenDSS Control Mapping

| Multiconductor Controller | OpenDSS Controller | OpenDSS DSS Element |
|---|---|---|
| `LoadTapChangerControl` | `LoadTapChangerControl` | `RegControl` |
| `LineDropControl` / `LineDropControlExtended` | `LineDropControl` | `RegControl` (with R, X) |
| `MulticonductorBinaryShuntController` | `BinaryShuntController` | `CapControl` |
| `VoltVarController` | `VoltVarController` | Generator/PVSystem kvar |

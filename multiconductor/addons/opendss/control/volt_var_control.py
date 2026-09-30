"""
OpenDSS Volt-Var Control.

Implements the same algorithm as
``multiconductor.control.volt_var_control.VoltVarController``
but operates on an OpenDSS model via ``py_dss_interface``.

The controller adjusts the reactive-power output of a Generator or
PVSystem element along a user-defined Q(V) curve.  Phase-to-phase
voltages are computed from the phase-to-ground results using the
same ``convert_v_p2g_p2p`` logic as the multiconductor version.
The worst-case (max deviation from 1.0 pu) phase-to-phase voltage
drives the Q setpoint (gang-mode operation).  Apparent power is
saturated to the inverter nameplate rating if specified.

Usage::

    from opendss.control.volt_var_control import VoltVarController, QVCurve

    dss = py_dss_interface.DSS()
    dss.text("compile my_circuit.dss")
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
    )
    result = ctrl.run()
"""

import logging
import numpy as np
from math import sqrt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QV Curve model (identical to multiconductor.control.volt_var_control)
# ---------------------------------------------------------------------------

class QVCurve:
    """Piecewise-linear Q(V) characteristic curve.

    Parameters
    ----------
    vm_points_pu : list[float]
        Voltage breakpoints in per-unit (ascending).
    q_points_pu : list[float]
        Normalised reactive-power values at each breakpoint.
        Positive = capacitive / underexcited; negative = inductive /
        overexcited.  Values are relative to the element's rated
        apparent power (*sn_kva*).
    """

    def __init__(self, vm_points_pu, q_points_pu):
        self.vm_points_pu = list(vm_points_pu)
        self.q_points_pu = list(q_points_pu)

    def step(self, vm_pu):
        """Interpolate Q/Sn from the curve for a given per-unit voltage."""
        return float(np.interp(vm_pu, self.vm_points_pu, self.q_points_pu))


# ---------------------------------------------------------------------------
# Phase-to-phase voltage conversion (from multiconductor.control.tools)
# ---------------------------------------------------------------------------

def _convert_v_p2g_p2p(v_3ph):
    """Convert 3 × (magnitude_pu, angle_deg) phase-to-ground to phase-to-phase.

    Parameters
    ----------
    v_3ph : list
        ``[[vm1, va1], [vm2, va2], [vm3, va3]]`` — per-unit magnitudes
        and angles in degrees.

    Returns
    -------
    list
        ``[[vm12, va12], [vm23, va23], [vm31, va31]]`` in per-unit and
        degrees.
    """
    arr = np.asarray(v_3ph, dtype=float)
    if arr.shape != (3, 2):
        return None
    s = 1.0 / sqrt(3)
    U = [arr[i, 0] * s * np.exp(1j * np.deg2rad(arr[i, 1])) for i in range(3)]

    pairs = [U[0] - U[1], U[1] - U[2], U[2] - U[0]]
    return [[abs(p), np.rad2deg(np.angle(p))] for p in pairs]


# ---------------------------------------------------------------------------
# VoltVarController
# ---------------------------------------------------------------------------

class VoltVarController:
    """Volt-var controller for an OpenDSS Generator or PVSystem element.

    Parameters
    ----------
    dss : py_dss_interface.DSS
        Active DSS session with a compiled circuit.
    element_name : str
        Full DSS element name, e.g. ``"Generator.DG1"`` or
        ``"PVSystem.PV1"``.
    bus_name : str
        Name of the bus where voltage is measured.
    phases : list[int]
        Node numbers to read (e.g. ``[1, 2, 3]``).
    sn_kva : float
        Rated apparent power of the element (kVA).
    qv_curve : QVCurve
        Q(V) characteristic curve.
    saturate_sn_kva : float or None
        If given, apparent power is clipped to this limit.
    damping_coef : float
        Damping coefficient (≥ 1).  Higher values slow the Q update.
    max_q_error_kvar : float
        Convergence tolerance on reactive power (kvar).
    max_iterations : int
        Maximum solve-then-adjust iterations.
    """

    def __init__(self, dss, element_name, bus_name, phases,
                 sn_kva, qv_curve, saturate_sn_kva=None,
                 damping_coef=2.0, max_q_error_kvar=0.1,
                 max_iterations=30):
        self.dss = dss
        self.element_name = element_name
        self.bus_name = bus_name
        self.phases = list(phases)
        self.sn_kva = float(sn_kva)
        self.qv_curve = qv_curve
        self.saturate_sn_kva = float(saturate_sn_kva) if saturate_sn_kva is not None else None
        self.damping_coef = float(damping_coef)
        self.max_q_error_kvar = float(max_q_error_kvar)
        self.max_iterations = max_iterations

        # Current Q output (kvar) — initialised from the model
        self._current_q_kvar = self._read_q_kvar()

    # ------------------------------------------------------------------
    # DSS helpers
    # ------------------------------------------------------------------

    def _read_q_kvar(self):
        """Read the current kvar setting of the element."""
        elem_type = self.element_name.split(".")[0]
        elem_id = self.element_name.split(".", 1)[1]
        self.dss.text(f"? {elem_type}.{elem_id}.kvar")
        try:
            return float(self.dss.text.result)
        except (ValueError, TypeError):
            return 0.0

    def _write_q_kvar(self, q_kvar):
        """Write a new kvar setting to the element."""
        elem_type = self.element_name.split(".")[0]
        elem_id = self.element_name.split(".", 1)[1]
        self.dss.text(f"{elem_type}.{elem_id}.kvar={q_kvar:.6g}")
        self._current_q_kvar = q_kvar

    def _read_p_kw(self):
        """Read current kW output of the element."""
        elem_type = self.element_name.split(".")[0]
        elem_id = self.element_name.split(".", 1)[1]
        self.dss.text(f"? {elem_type}.{elem_id}.kW")
        try:
            return float(self.dss.text.result)
        except (ValueError, TypeError):
            return 0.0

    def _get_worst_p2p_voltage_pu(self):
        """Return the phase-to-phase voltage with the worst deviation from 1.0.

        Uses the same gang-mode logic as the multiconductor VoltVarController:
        compute all three phase-to-phase voltages, pick the one with the
        largest ``|1 - V|`` deviation.
        """
        self.dss.circuit.set_active_bus(self.bus_name)
        va = self.dss.bus.vmag_angle_pu  # [mag1, ang1, mag2, ang2, …]
        nodes = list(self.dss.bus.nodes)

        # Build per-phase magnitudes and angles
        phase_v = {}
        for j, node in enumerate(nodes):
            if j * 2 + 1 < len(va):
                phase_v[int(node)] = [va[j * 2], va[j * 2 + 1]]

        # Need exactly 3 phases for P2P conversion
        v_3ph = []
        for ph in self.phases:
            if ph in phase_v:
                v_3ph.append(phase_v[ph])
            else:
                v_3ph.append([1.0, (ph - 1) * -120.0])

        if len(v_3ph) == 3:
            v_p2p = _convert_v_p2g_p2p(v_3ph)
            if v_p2p is not None:
                mags = [v[0] for v in v_p2p]
                worst_idx = int(np.argmax([abs(1.0 - m) for m in mags]))
                return mags[worst_idx]

        # Fallback: average phase-to-ground voltage
        if phase_v:
            return float(np.mean([v[0] for v in phase_v.values()]))
        return 1.0

    # ------------------------------------------------------------------
    # Control algorithm (mirrors multiconductor VoltVarController)
    # ------------------------------------------------------------------

    def _compute_target_q_kvar(self):
        """Determine target Q from the QV curve and current bus voltage."""
        vm_pu = self._get_worst_p2p_voltage_pu()
        q_pu = self.qv_curve.step(vm_pu)
        q_kvar = q_pu * self.sn_kva

        # Apparent-power saturation (Q-priority, then reduce P if needed)
        if self.saturate_sn_kva is not None:
            q_kvar = float(np.clip(q_kvar, -self.saturate_sn_kva, self.saturate_sn_kva))

        return q_kvar

    def control_step(self):
        """Execute one Q-adjustment step with damping.

        Same damping formula as multiconductor:
            ``new_q = current_q + (target_q - current_q) / damping_coef``
        """
        target_q = self._compute_target_q_kvar()
        damped_q = self._current_q_kvar + (target_q - self._current_q_kvar) / self.damping_coef

        # Apparent-power saturation after damping
        if self.saturate_sn_kva is not None:
            p_kw = self._read_p_kw()
            s_limit = self.saturate_sn_kva
            if p_kw ** 2 + damped_q ** 2 > s_limit ** 2:
                damped_q = float(np.clip(damped_q, -s_limit, s_limit))
                remaining_p = sqrt(max(s_limit ** 2 - damped_q ** 2, 0.0))
                if abs(p_kw) > remaining_p:
                    elem_type = self.element_name.split(".")[0]
                    elem_id = self.element_name.split(".", 1)[1]
                    self.dss.text(f"{elem_type}.{elem_id}.kW={remaining_p:.6g}")

        self._write_q_kvar(damped_q)
        return damped_q

    def is_converged(self):
        """Check whether current Q is close enough to the target Q."""
        target_q = self._compute_target_q_kvar()
        return abs(target_q - self._current_q_kvar) <= self.max_q_error_kvar

    def run(self):
        """Iterative control loop: solve → check → adjust → repeat.

        Returns
        -------
        dict
            ``{"converged": bool, "iterations": int, "q_kvar": float}``
        """
        for iteration in range(1, self.max_iterations + 1):
            self.dss.text("solve")
            if self.is_converged():
                return {
                    "converged": True,
                    "iterations": iteration,
                    "q_kvar": self._current_q_kvar,
                }
            self.control_step()

        return {
            "converged": False,
            "iterations": self.max_iterations,
            "q_kvar": self._current_q_kvar,
        }


def add_volt_var_control(dss, element_name, bus_name, phases, sn_kva,
                         qv_curve, saturate_sn_kva=None, damping_coef=2.0,
                         max_iterations=30):
    """Convenience function matching multiconductor's ``add_volt_var_control``.

    Creates a :class:`VoltVarController` and immediately runs the
    iterative control loop.

    Returns
    -------
    dict
        Result dictionary from :meth:`VoltVarController.run`.
    """
    ctrl = VoltVarController(
        dss,
        element_name=element_name,
        bus_name=bus_name,
        phases=phases,
        sn_kva=sn_kva,
        qv_curve=qv_curve,
        saturate_sn_kva=saturate_sn_kva,
        damping_coef=damping_coef,
        max_iterations=max_iterations,
    )
    return ctrl.run()

# -------------------------------------------------------------------------------------------------
# OpenDSS DSS-script generation
# -------------------------------------------------------------------------------------------------

from opendss.control.storage_controller import is_battery_name, storage_element_name


def _vvc_curve_points(ctrl):
    """Return (vm_points_pu, q_points_pu) extracted from a ``VoltVarController``.

    The qv-curve lives on ``ctrl.q_model.qv_curve``.
    """
    q_model = getattr(ctrl, "q_model", None)
    if q_model is None:
        return None, None
    qv = getattr(q_model, "qv_curve", None)
    if qv is None:
        return None, None
    vm = list(getattr(qv, "vm_points_pu", []))
    qp = list(getattr(qv, "q_points_pu", []))
    if not vm or not qp or len(vm) != len(qp):
        return None, None
    return vm, qp


def _sgen_dss_name(net, lidx, circ):
    """Mirror ``_write_sgens`` naming logic from the DSS converter."""
    try:
        multi = len(net.asymmetric_sgen.loc[lidx]) > 1
    except Exception:
        multi = False
    try:
        raw = net.asymmetric_sgen.loc[(lidx, circ), "name"]
        raw_name = str(raw) if raw is not None and str(raw) != "None" else None
    except Exception:
        raw_name = None
    try:
        bus_int = int(net.asymmetric_sgen.loc[(lidx, circ), "bus"])
    except Exception:
        bus_int = lidx
    is_pv = raw_name is not None and "PHOTOVOLTAIC" in raw_name.upper()
    is_storage = is_battery_name(raw_name)
    if is_storage:
        return storage_element_name(net, lidx, circ), "Storage"
    if raw_name:
        base = raw_name if not multi else f"{raw_name}_{circ}"
    else:
        prefix = "PV" if is_pv else "Gen"
        base = f"{prefix}_{bus_int}_{lidx}_{circ}"
    cls = "PVSystem" if is_pv else "Generator"
    return base.replace(" ", "_").replace(".", "_"), cls


def write_volt_var_controls(net, lines):
    """Append OpenDSS ``New XYCurve`` + ``New InvControl`` commands for every
    ``VoltVarController`` found in ``net.controller``.

    Notes
    -----
    OpenDSS ``InvControl`` only operates on ``PVSystem`` / ``Storage``
    elements.  The DSS converter emits ``asymmetric_sgen`` rows whose name
    contains ``PHOTOVOLTAIC`` as ``PVSystem`` and everything else as
    ``Generator``.  This writer emits a live ``InvControl`` only when **all**
    DERs in the controller's element list are PVSystems; otherwise the
    ``InvControl`` is written as a comment so the script still loads.

    Parameters
    ----------
    net : pandapowerNet
        Multiconductor network.
    lines : list[str]
        Accumulator list of DSS script lines.
    """
    if not hasattr(net, "controller"):
        return
    try:
        if net.controller is None or net.controller.empty:
            return
    except Exception:
        return

    has_content = False
    used_curves = set()

    for _idx, ctrl_row in net.controller.iterrows():
        ctrl = ctrl_row["object"]
        if type(ctrl).__name__ != "VoltVarController":
            continue
        if not ctrl_row.get("in_service", True):
            continue

        element = getattr(ctrl, "element", "asymmetric_sgen")
        if element != "asymmetric_sgen":
            continue

        vm_pts, q_pts = _vvc_curve_points(ctrl)
        if vm_pts is None:
            continue

        # Group element_index by first-level (lidx); each lidx is one DER
        seen_lidx = []
        for ei in getattr(ctrl, "element_index", []):
            lidx = ei[0] if isinstance(ei, tuple) else ei
            if lidx not in seen_lidx:
                seen_lidx.append(lidx)
        if not seen_lidx:
            continue

        # DER names: pick first circuit row per lidx
        der_names = []
        all_supported = True
        for lidx in seen_lidx:
            try:
                circ = net.asymmetric_sgen.loc[lidx].index[0]
            except Exception:
                circ = 0
            elt_name, cls = _sgen_dss_name(net, lidx, circ)
            if cls not in ("PVSystem", "Storage"):
                all_supported = False
            der_names.append(f"{cls}.{elt_name}")
        if not der_names:
            continue

        curve_name = f"VVC_Curve_{seen_lidx[0]}"
        suffix = 1
        while curve_name in used_curves:
            suffix += 1
            curve_name = f"VVC_Curve_{seen_lidx[0]}_{suffix}"
        used_curves.add(curve_name)

        invctrl_name = f"VVC_{seen_lidx[0]}"

        if not has_content:
            lines.append("! Volt-Var Controls")
            has_content = True

        npts = len(vm_pts)
        xarr = "[" + " ".join(f"{v:.6g}" for v in vm_pts) + "]"
        yarr = "[" + " ".join(f"{q:.6g}" for q in q_pts) + "]"

        # Always emit the XYCurve so users can wire it up manually if needed
        lines.append(
            f"New XYCurve.{curve_name} npts={npts} xarray={xarr} yarray={yarr}"
        )

        derlist = "[" + " ".join(der_names) + "]"
        invctrl_cmd = (
            f"New InvControl.{invctrl_name} DERList={derlist} "
            f"mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1={curve_name} "
            f"RefReactivePower=VARMAX eventlog=Yes deltaQ_factor=0.4 varchangetolerance=0.005 voltagechangetolerance=0.0001"
        )

        if all_supported:
            # Safe to emit live: every DER in the list is a PVSystem or Storage
            lines.append(invctrl_cmd)
        else:
            # OpenDSS InvControl crashes (Error 303 / access violation) when
            # referencing a Generator.  Emit as a comment so the script loads.
            lines.append(
                f"! Volt-var control for {derlist} not emitted: OpenDSS "
                f"InvControl requires PVSystem/Storage, not Generator. "
                f"Convert the listed generators to PVSystem to enable the "
                f"InvControl below."
            )
            lines.append(f"! {invctrl_cmd}")

    if has_content:
        lines.append("")

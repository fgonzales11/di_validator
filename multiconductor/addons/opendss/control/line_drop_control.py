"""
OpenDSS Line Drop Compensation Control.

Implements the same algorithm as
``multiconductor.control.line_drop_control.LineDropControl`` and
``multiconductor.control.line_drop_control.LineDropControlExtended``
but operates on an OpenDSS model via ``py_dss_interface``.

The controller estimates the voltage at a remote load centre by
subtracting the line-drop-compensator impedance drop from the
measured voltage at the transformer secondary.  The estimated
voltage is then compared to the voltage bandwidth and the tap is
adjusted accordingly.

Modes:

* **bidirectional** — regulates in both power-flow directions.
* **locked_forward** — regulates only when real power flows from
  primary to secondary (forward).  In reverse flow the tap is
  left unchanged.
* **locked_reverse** — regulates only when real power flows from
  secondary to primary.

Usage::

    from opendss.control.line_drop_control import LineDropControl

    dss = py_dss_interface.DSS()
    dss.text("compile my_circuit.dss")
    ctrl = LineDropControl(
        dss,
        transformer_names=["Reg1a", "Reg1b", "Reg1c"],
        mode="bidirectional",
        v_set_secondary_v=122.0,
        bandwidth_secondary_v=2.0,
        pt_ratio=20.0,
        ct_primary_rating_a=700.0,
        r_ldc_v=3.0,
        x_ldc_v=9.0,
    )
    result = ctrl.run()
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class LineDropControl:
    """Line-drop compensation controller for OpenDSS transformers.

    Parameters
    ----------
    dss : py_dss_interface.DSS
        Active DSS session with a compiled circuit.
    transformer_names : list[str]
        Names of single-phase transformer elements forming the
        regulating bank.
    mode : str
        ``"bidirectional"``, ``"locked_forward"``, or ``"locked_reverse"``.
    v_set_secondary_v : float
        Voltage setpoint on the 120 V secondary base (volts).
    bandwidth_secondary_v : float
        Voltage bandwidth on the 120 V secondary base (volts).
    pt_ratio : float
        Potential transformer ratio.
    ct_primary_rating_a : float
        CT primary current rating in amperes.
    r_ldc_v : float
        Line-drop compensator R setting (volts on 120 V base).
    x_ldc_v : float
        Line-drop compensator X setting (volts on 120 V base).
    winding : int
        Controlled winding (default ``2`` — secondary).
    max_iterations : int
        Maximum solve-then-adjust iterations.
    """

    def __init__(self, dss, transformer_names, mode,
                 v_set_secondary_v, bandwidth_secondary_v,
                 pt_ratio, ct_primary_rating_a,
                 r_ldc_v, x_ldc_v,
                 winding=2, max_iterations=30):
        valid_modes = ("bidirectional", "locked_forward", "locked_reverse")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

        self.dss = dss
        self.transformer_names = list(transformer_names)
        self.mode = mode
        self.winding = winding
        self.max_iterations = max_iterations

        # LDC settings
        self.pt_ratio = float(pt_ratio)
        self.ct_primary_rating_a = float(ct_primary_rating_a)
        self.r_ldc_v = float(r_ldc_v)
        self.x_ldc_v = float(x_ldc_v)
        self.z_ldc_v = complex(r_ldc_v, x_ldc_v)

        # Voltage setpoints — convert from secondary volts to per-unit
        # on the controlled-side nominal voltage.
        self._v_set_secondary_v = float(v_set_secondary_v)
        self._bandwidth_secondary_v = float(bandwidth_secondary_v)

        # Read tap parameters and bus info from DSS
        self._read_tap_params()

        # Compute per-unit setpoints from secondary-volt settings
        # V_pu = V_secondary / 120 (since secondary base is 120 V)
        self.vm_set_pu = v_set_secondary_v / 120.0
        self.vm_lower_pu = (v_set_secondary_v - bandwidth_secondary_v) / 120.0
        self.vm_upper_pu = (v_set_secondary_v + bandwidth_secondary_v) / 120.0

    # ------------------------------------------------------------------
    # DSS helpers
    # ------------------------------------------------------------------

    def _read_tap_params(self):
        """Cache tap limits, step size, buses, and kV ratings."""
        self.max_tap = []
        self.min_tap = []
        self.num_taps = []
        self.buses = []
        self.kv_ratings = []
        for name in self.transformer_names:
            self.dss.text(f"Transformer.{name}.wdg={self.winding}")
            self.dss.text(f"? Transformer.{name}.MaxTap")
            self.max_tap.append(float(self.dss.text.result))
            self.dss.text(f"? Transformer.{name}.MinTap")
            self.min_tap.append(float(self.dss.text.result))
            self.dss.text(f"? Transformer.{name}.NumTaps")
            self.num_taps.append(int(float(self.dss.text.result)))
            self.dss.text(f"? Transformer.{name}.buses")
            buses_raw = self.dss.text.result.replace("[", "").replace("]", "").split(",")
            self.buses.append(buses_raw[self.winding - 1].strip().split(".")[0])
            self.dss.text(f"? Transformer.{name}.kVs")
            kvs_raw = self.dss.text.result.replace("[", "").replace("]", "").split(",")
            self.kv_ratings.append(float(kvs_raw[self.winding - 1]))

        self.tap_step = [
            (mx - mn) / nt if nt > 0 else 0.0
            for mx, mn, nt in zip(self.max_tap, self.min_tap, self.num_taps)
        ]

    def _get_taps(self):
        taps = []
        for name in self.transformer_names:
            self.dss.text(f"Transformer.{name}.wdg={self.winding}")
            self.dss.text(f"? Transformer.{name}.Tap")
            taps.append(float(self.dss.text.result))
        return np.array(taps)

    def _set_taps(self, taps):
        for name, tap in zip(self.transformer_names, taps):
            self.dss.text(f"Transformer.{name}.wdg={self.winding}")
            self.dss.text(f"Transformer.{name}.Tap={tap}")

    def _get_bus_voltage_complex(self, bus_name):
        """Return voltage as complex kV (phase-to-neutral) at *bus_name*."""
        self.dss.circuit.set_active_bus(bus_name)
        va = self.dss.bus.vmag_angle_pu  # [mag1, ang1, …]
        if len(va) >= 2:
            kv_base = self.dss.bus.kv_base  # L-N kV
            v_mag = va[0] * kv_base  # kV
            v_ang = np.deg2rad(va[1])
            return v_mag * (np.cos(v_ang) + 1j * np.sin(v_ang))
        return complex(0.0)

    def _get_transformer_current_complex(self, name):
        """Return complex secondary current (kA) for transformer *name*.

        Uses the CktElement interface to read current magnitude and angle
        at the secondary terminal.
        """
        self.dss.circuit.set_active_element(f"Transformer.{name}")
        cma = self.dss.cktelement.currents_mag_ang  # [I1, A1, I2, A2, …]
        n_cond = self.dss.cktelement.num_conductors
        # Secondary currents start after the first winding's conductors
        sec_start = n_cond * 2  # each conductor has (mag, ang), secondary offset
        if sec_start + 1 < len(cma):
            i_mag = cma[sec_start]      # amps
            i_ang = np.deg2rad(cma[sec_start + 1])
            return (i_mag / 1000.0) * (np.cos(i_ang) + 1j * np.sin(i_ang))
        return complex(0.0)

    def _get_transformer_power(self, name):
        """Return complex power (P + jQ) in kW at the secondary."""
        self.dss.circuit.set_active_element(f"Transformer.{name}")
        powers = self.dss.cktelement.powers  # [P1, Q1, P2, Q2, …]
        n_cond = self.dss.cktelement.num_conductors
        sec_start = n_cond * 2
        if sec_start + 1 < len(powers):
            return complex(powers[sec_start], powers[sec_start + 1])
        return complex(0.0)

    # ------------------------------------------------------------------
    # Voltage estimation (same as multiconductor LineDropControlExtended)
    # ------------------------------------------------------------------

    def _estimate_voltage_pu(self):
        """Estimate per-unit voltage at load centre for each transformer.

        This replicates the ``LineDropControlExtended.estimated_voltage``
        algorithm: the tap-side voltage is reduced by the LDC impedance
        drop weighted by the actual current and CT/PT ratios.

        Returns
        -------
        numpy.ndarray
            Estimated per-unit voltage for each transformer.
        bool
            ``True`` if power flow is in the forward direction.
        """
        v_est = np.zeros(len(self.transformer_names))
        total_real_power = 0.0

        for i, name in enumerate(self.transformer_names):
            # Tap-side voltage (complex, L-N, kV)
            V_bus_kv = self._get_bus_voltage_complex(self.buses[i])
            # Current at secondary (complex, kA)
            I_sec_ka = self._get_transformer_current_complex(name)
            # Power at secondary (for forward/reverse detection)
            S = self._get_transformer_power(name)
            total_real_power += S.real

            kv_rating = self.kv_ratings[i]  # L-L kV rating of secondary

            # Convert to secondary voltage (volts)
            V_sec_v = V_bus_kv * 1000.0 / self.pt_ratio

            # LDC voltage drop:  Z_ldc * I / CT_rating
            # I_sec_ka is in kA, CT rating is in A -> multiply by 1000
            V_drop_v = self.z_ldc_v * (I_sec_ka * 1000.0) / self.ct_primary_rating_a

            # Estimated secondary voltage (volts)
            V_est_v = V_sec_v - V_drop_v

            # Convert back to per-unit (on the kV-base of the bus)
            V_est_kv = np.abs(V_est_v) * self.pt_ratio / 1000.0
            kv_base_ln = kv_rating / np.sqrt(3) if kv_rating > 0 else 1.0
            v_est[i] = V_est_kv / kv_base_ln if kv_base_ln > 0 else 1.0

        pf_forward = total_real_power < 0  # negative = power from primary to secondary
        return v_est, pf_forward

    # ------------------------------------------------------------------
    # Control algorithm
    # ------------------------------------------------------------------

    def control_step(self):
        """Execute one tap-adjustment step.

        Returns
        -------
        numpy.ndarray | None
            New tap positions, or *None* if skipped due to mode lock.
        """
        vm_pu, pf_forward = self._estimate_voltage_pu()
        taps = self._get_taps()
        max_tap = np.array(self.max_tap)
        min_tap = np.array(self.min_tap)
        tap_step = np.array(self.tap_step)

        # Mode-based gating (same as multiconductor)
        if self.mode == "locked_forward" and not pf_forward:
            return None
        if self.mode == "locked_reverse" and pf_forward:
            return None

        # Tap direction: raising tap raises secondary voltage for a step-up regulator
        increment = np.where(
            (vm_pu < self.vm_lower_pu) & (taps + tap_step <= max_tap),
            tap_step,
            np.where(
                (vm_pu > self.vm_upper_pu) & (taps - tap_step >= min_tap),
                -tap_step,
                0.0,
            ),
        )

        new_taps = np.clip(taps + increment, min_tap, max_tap)
        self._set_taps(new_taps)
        return new_taps

    def is_converged(self):
        """Check whether estimated voltages are within the voltage band."""
        vm_pu, pf_forward = self._estimate_voltage_pu()

        # If locked out by mode, consider converged
        if self.mode == "locked_forward" and not pf_forward:
            return True
        if self.mode == "locked_reverse" and pf_forward:
            return True

        taps = self._get_taps()
        max_tap = np.array(self.max_tap)
        min_tap = np.array(self.min_tap)

        in_band = (vm_pu >= self.vm_lower_pu) & (vm_pu <= self.vm_upper_pu)
        at_limit_low = (vm_pu < self.vm_lower_pu) & np.isclose(taps, max_tap)
        at_limit_high = (vm_pu > self.vm_upper_pu) & np.isclose(taps, min_tap)

        return bool(np.all(in_band | at_limit_low | at_limit_high))

    def run(self):
        """Iterative control loop: solve → estimate → adjust → repeat.

        Returns
        -------
        dict
            ``{"converged": bool, "iterations": int, "taps": list}``
        """
        for iteration in range(1, self.max_iterations + 1):
            self.dss.text("solve")
            if self.is_converged():
                return {
                    "converged": True,
                    "iterations": iteration,
                    "taps": self._get_taps().tolist(),
                }
            self.control_step()

        return {
            "converged": False,
            "iterations": self.max_iterations,
            "taps": self._get_taps().tolist(),
        }


def add_line_drop_control(dss, transformer_names, mode,
                          v_set_secondary_v, bandwidth_secondary_v,
                          pt_ratio, ct_primary_rating_a,
                          r_ldc_v, x_ldc_v, max_iterations=30):
    """Convenience function matching multiconductor's ``add_line_drop_control``.

    Creates a :class:`LineDropControl` and immediately runs the
    iterative control loop.

    Returns
    -------
    dict
        Result dictionary from :meth:`LineDropControl.run`.
    """
    ctrl = LineDropControl(
        dss,
        transformer_names=transformer_names,
        mode=mode,
        v_set_secondary_v=v_set_secondary_v,
        bandwidth_secondary_v=bandwidth_secondary_v,
        pt_ratio=pt_ratio,
        ct_primary_rating_a=ct_primary_rating_a,
        r_ldc_v=r_ldc_v,
        x_ldc_v=x_ldc_v,
        max_iterations=max_iterations,
    )
    return ctrl.run()

# -------------------------------------------------------------------------------------------------
# OpenDSS DSS-script generation
# -------------------------------------------------------------------------------------------------

def write_line_drop_controls(net, lines):
    """Append OpenDSS ``New RegControl`` commands for every
    ``LineDropControl`` / ``LineDropControlExtended`` controller in
    ``net.controller``.

    This is a thin wrapper around
    :func:`opendss.control.regulator.write_regcontrols` filtered to
    line-drop-compensation types.

    Parameters
    ----------
    net : pandapowerNet
        Multiconductor network.
    lines : list[str]
        Accumulator list of DSS script lines.
    """
    from opendss.control.regulator import write_regcontrols
    write_regcontrols(
        net, lines,
        type_filter=("LineDropControl", "LineDropControlExtended"),
        section_header="! Line Drop Compensation Controls",
    )

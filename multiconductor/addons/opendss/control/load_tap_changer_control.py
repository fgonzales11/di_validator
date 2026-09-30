"""
OpenDSS Load Tap Changer Control.

Implements the same algorithm as
``multiconductor.control.load_tap_changer_control.LoadTapChangerControl``
but operates on an OpenDSS model via ``py_dss_interface``.

Supports two operating modes:

* **gang** — all phases use the same tap; the worst-case (min/max)
  phase voltage drives the decision.
* **phase** — each single-phase transformer winding is tapped
  independently.

Usage::

    from opendss.control.load_tap_changer_control import LoadTapChangerControl

    dss = py_dss_interface.DSS()
    dss.text("compile my_circuit.dss")
    ctrl = LoadTapChangerControl(
        dss,
        transformer_names=["Reg1a", "Reg1b", "Reg1c"],
        vm_lower_pu=0.98,
        vm_upper_pu=1.02,
        mode="gang",
    )
    ctrl.run()
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class LoadTapChangerControl:
    """Discrete tap-changer controller for OpenDSS RegControl-style regulation.

    Parameters
    ----------
    dss : py_dss_interface.DSS
        Active DSS session with a compiled circuit.
    transformer_names : list[str]
        Names of the single-phase transformer elements that form the
        regulating bank (e.g. ``["Reg1a", "Reg1b", "Reg1c"]``).
    vm_lower_pu : float
        Lower voltage limit in per-unit.
    vm_upper_pu : float
        Upper voltage limit in per-unit.
    mode : str
        ``"gang"`` (default) or ``"phase"``.
    winding : int
        Controlled winding (default ``2``).
    detect_oscillation : bool
        If *True*, the controller declares convergence when it detects
        that the tap position is oscillating between two states.
    max_iterations : int
        Maximum solve-then-adjust iterations (default ``30``).
    """

    def __init__(self, dss, transformer_names, vm_lower_pu, vm_upper_pu,
                 mode="gang", winding=2, detect_oscillation=False,
                 max_iterations=30):
        if mode not in ("gang", "phase"):
            raise ValueError(f"mode must be 'gang' or 'phase', got '{mode}'")

        self.dss = dss
        self.transformer_names = list(transformer_names)
        self.vm_lower_pu = float(vm_lower_pu)
        self.vm_upper_pu = float(vm_upper_pu)
        self.mode = mode
        self.winding = winding
        self.detect_oscillation = detect_oscillation
        self.max_iterations = max_iterations
        self.oscillating = False

        # Tap parameters read once from the model
        self._read_tap_params()

    # ------------------------------------------------------------------
    # DSS helpers
    # ------------------------------------------------------------------

    def _read_tap_params(self):
        """Cache tap limits and step size from the DSS model."""
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
        """Return current tap positions as a numpy array."""
        taps = []
        for name in self.transformer_names:
            self.dss.text(f"Transformer.{name}.wdg={self.winding}")
            self.dss.text(f"? Transformer.{name}.Tap")
            taps.append(float(self.dss.text.result))
        return np.array(taps)

    def _set_taps(self, taps):
        """Write tap positions to the DSS model."""
        for name, tap in zip(self.transformer_names, taps):
            self.dss.text(f"Transformer.{name}.wdg={self.winding}")
            self.dss.text(f"Transformer.{name}.Tap={tap}")

    def _get_bus_voltages_pu(self):
        """Return per-unit voltage magnitude at each controlled bus."""
        vm_pu = []
        for bus in self.buses:
            self.dss.circuit.set_active_bus(bus)
            va = self.dss.bus.vmag_angle_pu  # [mag1, ang1, mag2, ang2, …]
            if len(va) >= 2:
                vm_pu.append(va[0])
            else:
                vm_pu.append(1.0)
        return np.array(vm_pu)

    # ------------------------------------------------------------------
    # Control algorithm
    # ------------------------------------------------------------------

    def control_step(self):
        """Execute one tap-adjustment step (same logic as multiconductor LTC)."""
        vm_pu = self._get_bus_voltages_pu()
        taps = self._get_taps()
        max_tap = np.array(self.max_tap)
        min_tap = np.array(self.min_tap)
        tap_step = np.array(self.tap_step)

        if self.mode == "phase":
            # Independent per-phase: raise tap to boost voltage, lower to reduce
            increment = np.where(
                (vm_pu < self.vm_lower_pu) & (taps + tap_step <= max_tap),
                tap_step,
                np.where(
                    (vm_pu > self.vm_upper_pu) & (taps - tap_step >= min_tap),
                    -tap_step,
                    0.0,
                ),
            )
        else:
            # Gang: use worst-case voltage, same increment for all
            v_min = np.min(vm_pu)
            v_max = np.max(vm_pu)
            if v_min < self.vm_lower_pu and np.all(taps + tap_step <= max_tap):
                increment = tap_step
            elif v_max > self.vm_upper_pu and np.all(taps - tap_step >= min_tap):
                increment = -tap_step
            else:
                increment = np.zeros_like(taps)

        new_taps = np.clip(taps + increment, min_tap, max_tap)
        self._set_taps(new_taps)
        return new_taps

    def is_converged(self):
        """Check whether all voltages are within band (or taps at limits)."""
        if self.oscillating and self.detect_oscillation:
            return True

        vm_pu = self._get_bus_voltages_pu()
        taps = self._get_taps()
        max_tap = np.array(self.max_tap)
        min_tap = np.array(self.min_tap)

        in_band = (vm_pu >= self.vm_lower_pu) & (vm_pu <= self.vm_upper_pu)
        at_limit_low = (vm_pu < self.vm_lower_pu) & np.isclose(taps, max_tap)
        at_limit_high = (vm_pu > self.vm_upper_pu) & np.isclose(taps, min_tap)

        if self.mode == "phase":
            return bool(np.all(in_band | at_limit_low | at_limit_high))
        else:
            v_min, v_max = np.min(vm_pu), np.max(vm_pu)
            band_ok = (v_min >= self.vm_lower_pu) and (v_max <= self.vm_upper_pu)
            limit_reached = (
                ((v_min < self.vm_lower_pu) and np.all(np.isclose(taps, max_tap)))
                or ((v_max > self.vm_upper_pu) and np.all(np.isclose(taps, min_tap)))
            )
            return band_ok or limit_reached

    def run(self):
        """Iterative control loop: solve → check → adjust → repeat.

        Returns
        -------
        dict
            ``{"converged": bool, "iterations": int, "taps": list}``
        """
        tap_prev1 = None
        tap_prev2 = None
        self.oscillating = False

        for iteration in range(1, self.max_iterations + 1):
            self.dss.text("solve")
            if self.is_converged():
                return {
                    "converged": True,
                    "iterations": iteration,
                    "taps": self._get_taps().tolist(),
                }

            tap_prev2 = tap_prev1
            tap_prev1 = self._get_taps().copy()

            new_taps = self.control_step()

            # Oscillation detection
            if tap_prev2 is not None and np.allclose(new_taps, tap_prev2):
                logger.warning(
                    "LoadTapChangerControl oscillating for transformers %s",
                    self.transformer_names,
                )
                self.oscillating = True
                if self.detect_oscillation:
                    return {
                        "converged": True,
                        "iterations": iteration,
                        "taps": new_taps.tolist(),
                    }

        return {
            "converged": False,
            "iterations": self.max_iterations,
            "taps": self._get_taps().tolist(),
        }


def add_load_tap_changer_control(dss, transformer_names, vm_lower_pu, vm_upper_pu,
                                 mode="gang", detect_oscillation=False, max_iterations=30):
    """Convenience function matching multiconductor's ``add_load_tap_changer_control``.

    Creates a :class:`LoadTapChangerControl` and immediately runs the
    iterative control loop.

    Returns
    -------
    dict
        Result dictionary from :meth:`LoadTapChangerControl.run`.
    """
    ctrl = LoadTapChangerControl(
        dss,
        transformer_names=transformer_names,
        vm_lower_pu=vm_lower_pu,
        vm_upper_pu=vm_upper_pu,
        mode=mode,
        detect_oscillation=detect_oscillation,
        max_iterations=max_iterations,
    )
    return ctrl.run()

# -------------------------------------------------------------------------------------------------
# OpenDSS DSS-script generation
# -------------------------------------------------------------------------------------------------

def write_load_tap_changer_controls(net, lines):
    """Append OpenDSS ``New RegControl`` commands for every
    ``LoadTapChangerControl`` controller in ``net.controller``.

    This is a thin wrapper around
    :func:`opendss.control.regulator.write_regcontrols` filtered to
    LTC-type controllers.

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
        type_filter=("LoadTapChangerControl",),
        section_header="! Load Tap Changer Controls",
    )

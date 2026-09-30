"""
OpenDSS Binary Shunt (Capacitor Bank) Controller.

Implements the same algorithm as
``multiconductor.control.shunt_controller.MulticonductorBinaryShuntController``
but operates on an OpenDSS model via ``py_dss_interface``.

The controller switches capacitor elements on or off based on
average bus voltage thresholds.  All capacitors in the group are
switched together (gang operation).

Modes:

* **switched** — the capacitor bank is opened/closed based on voltage.
* **fixed** — the capacitor is always energised regardless of voltage.

Usage::

    from opendss.control.shunt_controller import BinaryShuntController

    dss = py_dss_interface.DSS()
    dss.text("compile my_circuit.dss")
    ctrl = BinaryShuntController(
        dss,
        capacitor_names=["Cap1"],
        bus_names=["bus_650"],
        v_threshold_on=0.95,
        v_threshold_off=1.05,
        control_mode="switched",
    )
    result = ctrl.run()
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class BinaryShuntController:
    """Voltage-switched capacitor bank controller for OpenDSS.

    Parameters
    ----------
    dss : py_dss_interface.DSS
        Active DSS session with a compiled circuit.
    capacitor_names : list[str]
        Names of the ``Capacitor`` elements that form the bank.
    bus_names : list[str]
        Names of the buses whose voltages are monitored.
    v_threshold_on : float
        Per-unit voltage below which the bank is closed (energised).
    v_threshold_off : float
        Per-unit voltage above which the bank is opened.
    control_mode : str
        ``"switched"`` (default) or ``"fixed"``.
    max_iterations : int
        Maximum solve-then-adjust iterations (default ``30``).
    """

    def __init__(self, dss, capacitor_names, bus_names,
                 v_threshold_on=0.95, v_threshold_off=1.05,
                 control_mode="switched", max_iterations=30):
        if control_mode not in ("switched", "fixed"):
            raise ValueError(
                f"control_mode must be 'switched' or 'fixed', got '{control_mode}'"
            )

        self.dss = dss
        self.capacitor_names = list(capacitor_names)
        self.bus_names = list(bus_names)
        self.v_threshold_on = float(v_threshold_on)
        self.v_threshold_off = float(v_threshold_off)
        self.control_mode = control_mode
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # DSS helpers
    # ------------------------------------------------------------------

    def _get_avg_voltage_pu(self):
        """Return the average per-unit voltage across all monitored buses."""
        voltages = []
        for bus_name in self.bus_names:
            self.dss.circuit.set_active_bus(bus_name)
            va = self.dss.bus.vmag_angle_pu  # [mag1, ang1, mag2, ang2, …]
            # Collect all phase magnitudes
            for j in range(0, len(va), 2):
                voltages.append(va[j])
        if not voltages:
            return 1.0
        return float(np.mean(voltages))

    def _all_closed(self):
        """Return True if all capacitors are currently energised."""
        for name in self.capacitor_names:
            self.dss.text(f"? Capacitor.{name}.states")
            states_str = self.dss.text.result.replace("[", "").replace("]", "").strip()
            if not states_str:
                return False
            states = [int(s.strip()) for s in states_str.split(",") if s.strip()]
            if not all(s == 1 for s in states):
                return False
        return True

    def _all_open(self):
        """Return True if all capacitors are currently de-energised."""
        for name in self.capacitor_names:
            self.dss.text(f"? Capacitor.{name}.states")
            states_str = self.dss.text.result.replace("[", "").replace("]", "").strip()
            if not states_str:
                return True
            states = [int(s.strip()) for s in states_str.split(",") if s.strip()]
            if not all(s == 0 for s in states):
                return False
        return True

    def _close_all(self):
        """Energise (close) all capacitor elements."""
        for name in self.capacitor_names:
            # Determine number of phases to build the state vector
            self.dss.text(f"? Capacitor.{name}.phases")
            n = int(float(self.dss.text.result.strip() or "3"))
            state_vec = ", ".join(["1"] * n)
            self.dss.text(f"Capacitor.{name}.states=[{state_vec}]")
        logger.info("Capacitor bank closed: %s", self.capacitor_names)

    def _open_all(self):
        """De-energise (open) all capacitor elements."""
        for name in self.capacitor_names:
            self.dss.text(f"? Capacitor.{name}.phases")
            n = int(float(self.dss.text.result.strip() or "3"))
            state_vec = ", ".join(["0"] * n)
            self.dss.text(f"Capacitor.{name}.states=[{state_vec}]")
        logger.info("Capacitor bank opened: %s", self.capacitor_names)

    # ------------------------------------------------------------------
    # Control algorithm (mirrors MulticonductorBinaryShuntController)
    # ------------------------------------------------------------------

    def control_step(self):
        """Execute one switching step.

        * If *fixed* mode → always close.
        * If all closed and voltage >= off-threshold → open bank.
        * If all open and voltage <= on-threshold → close bank.
        """
        if self.control_mode == "fixed":
            self._close_all()
            return

        v_avg = self._get_avg_voltage_pu()
        closed = self._all_closed()
        opened = self._all_open()

        if closed and v_avg >= self.v_threshold_off:
            self._open_all()
        elif opened and v_avg <= self.v_threshold_on:
            self._close_all()

    def is_converged(self):
        """Check whether the controller has reached a stable state.

        Convergence conditions (same as multiconductor):

        * Fixed mode → always converged.
        * Voltage within (on, off) band → converged.
        * All open and voltage > on-threshold → converged (can't help).
        * All closed and voltage < off-threshold → converged (maxed out).
        """
        if self.control_mode == "fixed":
            return True

        v_avg = self._get_avg_voltage_pu()
        closed = self._all_closed()
        opened = self._all_open()

        # Within the deadband — no action needed
        if self.v_threshold_on < v_avg < self.v_threshold_off:
            return True
        # All open but voltage is not low enough to trigger → nothing to do
        if opened and v_avg > self.v_threshold_on:
            return True
        # All closed but voltage is not high enough to trigger → maxed out
        if closed and v_avg < self.v_threshold_off:
            return True

        return False

    def run(self):
        """Iterative control loop: solve → check → switch → repeat.

        Returns
        -------
        dict
            ``{"converged": bool, "iterations": int}``
        """
        # Initialise: respect fixed mode
        if self.control_mode == "fixed":
            self._close_all()

        for iteration in range(1, self.max_iterations + 1):
            self.dss.text("solve")
            if self.is_converged():
                return {"converged": True, "iterations": iteration}
            self.control_step()

        return {"converged": False, "iterations": self.max_iterations}

# -------------------------------------------------------------------------------------------------
# OpenDSS DSS-script generation
# -------------------------------------------------------------------------------------------------

def write_shunt_controllers(net, lines):
    """Append OpenDSS ``New CapControl`` commands for every
    ``MulticonductorBinaryShuntController`` in ``net.controller``.

    This is a thin wrapper around
    :func:`opendss.control.capacitor.write_capcontrols`.

    Parameters
    ----------
    net : pandapowerNet
        Multiconductor network.
    lines : list[str]
        Accumulator list of DSS script lines.
    """
    from opendss.control.capacitor import write_capcontrols
    write_capcontrols(net, lines)

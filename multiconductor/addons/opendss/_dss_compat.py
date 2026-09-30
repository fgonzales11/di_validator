"""
Compatibility adapter: makes DSS-Python (dss package) look like py_dss_interface.

DSS-Python uses the COM-style API (DSS.ActiveCircuit.ActiveBus.Name) while
py_dss_interface uses a flatter attribute style (d.bus.name).  This module
provides a thin wrapper so that code written for py_dss_interface works
unchanged when only DSS-Python is available (e.g. on Linux / Snowflake).
"""


class _BusProxy:
    __slots__ = ("_dss",)

    def __init__(self, dss):
        self._dss = dss

    @property
    def name(self):
        return self._dss.ActiveCircuit.ActiveBus.Name

    @property
    def kv_base(self):
        return self._dss.ActiveCircuit.ActiveBus.kVBase

    @property
    def nodes(self):
        return list(self._dss.ActiveCircuit.ActiveBus.Nodes)

    @property
    def vmag_angle_pu(self):
        return list(self._dss.ActiveCircuit.ActiveBus.puVmagAngle)


class _CktElementProxy:
    __slots__ = ("_dss",)

    def __init__(self, dss):
        self._dss = dss

    @property
    def currents_mag_ang(self):
        return list(self._dss.ActiveCircuit.ActiveCktElement.CurrentsMagAng)

    @property
    def powers(self):
        return list(self._dss.ActiveCircuit.ActiveCktElement.Powers)

    @property
    def num_conductors(self):
        return self._dss.ActiveCircuit.ActiveCktElement.NumConductors

    @property
    def bus_names(self):
        return list(self._dss.ActiveCircuit.ActiveCktElement.BusNames)


class _CircuitProxy:
    __slots__ = ("_dss",)

    def __init__(self, dss):
        self._dss = dss

    @property
    def num_buses(self):
        return self._dss.ActiveCircuit.NumBuses

    def set_active_bus_i(self, i):
        return self._dss.ActiveCircuit.SetActiveBusi(i)

    def set_active_element(self, name):
        return self._dss.ActiveCircuit.SetActiveElement(name)


class _IterableElementProxy:
    __slots__ = ("_dss", "_iface_name")

    def __init__(self, dss, iface_name):
        self._dss = dss
        self._iface_name = iface_name

    @property
    def _iface(self):
        return getattr(self._dss.ActiveCircuit, self._iface_name)

    def first(self):
        return self._iface.First

    def next(self):
        return self._iface.Next

    @property
    def name(self):
        return self._iface.Name

    @property
    def phases(self):
        return self._iface.Phases


class _SolutionProxy:
    __slots__ = ("_dss",)

    def __init__(self, dss):
        self._dss = dss

    @property
    def converged(self):
        return self._dss.ActiveCircuit.Solution.Converged

    @property
    def iterations(self):
        return self._dss.ActiveCircuit.Solution.Iterations


class DSSPythonAdapter:
    """Drop-in replacement for ``py_dss_interface.DSS`` backed by DSS-Python."""

    def __init__(self):
        from dss import DSS

        self._dss = DSS
        self.bus = _BusProxy(self._dss)
        self.cktelement = _CktElementProxy(self._dss)
        self.circuit = _CircuitProxy(self._dss)
        self.lines = _IterableElementProxy(self._dss, "Lines")
        self.transformers = _IterableElementProxy(self._dss, "Transformers")
        self.loads = _IterableElementProxy(self._dss, "Loads")
        self.generators = _IterableElementProxy(self._dss, "Generators")
        self.capacitors = _IterableElementProxy(self._dss, "Capacitors")
        self.solution = _SolutionProxy(self._dss)

    def text(self, command):
        self._dss.Text.Command = command
        return self._dss.Text.Result

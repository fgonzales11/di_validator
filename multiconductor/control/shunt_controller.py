from pandapower.control.basic_controller import Controller
import logging
import pandas as pd
logging.basicConfig(level=logging.WARNING) 
#logging.getLogger().setLevel(logging.WARNING) 

# to do: 
# check/improve runtime


class MulticonductorBinaryShuntController(Controller):
    """
    Controller for switching of asymmetric shunts bus voltages.

    The MultiPhaseBinaryShuntController operates several asymmetric shunt elements as a synchronized group,
    emulating the gang operation of a three-phase capacitor bank. The switching is performed with the
    'closed' parameter: If 'closed' is True, the shunt injects/consumes the specified 'max_q_mvar' and 'max_p_mw';
    if 'closed' is False, the shunt injects zero reactive and active power.

    The controller supports fixed mode (always closed) as well as switching operation based on voltage thresholds.
    All phases in 'shunt_indices' are always switched together (gang-operation).
    
    Parameters
    ----------
    net : object
        The pp-mc or pp-os grid with the asymmetric shunt or shunt elements (must include net.asymmetric_shunt for pp-mc or net.shunt for pp-os).
    shunt_indices : list of tuples
        MultiIndex keys of all phases of the shunt group to be controlled (e.g. [(0,0), (0,1), (0,2)]).
    bus_indices : list
        List of bus index tuples that correspond to the buses/phases being monitored for voltage magnitude.
    index_type : string (default "mc")
        Choose "mc" for mc-indexing or "pp" for pp-indexing 
    in_service : bool, optional
        Passed through to base Controller (default True).
    order : int, optional
        Controller execution order (default 0).
    level : int, optional
        Controller level (default 0).
    max_iter : int, optional
        Maximum iteration count (default 30).
    
    Behavior
    --------
    - If the average of the monitored voltages falls below v_threshold_on, the entire 3ph-shunt is "closed" (all phases set to closed=True),
      and max_q_mvar, max_p_mw are injected/consumed.
    - If the average of the monitored voltage rises above v_threshold_off, the shunt is "opened" (all phases set to closed=False), and no q/p is injected/consumed.
    - If `control_mode` is "fixed" for any group member, the shunt is always closed and injects q/p regardless of voltage.
    - The controller only manipulates the 'closed' parameter and corresponding q/p values; the physical connection is maintained.
    
    Note
    ----
    The controller assumes all phases in `shunt_indices` have the same v_threshold_on and v_threshold_off.
    """
    
    def __init__(self, net, shunt_indices, bus_indices, 
                     index_type="mc", in_service=True, order=0, level=0, max_iter=30, **kwargs):
            super().__init__(net=net, in_service=in_service, order=order, level=level, **kwargs)
            assert index_type in ["mc", "pp", "pp_3ph"], \
            "index_type must be 'mc', 'pp' or 'pp_3ph'!"
            self.logger = logging.getLogger(self.__class__.__name__)
            self.index_type = index_type
            
            self.shunt_indices = shunt_indices
            self.bus_indices = bus_indices
            self.shunt_key = "asymmetric_shunt" if index_type == 'mc' else "shunt"
            
            assert isinstance(self.bus_indices, list), "bus_indices must be a list!"
            assert len(self.shunt_indices)>0, "shunt_indices missing!"
                
            allowed_modes = ("switched", "fixed")
            for idx in self.shunt_indices:
                mode = net[self.shunt_key].loc[idx, "control_mode"]
                if mode not in allowed_modes:
                    raise ValueError(
                        f"Invalid control_mode '{mode}' for shunt at idx {idx}. "
                        f"Allowed values are {allowed_modes}.")

    def __str__(self):
        return "MulticonductorBinaryShuntController for shunt index "+str(self.shunt_indices)
    
    def _get_voltage(self, net):
        if self.index_type == "mc": #bus_idx (bus, phase)
            if not isinstance(net.res_bus.index, pd.MultiIndex) or net.res_bus.index.nlevels < 2:
                raise ValueError("For index_type='mc', net.res_bus must have (bus, phase).")
            v = [net.res_bus.at[bus_idx, "vm_pu"] for bus_idx in self.bus_indices]  #bus_idx: (bus, terminal)
        elif self.index_type == "pp":
            v = [net.res_bus.at[bus_idx, "vm_pu"] for bus_idx in self.bus_indices]  #bus_idx: int
        elif self.index_type == "pp_3ph":
            phases = ('a', 'b', 'c')
            v = [net.res_bus_3ph.at[bus_idx, f"vm_{ph}_pu"]
                for bus_idx in self.bus_indices
                for ph in phases]   #bus_idx: int
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")
        return v
    

    def _get_avg_voltage(self, net):
        v = self._get_voltage(net)
        v_avg = sum(v) / len(v)
        return v_avg
            
    
    def _update_p_mw_q_mvar(self, net):
        """Set q_mvar/p_mw based on each shunt's state"""
        for idx in self.shunt_indices:
            if net[self.shunt_key].loc[idx, "closed"]:
                net[self.shunt_key].loc[idx, "q_mvar"] = net[self.shunt_key].loc[idx, "max_q_mvar"]
                net[self.shunt_key].loc[idx, "p_mw"] = net[self.shunt_key].loc[idx, "max_p_mw"]
            else:
                net[self.shunt_key].loc[idx, "q_mvar"] = 0
                net[self.shunt_key].loc[idx, "p_mw"] = 0


    def initialize_control(self, net):
        self._update_p_mw_q_mvar(net)


    def _fixed_mode(self, net):
        """Return True if in fixed mode"""
        modes = net[self.shunt_key].loc[self.shunt_indices, 'control_mode']
        return bool((modes == "fixed").any())


    def _thresholds(self, net):
        """Fetch thresholds for on and off"""
        row = net[self.shunt_key].loc[self.shunt_indices[0], ['v_threshold_on', 'v_threshold_off']]
        return float(row['v_threshold_on']), float(row['v_threshold_off'])
    

    def _get_voltage_and_shunt_status(self, net):
        vms = self._get_avg_voltage(net)
        v_thr_on, v_thr_off = self._thresholds(net)
        all_closed = all(net[self.shunt_key].loc[idx, 'closed'] for idx in self.shunt_indices)
        all_open   = not any(net[self.shunt_key].loc[idx, 'closed'] for idx in self.shunt_indices)
        return vms, v_thr_on, v_thr_off, all_closed, all_open


    def control_step(self, net):
        if self._fixed_mode(net):
            for idx in self.shunt_indices:
                net[self.shunt_key].loc[idx, 'closed'] = True
            self._update_p_mw_q_mvar(net)
            return
        vms, v_thr_on, v_thr_off, all_closed, all_open = self._get_voltage_and_shunt_status(net)
        
        if all_closed and vms >= v_thr_off:
            for idx in self.shunt_indices:
                net[self.shunt_key].loc[idx, 'closed'] = False
            self.logger.info(f"Shunt opened by threshold: vms_pu={[round(float(vms),4)]}")
        elif all_open and vms <= v_thr_on:
            for idx in self.shunt_indices:
                    net[self.shunt_key].loc[idx, 'closed'] = True
            self.logger.info(f"Shunt closed by threshold: vms_pu={[round(float(vms),4)]}")

        self._update_p_mw_q_mvar(net)


    def is_converged(self, net):
        if self._fixed_mode(net):
            self.logger.info("Converged - fixed mode, shunt is always connected")
            return True
        vms, v_thr_on, v_thr_off, all_closed, all_open = self._get_voltage_and_shunt_status(net)
        in_voltage_range = v_thr_on < vms < v_thr_off
    
        if in_voltage_range:
            #self.logger.info("Converged - all observed voltages within range: "
            #                 f"vms_pu={[round(float(vms),4)]}")
            return True
        if all_open and not vms <= v_thr_on:
            #self.logger.info("Converged - but all shunts open and observed voltages above threshold: "
            #                 f"vms_pu={[round(float(vms),4)]}")
            return True
        if all_closed and not vms >= v_thr_off:
            #self.logger.info("Converged - but all shunts closed and observed voltages below threshold: "
            #                 f"vms_pu={[round(float(vms),4)]}")

            return True
        return False

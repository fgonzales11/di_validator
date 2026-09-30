

import numpy as np
import math as m
from pandapower.auxiliary import read_from_net, write_to_net
from pandapower.control.basic_controller import Controller

import logging
logger = logging.getLogger(__name__)
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings("ignore", category=PerformanceWarning)


# -------------------------------------------------------------------------------------------------
# Tools
# -------------------------------------------------------------------------------------------------


def check_ldc_stable_state(mc_net, controlled_bus_index, controlled_trafo_index, voltage_bandwidth, mode, side="lv"):
    """
    Check transformer tap settings vs. voltage at controlled bus after controller action
    
    INPUT:
        **net** (pandapower-mc net)

        **controlled_bus_index** (int) - First-Level index of the controlled bus  
        
        **controlled_trafo_index** (int) - First-level index of the transformer 
    
        **voltage_bandwidth** (list of float) - Voltage tolerance band for LDC, in p.u.
        
        **mode** (String) - controller operation mode. Allowed values: "locked_forward", "locked_reverse" or "bidirectional"
        
    OPTIONAL:    
        **side** (String, default "lv") - Trafo side of the tap control. Allowed values: "lv", "hv". 
        
    """
    tap_pos = mc_net.trafo1ph.loc[controlled_trafo_index,1 if side=="lv" else 0]['tap_pos']
    tap_min = mc_net.trafo1ph.loc[controlled_trafo_index,1 if side=="lv" else 0]['tap_min']
    tap_max = mc_net.trafo1ph.loc[controlled_trafo_index,1 if side=="lv" else 0]['tap_max']
    tap_saturation = ((tap_pos==tap_min)|(tap_pos==tap_max))        
    vm_pu_ctr = mc_net.res_bus.loc[controlled_bus_index][1:]['vm_pu']
    voltage_violations = ((vm_pu_ctr<voltage_bandwidth[0]) | (vm_pu_ctr>voltage_bandwidth[1]))        
    P_trafo = mc_net.res_trafo.loc[controlled_trafo_index,1]['p_mw']
    if sum(P_trafo)>0 and mode=="locked_forward" or sum(P_trafo)<0 and mode=="locked_reverse":
        return all(tap_pos==0)
    else:
        return ~np.any(voltage_violations.values & ~tap_saturation.values)
   
   
def add_line_drop_control(mc_net, trafo_top_level_index, mode, v_set_secondary_v, bandwidth_secondary_v, pt_ratio, 
                                       ct_primary_rating_a, r_ldc_v, x_ldc_v, phasemode='phase', detect_oscillation=False):
    """
    Add Line drop controller to a pp-mc transformer 
    
    INPUT:
        **mc_net** (pandapower-mc net)

        **trafo_top_level_index** (int) - First-level index of the transformer 
    
        **mode** (String) - controller operation mode. Allowed values: "locked_forward", "locked_reverse" or "bidirectional"
        
        **v_set_secondary_v** (float) - Secondary voltage setpoint for LDC
        
        **bandwidth_secondary_v** (float) - Voltage tolerance bandwidth for LDC 
        
        **pt_ratio** (float) - Ratio of the potential transformer
        
        **ct_primary_rating_a** (float) - Ratio of primary current transformer winding
        
        **r_ldc_v** (float) - Resistance setting for LDC
        
        **x_ldc_v** (float) - Reactance setting for LDC    
    
    OPTIONAL:
        **phasemode** (String, 'phase') - Controller operation mode. Allowed: 'gang' or 'phase'. 
        
        **detect_oscillation** (boolean, default False) - If False, controller will not converge if oscillating beween two tap settings. 
                                                          If True, controller will converge as soon as oscillation is detected 
                                                          but voltages may be outside allowed bandwidth
        
    """
    
    LineDropControlExtended(
        net=mc_net,
        trafo_top_level_index=trafo_top_level_index,
        mode = mode,        
        v_set_secondary_v=v_set_secondary_v,
        bandwidth_secondary_v=bandwidth_secondary_v,        
        pt_ratio=pt_ratio,
        ct_primary_rating_a=ct_primary_rating_a,
        r_ldc_v=r_ldc_v,
        x_ldc_v=x_ldc_v,
        loc_sen=False,
        phasemode=phasemode,
        detect_oscillation=detect_oscillation
    )
    

class McTrafoController(Controller):
    """
    Multiconductor Trafo Controller with local tap ch
    Based on pandapower Trafo Controller

    INPUT:
       **net** (attrdict) - Pandapower-multiconductor struct
       
       **trafo_top_level_index** (int) - First-level index of the transformer     

       **side** (string) - Side of the transformer where the voltage is controlled ("hv", "mv"
       or "lv")

       **tol** (float) - Voltage tolerance band at bus in Percent

       **in_service** (bool) - Indicates if the element is currently active

       **element** (string) - Type of the controlled trafo ("trafo" or "trafo3w")            

    OPTIONAL:
        **level** (int, default 0) - Controller hierarchy level, see pandapower Controller documentation 
       
        **order** (int, default 0) - Controller hierarchy order, see pandapower Controller documentation
        
        **recycle** (bool, True) - Re-use of internal-data in a time series loop.
    """

    def __init__(self, net, trafo_top_level_index, side, tol, in_service, element, level=0, order=0,
                 recycle=True, **kwargs):
        super().__init__(net, in_service=in_service, level=level, order=order, recycle=recycle,
                         **kwargs)

        self.element = element
        self.trafo_top_level_index = trafo_top_level_index

        self._set_side(side)
        self.tap_side_bus, self.tap_phases = self._get_trafo_tapside_bus_idx(net, self.side)
        self._set_tap_parameters(net)
        self._set_tap_side_coeff(net)              
        self.tol = tol
        self.set_recycle(net)

        
    def _get_trafo_tapside_bus_idx(self, net, trafo_side):
        if trafo_side=='hv':
            side_idx = net[self.element].loc[self.trafo_top_level_index].index.unique(level='bus')[0]
        else: 
            side_idx = net[self.element].loc[self.trafo_top_level_index].index.unique(level='bus')[1]
        phases= np.array(net[self.element].loc[self.trafo_top_level_index, side_idx]['from_phase'])
        return side_idx, phases
        
    def _read_from_trafo1ph(self, net, trafo_side, elt_name):
        side_idx = self.tap_side_bus
        windings = net[self.element].loc[(self.trafo_top_level_index,side_idx)].index
        retval = []
        for w in list(windings):
            retval += [net[self.element].loc[(self.trafo_top_level_index,side_idx,w)][elt_name]]
        return np.array(retval)
    
    def _write_to_trafo1ph(self, net, trafo_side, elt_name, values):
        side_idx = self.tap_side_bus
        windings = net[self.element].loc[(self.trafo_top_level_index,side_idx)].index
        net[self.element].loc[(self.trafo_top_level_index,side_idx,windings),elt_name]=values

    def initialize_control(self, net):
        # in case changes applied to net in the meantime:
        # the update occurs in case the in_service parameter of tranformers is changed in the meantime
        # update valid trafo and bus
        # update trafo tap parameters
        # we assume side does not change after the controller is created                
        if self.nothing_to_do(net):
            return
        self._set_tap_parameters(net)
        self._set_tap_side_coeff(net)
    
    def nothing_to_do(self, net):        
        # Set self_controlled=True if trafo is in the network, in service, and the tap side is NOT cpnnected to an external grid 
        element_in_service = any(self._read_from_trafo1ph(net, self.side, 'in_service'))
        ext_grid_buses = np.concatenate([net.ext_grid.loc[net.ext_grid.in_service, 'bus'].values,
                                         net.ext_grid_sequence.loc[net.ext_grid_sequence.in_service, 'bus'].values])        
        tap_side_is_ext_grid_bus = np.isin(self.tap_side_bus, ext_grid_buses)
        trafo_index_in_net = np.isin(self.trafo_top_level_index, net[self.element].index.get_level_values(0))        
        self.controlled = np.logical_and(np.logical_and(element_in_service, trafo_index_in_net),
                                         np.logical_not(tap_side_is_ext_grid_bus))
        
        if isinstance(self.trafo_top_level_index, np.int64) or isinstance(self.trafo_top_level_index, int):
            # if the controller shouldn't do anything, return True
            if not element_in_service or tap_side_is_ext_grid_bus or not trafo_index_in_net:
                return True
            return False
        else:
            # if the controller shouldn't do anything, return True
            if np.all(~element_in_service[self.controlled]) or np.all(tap_side_is_ext_grid_bus[self.controlled]) or np.all(
                    ~trafo_index_in_net[self.controlled]):
                return True
            return False

    def _set_tap_side_coeff(self, net):            
        self.tap_side_coeff = 1 if self.side == 'hv' else -1
        
    def _set_side(self, side):        
        if side not in ["hv", "lv"]:
            raise UserWarning("side has to be 'hv' or 'lv' for high/low voltage, "
                              "received %s" % side)        
        self.side = side

    def _set_valid_controlled_index_and_bus(self, net):        
        element_in_service = self._read_from_trafo1ph(net, self.side, 'in_service')
        ext_grid_buses = np.concatenate([net.ext_grid.loc[net.ext_grid.in_service, 'bus'].values,
                                         net.ext_grid_sequence.loc[net.ext_grid_sequence.in_service, 'bus'].values])        
        tap_side_is_ext_grid_bus = np.isin(self.tap_side_bus, ext_grid_buses)
        trafo_index_in_net = np.isin(self.trafo_top_level_index, net[self.element].index.get_level_values(0))    
        
        self.controlled = np.logical_and(np.logical_and(element_in_service, trafo_index_in_net),
                                         np.logical_not(tap_side_is_ext_grid_bus))        

        if np.all(~self.controlled):
            logger.warning("All controlled buses are not valid: controller has no effect")

    def _set_tap_parameters(self, net):   
        self.tap_min = self._read_from_trafo1ph(net, self.side, "tap_min")
        self.tap_max = self._read_from_trafo1ph(net, self.side, "tap_max")
        self.tap_neutral = self._read_from_trafo1ph(net, self.side, "tap_neutral")
        self.tap_step_percent = self._read_from_trafo1ph(net, self.side, "tap_step_percent")
        self.tap_step_degree = self._read_from_trafo1ph(net, self.side, "tap_step_degree")
        self.tap_pos = self._read_from_trafo1ph(net, self.side, "tap_pos")
        
        self.tap_sign = np.where(np.isnan(self.tap_step_degree), 1,
                                 np.sign(np.cos(np.deg2rad(self.tap_step_degree))))
        self.tap_sign = np.where((self.tap_sign == 0) | (np.isnan(self.tap_sign)), 1, self.tap_sign)
        self.tap_pos = np.where(np.isnan(self.tap_pos), self.tap_neutral, self.tap_pos)

        if np.any(np.isnan(self.tap_min)) or np.any(np.isnan(self.tap_max)) or np.any(np.isnan(self.tap_step_percent)):
            logger.error("Trafo-Controller has been initialized with NaN values, check "
                         "net.trafo.tap_pos etc. if they are set correctly!")

    def set_recycle(self, net):
        allowed_elements = ["trafo1ph"]
        if net.controller.at[self.index, 'recycle'] is False or self.element not in allowed_elements:
            # if recycle is set to False by the user when creating the controller it is deactivated or when
            # const control controls an element which is not able to be recycled
            net.controller.at[self.index, 'recycle'] = False
            return
        # these variables determine what is re-calculated during a time series run
        recycle = dict(trafo=True, gen=False, bus_pq=False)
        net.controller.at[self.index, 'recycle'] = recycle

    # def timestep(self, net):
    #     self.tap_pos = net[self.element].at[self.trafo_top_level_index, "tap_pos"]

    def __repr__(self):
        s = '%s of %s %s' % (self.__class__.__name__, self.element, self.trafo_top_level_index)
        return s

    def __str__(self):
        s = '%s of %s %s' % (self.__class__.__name__, self.element, self.trafo_top_level_index)
        return s

class LineDropControl(McTrafoController):
    """
    Trafo Controller with local tap changer voltage control.

    INPUT:
        **net** (pandapower-mc net)

        **trafo_top_level_index** (int) - ID of the trafo that is controlled
        
        **mode** (string) - Sensing mode. Must be one of ['locked_forward','locked_reverse', 'bidirectional']

        **vm_lower_pu** (float) - Lower voltage limit in pu

        **vm_upper_pu** (float) - Upper voltage limit in pu
        
        **vm_set_pu_val** (floct) - Initial value for voltage setpoint in p.u.
        
        **R_comp** (float) - Real part of impedance between trafo tap-controlled side and controlled bus
        
        **X_comp** (float) - Imaginary part of impedance between trafo tap-controlled side and controlled bus

    OPTIONAL:
    
        **loc_sen** (boolean, default False - If True, the trafo tap-side voltage is used for control. If False, voltage to the controlled bus is estimated. 

        **phasemode** (string, 'phase') - Controller operation mode. Allowed: 'gang' or 'phase'
        
        **detect_oscillation** (boolean, default False) - If False, controller will not converge if oscillating beween two tap settings. 
                                                          If True, controller will converge as soon as oscillation is detected 
                                                          but voltages may be outside allowed bandwidth
                                                          
        **side** (string, "lv") - Side of the transformer where the voltage is controlled (hv or lv)

        **trafotype** (float, "2W") - Trafo type ("2W" or "3W"). Note: Currently only 2W is tested.

        **tol** (float, 0.001) - Voltage tolerance band at bus in Percent (default: 1% = 0.01pu)

        **in_service** (bool, True) - Indicates if the controller is currently in_service
        
        **level** (int, default 0) - Controller hierarchy level, see pandapower Controller documentation 
       
        **order** (int, default 0) - Controller hierarchy order, see pandapower Controller documentation

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of the same type and with the same matching parameters (e.g. at same element) should be dropped
    """

    def __init__(self, net, trafo_top_level_index, mode, vm_lower_pu, vm_upper_pu, vm_set_pu_val, R_comp, X_comp, loc_sen=False, phasemode="phase", detect_oscillation=False, 
                 side="lv", trafotype="2W", tol=1e-3, in_service=True, level=0, order=0, drop_same_existing_ctrl=False,
                 matching_params=None, **kwargs):
        #if matching_params is None:
        #    matching_params = {"tid": trafo_index, 'trafotype': trafotype}
        element = "trafo1ph"
        super().__init__(net, trafo_top_level_index, side, tol=tol, in_service=in_service, element=element, level=level, order=order, 
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         **kwargs)
        
        if mode not in ['locked_forward', 'locked_reverse', 'bidirectional']:
            print("LineDropControl: mode "+str(mode)+" unsupported, using default (locked_forward)")
            self.mode = 'locked_forward'
        else:
            self.mode = mode
        if phasemode not in ['phase', 'gang']:
            print("LineDropControl: phasemode "+str(mode)+" unsupported, using default (phase)")
            self.phasemode = 'phase'
        else:
            self.phasemode = phasemode   
        self.vm_lower_pu = vm_lower_pu
        self.vm_upper_pu = vm_upper_pu        
        self.vm_delta_pu = self.tap_step_percent / 100. * .5 + self.tol # Probably expected voltage pu change per tap 
        self.vm_set_pu_val = vm_set_pu_val
        self.R_comp = R_comp
        self.X_comp = X_comp
        self.control_by_tap_side_voltage= loc_sen    
        if not hasattr(self, 'vn_kv') or self.vn_kv is None:
            # get controlled side side nominal bus voltages  
            trafoidx = trafo_top_level_index        
            busidx = self.tap_side_bus
            phases = self.tap_phases 
            self.vn_kv = np.array(net.bus.loc[busidx]['vn_kv'][phases])   
            if (net.trafo1ph.loc[trafoidx,busidx,0]["from_phase"] == 0) | (net.trafo1ph.loc[trafoidx,busidx,0]["to_phase"] == 0):
                self.vn_kv /= np.sqrt(3)
            
        self.detect_oscillation = detect_oscillation
        self.oscillating = False 
            
    def __str__(self):
        return "LineDropControl for "+str(self.element)+" index "+str(self.element_index)
        

    @classmethod
    def from_tap_step_percent(cls, net, tid, vm_set_pu, side="lv", trafotype="2W", tol=1e-3, in_service=True, order=0,
                              drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        """
        Alternative mode of the controller, which uses a set point for voltage and the value of net.trafo.tap_step_percent to calculate
        vm_upper_pu and vm_lower_pu. To this end, the parameter vm_set_pu should be provided, instead of vm_lower_pu and vm_upper_pu.
        To use this mode of the controller, the controller can be initialized as following:

        >>> c = DiscreteTapControl.from_tap_step_percent(net, tid, vm_set_pu)

        INPUT:
            **net** (attrdict) - Pandapower struct

            **tid** (int) - ID of the trafo that is controlled

            **vm_set_pu** (float) - Voltage setpoint in pu
        """
        self = cls(net, tid=tid, vm_lower_pu=None, vm_upper_pu=None, side=side, trafotype=trafotype, tol=tol,
                   in_service=in_service, order=order, drop_same_existing_ctrl=drop_same_existing_ctrl,
                   matching_params=matching_params, vm_set_pu=vm_set_pu, **kwargs)
        return self

    @property
    def vm_set_pu(self):
        return self._vm_set_pu
    
    @property
    def print_volt_set_120(self):
        print(self.vm_set_pu_val*120)

    def get_vm_set_pu(self, value):
        self._vm_set_pu = value
        if value is None:
            return
        self.vm_lower_pu = value - self.vm_delta_pu
        self.vm_upper_pu = value + self.vm_delta_pu        

    def initialize_control(self, net):
        super().initialize_control(net)
        if hasattr(self, 'vm_set_pu') and self.vm_set_pu is not None:
            self.vm_delta_pu = self.tap_step_percent / 100. * .5 + self.tol
        self.tap_pos_prev1 = None
        self.tap_pos_prev2 = None        
        self.oscillating = False         

    def control_step(self, net):
        
        """
        Implements one step of the Discrete controller, always stepping only one tap position up or down
        """
        if self.nothing_to_do(net):
            return
        
        I_prim_ka, V_prim_kv, V_L_pu, pf_fwd = self._estimate_IV(net)
        if (self.mode=="locked_forward" and not pf_fwd) or (self.mode=="locked_reverse" and pf_fwd): 
            return True        
        vm_pu = self.estimated_voltage(net, I_prim_ka, V_prim_kv, V_L_pu)
        
        self.tap_pos = self._read_from_trafo1ph(net, self.side, "tap_pos")        
        self.tap_pos_prev2 = np.copy(self.tap_pos_prev1)
        self.tap_pos_prev1 = np.copy(self.tap_pos)
        
        # print('self tap postion:',self.tap_pos)
        
        #Iout = read_from_net(net, "res_trafo", self.tap_side_bus, "i_lv_ka", self._read_write_flag)
        #Iout=net.res_trafo.loc[self.controlled_tid,'i_lv_ka']
        #print(Iout)
        #voltage angle at the controll bus
        
        if not hasattr(self, "phasemode") or self.phasemode == "phase":
            increment = np.where(self.tap_side_coeff * self.tap_sign == 1,
                                 np.where(np.logical_and(vm_pu < self.vm_lower_pu, self.tap_pos > self.tap_min), -1,
                                          np.where(np.logical_and(vm_pu > self.vm_upper_pu, self.tap_pos < self.tap_max), 1, 0)),
                                 np.where(np.logical_and(vm_pu < self.vm_lower_pu, self.tap_pos < self.tap_max), 1,
                                          np.where(np.logical_and(vm_pu > self.vm_upper_pu, self.tap_pos > self.tap_min), -1, 0)))
        else:
            increment = np.where(self.tap_side_coeff * self.tap_sign == 1,
                                 np.where( (min(vm_pu) < np.average(self.vm_lower_pu) and self.tap_pos > self.tap_min), -1,
                                           np.where((max(vm_pu) > np.average(self.vm_upper_pu) and self.tap_pos < self.tap_max), 1, 0)),
                                 np.where( (min(vm_pu) < np.average(self.vm_lower_pu) and self.tap_pos < self.tap_max),1,
                                           np.where((max(vm_pu) > np.average(self.vm_upper_pu) and self.tap_pos > self.tap_min), -1, 0)))       
            
        self.tap_pos += increment
        if self.tap_pos_prev2 is not None and all(self.tap_pos != 0) and all(self.tap_pos==self.tap_pos_prev2):
            print("WARNING: LineDropControl for trafo index "+str(self.trafo_top_level_index)+" was found oscillating.")
            self.oscillating = True 
        self._write_to_trafo1ph(net, self.side, 'tap_pos', self.tap_pos)

    def is_converged(self, net):
        """
        Checks if the voltage is within the desired voltage band, then returns True
        """
        if self.nothing_to_do(net):
            return True

        if self.oscillating and self.detect_oscillation:
            return True 
        
        #vm_pu = read_from_net(net, "res_bus", self.tap_side_bus, "vm_pu", self._read_write_flag)
        I_prim_ka, V_prim_kv, V_L_pu, pf_fwd = self._estimate_IV(net)        
        if (self.mode=="locked_forward" and not pf_fwd) or (self.mode=="locked_reverse" and pf_fwd): 
            return True
        vm_pu = self.estimated_voltage(net, I_prim_ka, V_prim_kv, V_L_pu)
                        
        self.tap_pos = self._read_from_trafo1ph(net, self.side, "tap_pos")        
        
        reached_limit = np.where(self.tap_side_coeff * self.tap_sign == 1,
                                 (vm_pu < self.vm_lower_pu) & (self.tap_pos == self.tap_min) |
                                 (vm_pu > self.vm_upper_pu) & (self.tap_pos == self.tap_max),
                                 (vm_pu < self.vm_lower_pu) & (self.tap_pos == self.tap_max) |
                                 (vm_pu > self.vm_upper_pu) & (self.tap_pos == self.tap_min))

        converged = np.all(np.logical_or(reached_limit, np.logical_and(self.vm_lower_pu < vm_pu, vm_pu < self.vm_upper_pu)))
        
        #if converged:
        #    print("Trafo tap control converged at tap position:",self.tap_pos)
        #    print("Estimated voltage p.u. at controlled bus:", vm_pu)
        
        return converged

    def _estimate_IV(self,net)->float:
    # get reactive power, low side voltage, low side voltage angle in degree, low side current 
    # from power flow result
    # Q in Mvar    
                
        idx3 = net[self.element].loc[self.trafo_top_level_index].index.unique(level='bus')[1]     
        Qinput = net.res_trafo.loc[(self.trafo_top_level_index,idx3),'q_mvar']
        Pinput = net.res_trafo.loc[(self.trafo_top_level_index,idx3),'p_mw']
        Sinput = (Pinput + 1j * Qinput).values
        pf_fwd = sum(Sinput).real < 0 
       
        # change the sign for reactive power inject to the grid
        V_L_pu = read_from_net(net, "res_bus", self.tap_side_bus, "vm_pu", "loc")
        V_L_deg = read_from_net(net, "res_bus", self.tap_side_bus, "va_degree", "loc")
        V_L_deg = V_L_deg[self.tap_phases] 
        V_L_pu = V_L_pu[self.tap_phases]    

        # I_prim_ka = np.array(net.res_trafo.loc[(self.trafo_top_level_index,1),'i_ka']) # this is the worst case current, in this case the HV current!!!
        V_prim_kv = self.vn_kv*V_L_pu * (np.cos(np.deg2rad(V_L_deg)) + 1j * np.sin(np.deg2rad(V_L_deg)))
        I_prim_ka = -np.conj(Sinput/(V_prim_kv/np.sqrt(3)))

        return I_prim_ka, V_prim_kv, V_L_pu, pf_fwd

    def estimated_voltage(self,net,I_prim_ka, V_prim_kv, V_L_pu)->float:    
        # print("Measured voltage at trafo tap side:",V_L_pu)
        # if control_by_tap_side_voltage is true the regulator will only take the tap-side voltage for control.
        if self.control_by_tap_side_voltage is True: 
            # print("Tap-side bus voltage will be used for control.")
            return V_L_pu 
        else:
            Z_comp = (self.R_comp + 1j * self.X_comp)
            # calculate Vdrop
            Vdrop_kv = Z_comp*I_prim_ka
            V_estimated_kv = ((V_prim_kv/np.sqrt(3))-Vdrop_kv)*np.sqrt(3)            
            return np.abs(V_estimated_kv)/self.vn_kv

class LineDropControlExtended(LineDropControl):
    """
    Trafo Controller with local tap changer voltage control and model of voltage / current sensing.

    INPUT:
        **net** (pandapower-mc net)

        **trafo_top_level_index** (int) - ID of the trafo that is controlled
        
        **mode** (string) - Sensing mode. Must be one of ['locked_forward','locked_reverse', 'bidirectional']

        **v_set_secondary_v** (float) - Initial value for absolute voltage setpoint

        **bandwidth_secondary_v** (float) - Voltage tolerance bandwidth for LDC 
        
        **pt_ratio** (floct) - Ratio of voltage potential transformer
        
        **ct_primary_rating_a** (float) - Ratio of current measurement transformer
        
        **r_ldc_v** (float) - Imaginary part of impedance between trafo tap-controlled side and controlled bus
        
        **x_ldc_v** (float) - Imaginary part of impedance between trafo tap-controlled side and controlled bus

    OPTIONAL:
    
        **loc_sen** (boolean, default False - If True, the trafo tap-side voltage is used for control. If False, voltage to the controlled bus is estimated. 

        **phasemode** (string, 'phase') - Controller operation mode. Allowed: 'gang' or 'phase'
        
        **detect_oscillation** (boolean, default False) - If False, controller will not converge if oscillating beween two tap settings. 
                                                          If True, controller will converge as soon as oscillation is detected 
                                                          but voltages may be outside allowed bandwidth
        
        **side** (string, "lv") - Side of the transformer where the voltage is controlled (hv or lv)
        
        **in_service** (bool, True) - Indicates if the controller is currently in_service
        
        **level** (int, default 0) - Controller hierarchy level, see pandapower Controller documentation 
       
        **order** (int, default 0) - Controller hierarchy order, see pandapower Controller documentation

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of the same type and with the same matching parameters (e.g. at same element) should be dropped
    """
    def __init__(self, net, trafo_top_level_index, mode, v_set_secondary_v, bandwidth_secondary_v, pt_ratio,
                 ct_primary_rating_a, r_ldc_v, x_ldc_v, loc_sen=False, phasemode='phase', detect_oscillation=False, side="lv", 
                 in_service=True, level=0, order=0, drop_same_existing_ctrl=False, **kwargs):                        
        
        super().__init__(
            net, trafo_top_level_index, mode, vm_lower_pu=1, vm_upper_pu=1, vm_set_pu_val=1,
            R_comp=r_ldc_v, X_comp=x_ldc_v, loc_sen=loc_sen, phasemode=phasemode, detect_oscillation=detect_oscillation, side=side, trafotype="2W", 
            tol=1e-3, in_service=in_service, level=level, order=order, 
            drop_same_existing_ctrl=drop_same_existing_ctrl, **kwargs
        )
        self.vm_lower_pu = (v_set_secondary_v - bandwidth_secondary_v) / 1000 * pt_ratio / self.vn_kv
        self.vm_upper_pu = (v_set_secondary_v + bandwidth_secondary_v) / 1000 * pt_ratio / self.vn_kv
        self.vm_set_pu_val = v_set_secondary_v / 1000 * pt_ratio / self.vn_kv    
        self.vm_delta_pu = self.tap_step_percent / 100. * .5 + self.tol
        self.ct_primary_rating_a = ct_primary_rating_a
        self.pt_ratio = pt_ratio
        self.r_ldc_v = r_ldc_v
        self.x_ldc_v = x_ldc_v
        
    def __str__(self):
        return "LineDropControlExtended for "+str(self.element)+" top level index "+str(self.trafo_top_level_index)

    def estimated_voltage(self, net, I_prim_ka, V_prim_kv, V_L_pu):              
                
        z_ldc_v = self.r_ldc_v + 1j * self.x_ldc_v
        V_trafo_bus_sec_kv = V_prim_kv/(self.pt_ratio*np.sqrt(3))
        V_estimated_sec_kv = V_trafo_bus_sec_kv - z_ldc_v*I_prim_ka/self.ct_primary_rating_a
        
        V_estimated = np.sqrt(3)*self.pt_ratio*np.abs(V_estimated_sec_kv)/self.vn_kv
        return V_estimated
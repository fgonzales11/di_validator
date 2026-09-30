


import numpy as np
import math as m
from pandapower.control.controller.trafo_control import TrafoController
from pandapower.toolbox import read_from_net, write_to_net

class LineDropControl(TrafoController):
    """
    Trafo Controller with local tap changer voltage control.

    INPUT:
        **net** (attrdict) - Pandapower struct

        **tid** (int) - ID of the trafo that is controlled

        **vm_lower_pu** (float) - Lower voltage limit in pu

        **vm_upper_pu** (float) - Upper voltage limit in pu

    OPTIONAL:

        **side** (string, "lv") - Side of the transformer where the voltage is controlled (hv or lv)

        **trafotype** (float, "2W") - Trafo type ("2W" or "3W")

        **tol** (float, 0.001) - Voltage tolerance band at bus in Percent (default: 1% = 0.01pu)

        **in_service** (bool, True) - Indicates if the controller is currently in_service

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of the same type and with the same matching parameters (e.g. at same element) should be dropped
    """

    def __init__(self, net, tid, vm_lower_pu, vm_upper_pu, vm_set_pu_val,CT, PT,R_comp,X_comp,loc_sen=False, side="lv", trafotype="2W",
                 tol=1e-3, in_service=True, level=0, order=0, drop_same_existing_ctrl=False,
                 matching_params=None, **kwargs):
        if matching_params is None:
            matching_params = {"tid": tid, 'trafotype': trafotype}
        super().__init__(net, tid, side, tol=tol, in_service=in_service, level=level, order=order, trafotype=trafotype,
                         drop_same_existing_ctrl=drop_same_existing_ctrl, matching_params=matching_params,
                         **kwargs)

        self.vm_lower_pu = vm_lower_pu
        self.vm_upper_pu = vm_upper_pu

        self.vm_delta_pu = self.tap_step_percent / 100. * .5 + self.tol
        self.vm_set_pu = kwargs.get("vm_set_pu")
        self.vm_set_pu_val = vm_set_pu_val
        self.CT = CT
        self.PT = PT
        self.R_comp = R_comp
        self.X_comp = X_comp
        self.control_by_tap_side_voltage= loc_sen

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

    @vm_set_pu.setter
    def vm_set_pu(self, value):
        self._vm_set_pu = value
        if value is None:
            return
        self.vm_lower_pu = value - self.vm_delta_pu
        self.vm_upper_pu = value + self.vm_delta_pu

    def initialize_control(self, net):
        super().initialize_control(net)
        if hasattr(self, 'vm_set_pu') and self.vm_set_pu is not None:
            self.vm_delta_pu = self.tap_step_percent / 100. * .5 + self.tol

    def control_step(self, net):
        
        # print('CT and PT')
        # print(self.CT)
        # print(self.PT)
        # print('R and X')
        # print(self.R_comp)
        # print(self.X_comp)
        """
        Implements one step of the Discrete controller, always stepping only one tap position up or down
        """
        if self.nothing_to_do(net):
            return
        
        #vm_pu = read_from_net(net, "res_bus", self.tap_side_bus, "vm_pu", self._read_write_flag)
        vm_pu = self.line_drop_voltage(net)
        self.tap_pos = read_from_net(net, self.trafotable, self.controlled_tid, "tap_pos", self._read_write_flag)
        
        print('self tap postion:',self.tap_pos)
        print("contoller recerived voltage:",vm_pu)
        #Iout = read_from_net(net, "res_trafo", self.tap_side_bus, "i_lv_ka", self._read_write_flag)
        #Iout=net.res_trafo.loc[self.controlled_tid,'i_lv_ka']
        #print(Iout)
        #voltage angle at the controll bus
        
        increment = np.where(self.tap_side_coeff * self.tap_sign == 1,
                             np.where(np.logical_and(vm_pu < self.vm_lower_pu, self.tap_pos > self.tap_min), -1,
                                      np.where(np.logical_and(vm_pu > self.vm_upper_pu, self.tap_pos < self.tap_max), 1, 0)),
                             np.where(np.logical_and(vm_pu < self.vm_lower_pu, self.tap_pos < self.tap_max), 1,
                                      np.where(np.logical_and(vm_pu > self.vm_upper_pu, self.tap_pos > self.tap_min), -1, 0)))
        
        self.tap_pos += increment
        print("new tap postion:",self.tap_pos)
        # WRITE TO NET
        write_to_net(net, self.trafotable, self.controlled_tid, 'tap_pos', self.tap_pos, self._read_write_flag)

    def is_converged(self, net):
        """
        Checks if the voltage is within the desired voltage band, then returns True
        """
        if self.nothing_to_do(net):
            return True

        #vm_pu = read_from_net(net, "res_bus", self.tap_side_bus, "vm_pu", self._read_write_flag)
        vm_pu = self.line_drop_voltage(net)
        
        print("check converge voltage:", vm_pu)
        self.tap_pos = read_from_net(net, self.trafotable, self.controlled_tid, "tap_pos", self._read_write_flag)
        
        reached_limit = np.where(self.tap_side_coeff * self.tap_sign == 1,
                                 (vm_pu < self.vm_lower_pu) & (self.tap_pos == self.tap_min) |
                                 (vm_pu > self.vm_upper_pu) & (self.tap_pos == self.tap_max),
                                 (vm_pu < self.vm_lower_pu) & (self.tap_pos == self.tap_max) |
                                 (vm_pu > self.vm_upper_pu) & (self.tap_pos == self.tap_min))

        converged = np.logical_or(reached_limit, np.logical_and(self.vm_lower_pu < vm_pu, vm_pu < self.vm_upper_pu))

        return np.all(converged)

    def line_drop_voltage(self,net)->float:
    # get reactive power, low side voltage, low side voltage angle in degree, low side current 
    # from power flow result
    # Q in Mvar
            
        Qinput = net.res_trafo.loc[self.controlled_tid,'q_lv_mvar']
        # change the sign for reactive power inject to the grid
        Q = -Qinput
        V_pu_input = read_from_net(net, "res_bus", self.tap_side_bus, "vm_pu", self._read_write_flag)
        print("measured voltage at low side:",V_pu_input)
        # if control_by_tap_side_voltage is true the regulator will only take the low-side voltage for control.
        if self.control_by_tap_side_voltage is True:
            print("calculated output voltage:", V_pu_input)
            return V_pu_input 
        else:
            #change to low side kV Unit kV
            LV_volt_rateing_kV = net.trafo.loc[self.controlled_tid,'vn_lv_kv']
            Vout_kv = V_pu_input*LV_volt_rateing_kV
            #change to 120v base Unit Volt
            Vout = (V_pu_input*LV_volt_rateing_kV*1000)/(m.sqrt(3)*self.PT)
            #print(Vout)
            Vout_degree = read_from_net(net, "res_bus", self.tap_side_bus, "va_degree", self._read_write_flag)
            #change from kA to A
            Iout_ka = net.res_trafo.loc[self.controlled_tid,'i_lv_ka']
            Iout = Iout_ka*1000
            print('The current is',Iout)
            # calculate the angle difference between voltage and current
            delta_ang = self.radians_to_degree(m.asin(Q/(m.sqrt(3)*Vout_kv*Iout_ka)))
            # calculate the current angle
            Iout_degree = Vout_degree - delta_ang
            #Iout_degree = -35
            # calculate the angle for input Z = R_comp + j*X_comp = Z_comp Zdegree
            Z_comp = m.sqrt(self.R_comp**2+self.X_comp**2)
            Z_degree = self.radians_to_degree(m.atan(self.X_comp/self.R_comp))
            # calculate the Ict magnitude
            Ict = Iout/self.CT
            #Ict = 0.591
            # calculate Vdrop
            Vdrop_real = Z_comp*Ict*m.cos(self.degree_to_radians(Iout_degree+Z_degree))
            Vdrop_img =  Z_comp*Ict*m.sin(self.degree_to_radians(Iout_degree+Z_degree))
            
            # calculate Vrr real and image parts
            Vrr_real = Vout*m.cos(self.degree_to_radians(Vout_degree))-Vdrop_real
            Vrr_img = Vout*m.sin(self.degree_to_radians(Vout_degree))-Vdrop_img
            
            # calculate the Vrr magnitude
            
            Vrr_mag = m.sqrt(Vrr_real**2+Vrr_img**2) 
            
            print("calculated output voltage Vrr_mag:", Vrr_mag/120)
            return Vrr_mag/120
        

    
    def radians_to_degree(self,rad:float)->float:
        return rad*(180/m.pi)

    def degree_to_radians(self,degree:float)->float:
        return degree*(m.pi/180)
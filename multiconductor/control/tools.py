# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from math import sqrt

import logging
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Tools
# -------------------------------------------------------------------------------------------------

def convert_v_p2g_p2p(v_3ph):
    """ Converts complex phase-to-ground or phase_to_neutral voltage vector in phase-to-phase  
    
         INPUT:
         **v_3ph (2D list)** - 3x2 array/list, where lines represent phases 1..3 and columns represent voltage p.u. and angle in degrees
                               NOTE: voltages may be phase-to-neutral or phase-to-ground.     
         RETURN:
         **v_p2p (2D list)** - 3x2 array/list, representing phase-to-phase voltages. Rows represent L1-L2, L2-L3 and L3-L1.
                               Values represent vm_pu relative and angle in degreed.
    """
    # Parameter checks
    if not isinstance(v_3ph, list) or np.array(v_3ph).shape != (3,2):
        logger.warning("Input must be a 2D 3x2 list")
        print("Input must be a 2D 3x2 list")
        return None        
         
    
    s = 1/sqrt(3)
    # convert v_3ph to complex 
    U_1 = v_3ph[0][0]*s*np.exp(1j * v_3ph[0][1]*(np.pi/180))
    U_2 = v_3ph[1][0]*s*np.exp(1j * v_3ph[1][1]*(np.pi/180))
    U_3 = v_3ph[2][0]*s*np.exp(1j * v_3ph[2][1]*(np.pi/180))
    
    # calculate phase-to-phase voltages
    U_12 = U_1-U_2
    U_23 = U_2-U_3
    U_31 = U_3-U_1
    
    # convert to vm_pu / angle in degrees list 
    U_p2p = [[np.abs(U_12), np.angle(U_12)*(180/np.pi)],
             [np.abs(U_23), np.angle(U_23)*(180/np.pi)],
             [np.abs(U_31), np.angle(U_31)*(180/np.pi)]]
    
    return U_p2p

def __print_header(header_printed):
    if not header_printed:
        print("Controller status:")
    return True

def print_controller_status(mc_net, also_converged=False):
    """ Converts complex phase-to-ground or phase_to_neutral voltage vector in phase-to-phase
        
        INPUT:    
            mc_net (pp-mc net) - pp-mc network with controllers 
        
        OPTIONAL:
            also_converged (boolean, default True) - if True, also prints controllers that are converged. If False, only non-converging controllers are printed.  
        
    """
    if not hasattr(mc_net,"controller") or len(mc_net.controller)==0:
        print("Controller status: net has no controllers.")
        return 
    if not hasattr(mc_net,"res_bus") or mc_net['res_bus'].isna().values.all():
        print("Controller status: no load flow results available.")
        return 
    header_printed = False
    all_converged = True
    for i, ctr in mc_net.controller.iterrows():
        if not mc_net.controller.at[i,'in_service']:
            header_printed = __print_header(header_printed)                
            print(ctr.object.__str__()+" (controller index "+str(i)+") is out of service.")       
            continue 
        if ctr.object.is_converged(mc_net):            
            if also_converged:
                header_printed = __print_header(header_printed)
                if hasattr(ctr.object, 'oscillating') and ctr.object.oscillating:
                    print(ctr.object.__str__()+" (controller index "+str(i)+") converged but oscillates.")
                else:
                    print(ctr.object.__str__()+" (controller index "+str(i)+") converged.")
        else:
            header_printed = __print_header(header_printed) 
            all_converged=False     
            if hasattr(ctr.object, 'oscillating'):
                if ctr.object.oscillating:
                    print(ctr.object.__str__()+" (controller index "+str(i)+") did not converge - is oscillating.")
                else:
                    print(ctr.object.__str__()+" (controller index "+str(i)+") did not converge - is not oscillating.")
            else: 
                print(ctr.object.__str__()+" (controller index "+str(i)+") did not converge.")

    if not also_converged and all_converged:
        print("Controller status: all converged.")
        for i, ctr in mc_net.controller.iterrows():
            if hasattr(ctr.object, 'oscillating') and ctr.object.oscillating:
                print(ctr.object.__str__()+" (controller index "+str(i)+") converged but oscillates.")
            
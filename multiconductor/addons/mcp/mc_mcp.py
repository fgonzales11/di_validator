from typing import Dict, List, Optional, Tuple, Any, Union
import sys
import os
import logging
import pickle
import numpy as np
import pandas as pd

# Ensure package root is in sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandapower as pp
from mcp.server.fastmcp import FastMCP

# Import multiconductor functions
from multiconductor.file_io import create_empty_network as create_empty_net
from multiconductor.pycci.cci_powerflow import run_pf
import multiconductor.create as mc_create

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server with logging
logger.info("Initializing Multiconductor Analysis Server")
mcp = FastMCP("Multiconductor Analysis Server")

# Global variable to store the current network
_current_net = None

def _get_network() -> pp.pandapowerNet:
    """Get the current pandapower network instance.
    
    Returns:
        pp.pandapowerNet: The current network or raises error if none loaded
    """
    global _current_net
    
    if _current_net is None:
        raise RuntimeError("No multiconductor network is currently loaded. Please create or load a network first.")
    return _current_net


def _clean_data(obj: Any) -> Any:
    """Helper to systematically clean numpy/pandas values for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _clean_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_data(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_clean_data(x) for x in obj)
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return _clean_data(obj.to_dict())
    elif isinstance(obj, np.ndarray):
        return _clean_data(obj.tolist())
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif pd.isna(obj):
        return None
    return obj


@mcp.tool()
def create_empty_network(name: str = "", sn_mva: float = 1.0, 
                         rho_ohmm: float = 100.0, f_hz: float = 50.0) -> Dict[str, Any]:
    """Create an empty multiconductor network.
    
    Args:
        name: Name for the network
        sn_mva: Reference apparent power for per unit system
        rho_ohmm: Soil resistivity in Ohm*m (used in Carson-Clem formulation)
        f_hz: Power system frequency in Hertz
        
    Returns:
        Dict containing status and network information
    """
    logger.info("Creating an empty pandapower network")
    global _current_net
    try:
        _current_net = create_empty_net(name=name, sn_mva=sn_mva, rho_ohmm=rho_ohmm, f_hz=f_hz, add_stdtypes=True)
        return {
            "status": "success",
            "message": "Empty multiconductor network created successfully",
            "network_info": {
                "buses": len(_current_net.bus),
                "lines": len(_current_net.line),
                "trafos": len(_current_net.trafo1ph) if hasattr(_current_net, "trafo1ph") else len(_current_net.trafo)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create empty multiconductor network: {str(e)}"
        }


@mcp.tool()
def load_network(file_path: str) -> Dict[str, Any]:
    """Load a multiconductor network from a file.
    
    Args:
        file_path: Path to the network file (.pkl, .p)
        
    Returns:
        Dict containing status and network information
    """
    logger.info(f"Loading network from file: {file_path}")
    global _current_net
    try:
        if file_path.endswith('.pkl'):
            _current_net = pickle.load(open(file_path, 'rb'))
        elif file_path.endswith('.p'):
            _current_net = pickle.load(open(file_path, 'rb'))
        else:
            raise ValueError("Unsupported file format. Use .pkl or .p files.")
            
        return {
            "status": "success",
            "message": f"Network loaded successfully from {file_path}",
            "network_info": {
                "buses": len(_current_net.bus),
                "lines": len(_current_net.line),
                "trafos": len(_current_net.trafo1ph) if hasattr(_current_net, "trafo1ph") else len(_current_net.trafo)
            }
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"File not found: {file_path}"
        }
    except ValueError as ve:
        return {
            "status": "error",
            "message": str(ve)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load network: {str(e)}"
        }


@mcp.tool()
def save_network(file_path: str) -> Dict[str, Any]:
    """Save the current multiconductor network to a file.
    
    Args:
        file_path: Path where the network will be saved (.pkl, .p)
        
    Returns:
        Dict containing status and confirmation message
    """
    logger.info(f"Saving network to file: {file_path}")
    try:
        net = _get_network()
        dir_name = os.path.dirname(os.path.abspath(file_path))
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        with open(file_path, 'wb') as f:
            pickle.dump(net, f)
            
        return {
            "status": "success",
            "message": f"Network saved successfully to {file_path}"
        }
    except RuntimeError as re:
        return {
            "status": "error",
            "message": str(re)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to save network: {str(e)}"
        }


@mcp.tool()
def run_power_flow(max_iteration: int = 100, tol_vmag_pu: float = 1e-6, 
                   tol_vang_rad: float = 1e-6, runControls: bool = False) -> Dict[str, Any]:
    """Run power flow analysis on the current network.
    
    Args:
        max_iteration: Maximum number of iterations
        tol_vmag_pu: Convergence tolerance for voltage magnitude in per unit
        tol_vang_rad: Convergence tolerance for voltage angle in radians
        runControls: Whether to run control elements during power flow
        
    Returns:
        Dict containing power flow results
    """
    logger.info("Running power flow analysis")
    try:
        net = _get_network()
        run_pf(net, max_iteration=max_iteration, tol_vmag_pu=tol_vmag_pu,
               tol_vang_rad=tol_vang_rad, runControls=runControls)
        
        # Extract and clean results
        results = {
            "bus_results": _clean_data(net.res_bus) if hasattr(net, "res_bus") else {},
            "line_results": _clean_data(net.res_line) if hasattr(net, "res_line") else {},
            "trafo_results": _clean_data(net.res_trafo1ph) if hasattr(net, "res_trafo1ph") else (_clean_data(net.res_trafo) if hasattr(net, "res_trafo") else {}),
            "converged": bool(net.converged)
        }
        
        return {
            "status": "success",
            "message": "Power flow calculation completed successfully" if net.converged else "Power flow did not converge",
            "results": results
        }
    except RuntimeError as re:
        return {
            "status": "error",
            "message": str(re)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Power flow calculation failed: {str(e)}"
        }


@mcp.tool()
def get_network_info() -> Dict[str, Any]:
    """Get information about the current network.
    
    Returns:
        Dict containing network statistics and information
    """
    logger.info("Retrieving network information")
    try:
        net = _get_network()
        info = {
            "buses": len(net.bus) if hasattr(net, "bus") else 0,
            "lines": len(net.line) if hasattr(net, "line") else 0,
            "trafos1ph": len(net.trafo1ph) if hasattr(net, "trafo1ph") else 0,
            "generators": len(net.asymmetric_sgen) if hasattr(net, "asymmetric_sgen") else 0,
            "loads": len(net.asymmetric_load) if hasattr(net, "asymmetric_load") else 0,
            "switches": len(net.switch) if hasattr(net, "switch") else 0,
            "bus_data": _clean_data(net.bus) if hasattr(net, "bus") else {},
            "line_data": _clean_data(net.line) if hasattr(net, "line") else {},
            "trafo1ph_data": _clean_data(net.trafo1ph) if hasattr(net, "trafo1ph") else {},
            "load_data": _clean_data(net.asymmetric_load) if hasattr(net, "asymmetric_load") else {},
            "sgen_data": _clean_data(net.asymmetric_sgen) if hasattr(net, "asymmetric_sgen") else {},
            "switch_data": _clean_data(net.switch) if hasattr(net, "switch") else {}
        }
        
        return {
            "status": "success",
            "message": "Network information retrieved successfully",
            "info": info
        }
    except RuntimeError as re:
        return {
            "status": "error",
            "message": str(re)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get network information: {str(e)}"
        }


@mcp.tool()
def get_std_types(category: Optional[str] = None) -> Dict[str, Any]:
    """Get standard types (e.g. line configs, transformer specifications) from the active database.
    
    Args:
        category: Filter by specific category ('configuration', 'matrix', 'sequence', 'trafo', 'trafo3w')
    """
    logger.info("Retrieving standard library types")
    try:
        net = _get_network()
        if not hasattr(net, "std_types"):
            return {
                "status": "success",
                "message": "No standard types library defined in current network.",
                "std_types": {}
            }
            
        if category:
            if category in net.std_types:
                data = {category: net.std_types[category]}
            else:
                return {
                    "status": "error",
                    "message": f"Category '{category}' not found in standard types. Available: {list(net.std_types.keys())}"
                }
        else:
            data = net.std_types
            
        return {
            "status": "success",
            "message": "Standard library types retrieved successfully",
            "std_types": _clean_data(data)
        }
    except RuntimeError as re:
        return {
            "status": "error",
            "message": str(re)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve standard types: {str(e)}"
        }


@mcp.tool()
def create_bus(vn_kv: float, num_phases: int = 4, name: Optional[str] = None, 
               grounded_phases: Optional[List[int]] = None, grounding_r_ohm: float = 0.0, 
               grounding_x_ohm: float = 0.0, in_service: bool = True, 
               type: str = "b", zone: Optional[str] = None, index: Optional[int] = None) -> Dict[str, Any]:
    """Create a bus in the current network.
    
    Args:
        vn_kv: Rated voltage in kV
        num_phases: Number of phases (e.g. 4)
        name: Name of the bus
        grounded_phases: Indices of phases that are grounded (e.g. [0])
        grounding_r_ohm: Resistance of the grounding connection (Ohm)
        grounding_x_ohm: Reactance of the grounding connection (Ohm)
        in_service: Whether the bus is standard in service
        type: Bus type ("b" for busbar, etc.)
        zone: Bus zone (optional)
        index: Index of the bus (optional)
    """
    logger.info(f"Creating bus: {name} (Vn={vn_kv} kV)")
    try:
        net = _get_network()
        cls_grounded = tuple(grounded_phases) if grounded_phases is not None else (0,)
        
        bus_idx = mc_create.create_bus(
            net, vn_kv=vn_kv, num_phases=num_phases, name=name, 
            grounded_phases=cls_grounded, grounding_r_ohm=grounding_r_ohm, 
            grounding_x_ohm=grounding_x_ohm, in_service=in_service, 
            type=type, zone=zone, index=index
        )
        return {
            "status": "success",
            "message": "Bus created successfully",
            "index": int(bus_idx)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create bus: {str(e)}"
        }


@mcp.tool()
def create_line(std_type: str, model_type: str, from_bus: int, from_phase: Union[int, List[int]], 
                to_bus: int, to_phase: Union[int, List[int]], length_km: float, 
                name: Optional[str] = None, in_service: bool = True, index: Optional[int] = None) -> Dict[str, Any]:
    """Create a line in the current network.
    
    Args:
        std_type: Standard line type name
        model_type: Model type selection ('configuration', 'matrix', 'sequence')
        from_bus: Index of sending bus
        from_phase: Phase identifier(s) at sending bus (single integer or list of integers)
        to_bus: Index of receiving bus
        to_phase: Phase identifier(s) at receiving bus (single integer or list of integers)
        length_km: Line length in km
        name: Name of the line (optional)
        in_service: True if the line is in service (default True)
        index: Index of the line (optional)
    """
    logger.info(f"Creating line: {name} (length={length_km} km)")
    try:
        net = _get_network()
        
        line_idx = mc_create.create_line(
            net, std_type=std_type, model_type=model_type, 
            from_bus=from_bus, from_phase=from_phase, 
            to_bus=to_bus, to_phase=to_phase, 
            length_km=length_km, name=name, 
            in_service=in_service, index=index
        )
        return {
            "status": "success",
            "message": "Line created successfully",
            "index": int(line_idx)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create line: {str(e)}"
        }


@mcp.tool()
def create_switch(bus: int, phase: Union[int, List[int]], element: int, et: str, 
                  type: Optional[str] = None, closed: bool = True, name: Optional[str] = None, 
                  r_ohm: float = 0.0, index: Optional[int] = None) -> Dict[str, Any]:
    """Create a switch in the current network.
    
    Args:
        bus: Connected bus index
        phase: Connected phase identifier(s) (single integer or list of integers)
        element: Index of the element (bus ID if et=='b', line ID if et=='l', trafo ID if et=='t')
        et: Element type: 'l' (line), 't' (transformer), 'b' (bus)
        type: Switch type (optional: 'LS', 'CB', 'LBS', 'DS')
        closed: Initial state (default True)
        name: Name of the switch (optional)
        r_ohm: Internal resistance of switch in Ohms (default 0.0)
        index: Index of the switch (optional)
    """
    logger.info(f"Creating switch: {name}")
    try:
        net = _get_network()
        
        switch_idx = mc_create.create_switch(
            net, bus=bus, phase=phase, element=element, et=et,
            type=type, closed=closed, name=name, r_ohm=r_ohm, index=index
        )
        return {
            "status": "success",
            "message": "Switch created successfully",
            "index": int(switch_idx)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create switch: {str(e)}"
        }


@mcp.tool()
def create_transformer_3ph(hv_bus: int, lv_bus: int, std_type: str, 
                           tap_pos: Optional[float] = None, name: str = "tr", 
                           in_service: bool = True, index: Optional[int] = None, 
                           shift_degree: int = 0) -> Dict[str, Any]:
    """Create a three-phase transformer.
    
    Args:
        hv_bus: High-voltage bus index
        lv_bus: Low-voltage bus index
        std_type: Standard transformer type
        tap_pos: Initial tap position (optional)
        name: Transformer name (default 'tr')
        in_service: Whether active (default True)
        index: Index of transformer (optional)
        shift_degree: Voltage angle shift degree (default 0)
    """
    logger.info(f"Creating transformer: {name} (type={std_type})")
    try:
        net = _get_network()
        t_pos = tap_pos if tap_pos is not None else np.nan
        
        trafo_idx = mc_create.create_transformer_3ph(
            net, hv_bus=hv_bus, lv_bus=lv_bus, std_type=std_type,
            tap_pos=t_pos, name=name, in_service=in_service,
            index=index, shift_degree=shift_degree
        )
        return {
            "status": "success",
            "message": "Transformer created successfully",
            "index": int(trafo_idx)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create transformer: {str(e)}"
        }


@mcp.tool()
def create_asymmetric_load(bus: int, from_phase: Union[int, List[int]], to_phase: Union[int, List[int]], 
                           p_mw: float, q_mvar: float, const_z_percent_p: float = 0.0, 
                           const_i_percent_p: float = 0.0, const_z_percent_q: float = 0.0, 
                           const_i_percent_q: float = 0.0, scaling: float = 1.0, 
                           in_service: bool = True, name: Optional[str] = None, 
                           type: Optional[str] = None, index: Optional[int] = None) -> Dict[str, Any]:
    """Create an asymmetric load in the current network.
    
    Args:
        bus: Connected bus index
        from_phase: Phase identifier(s) at sending side (int or list of ints)
        to_phase: Phase identifier(s) at receiving side (int or list of ints)
        p_mw: Active power in MW
        q_mvar: Reactive power in MVAr
        const_z_percent_p: Active power constant impedance percent share (default 0.0)
        const_i_percent_p: Active power constant current percent share (default 0.0)
        const_z_percent_q: Reactive power constant impedance percent share (default 0.0)
        const_i_percent_q: Reactive power constant current percent share (default 0.0)
        scaling: Scaling factor for active and reactive power (default 1.0)
        in_service: Whether load is in service (default True)
        name: Name of load (optional)
        type: Load type classification (optional)
        index: Index of the load (optional)
    """
    logger.info(f"Creating asymmetric load at bus: {bus}")
    try:
        net = _get_network()
        
        load_idx = mc_create.create_asymmetric_load(
            net, bus=bus, from_phase=from_phase, to_phase=to_phase,
            p_mw=p_mw, q_mvar=q_mvar, const_z_percent_p=const_z_percent_p,
            const_i_percent_p=const_i_percent_p, const_z_percent_q=const_z_percent_q,
            const_i_percent_q=const_i_percent_q, scaling=scaling,
            in_service=in_service, name=name, type=type, index=index
        )
        return {
            "status": "success",
            "message": "Asymmetric load created successfully",
            "index": int(load_idx)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create asymmetric load: {str(e)}"
        }


@mcp.tool()
def create_asymmetric_sgen(bus: int, from_phase: Union[int, List[int]], to_phase: Union[int, List[int]], 
                            p_mw: float, q_mvar: float, const_z_percent_p: float = 0.0, 
                            const_i_percent_p: float = 0.0, const_z_percent_q: float = 0.0, 
                            const_i_percent_q: float = 0.0, scaling: float = 1.0, 
                            in_service: bool = True, name: Optional[str] = None, 
                            type: Optional[str] = None, index: Optional[int] = None) -> Dict[str, Any]:
    """Make asymmetric static generator (e.g. solar, wind, battery) in the current network.
    
    Args:
        bus: Connected bus index
        from_phase: Phase identifier(s) at sending side (int or list of ints)
        to_phase: Phase identifier(s) at receiving side (int or list of ints)
        p_mw: Produced active power in MW
        q_mvar: Produced reactive power in MVAr
        const_z_percent_p: Active power constant impedance percent share (default 0.0)
        const_i_percent_p: Active power constant current percent share (default 0.0)
        const_z_percent_q: Reactive power constant impedance percent share (default 0.0)
        const_i_percent_q: Reactive power constant current percent share (default 0.0)
        scaling: Scaling factor (default 1.0)
        in_service: Whether in service (default True)
        name: Name of static generator (optional)
        type: Generator type (optional)
        index: Index of the static generator (optional)
    """
    logger.info(f"Creating asymmetric static generator at bus: {bus}")
    try:
        net = _get_network()
        
        sgen_idx = mc_create.create_asymmetric_sgen(
            net, bus=bus, from_phase=from_phase, to_phase=to_phase,
            p_mw=p_mw, q_mvar=q_mvar, const_z_percent_p=const_z_percent_p,
            const_i_percent_p=const_i_percent_p, const_z_percent_q=const_z_percent_q,
            const_i_percent_q=const_i_percent_q, scaling=scaling,
            in_service=in_service, name=name, type=type, index=index
        )
        return {
            "status": "success",
            "message": "Asymmetric static generator created successfully",
            "index": int(sgen_idx)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create asymmetric static generator: {str(e)}"
        }


@mcp.tool()
def create_ext_grid(bus: int, from_phase: Union[int, List[int]], to_phase: Union[int, List[int]], 
                    vm_pu: float, va_degree: float, r_ohm: float, x_ohm: float, 
                    name: Optional[str] = None, in_service: bool = True, 
                    index: Optional[int] = None) -> Dict[str, Any]:
    """Create an external grid connection representing an upstream slack bus.
    
    Args:
        bus: Connected bus index
        from_phase: Phase identifier(s) at sending side (int or list of ints)
        to_phase: Phase identifier(s) at receiving side (int or list of ints)
        vm_pu: Voltage set point in per-unit
        va_degree: Voltage angle degree
        r_ohm: Internal grid resistance in Ohms
        x_ohm: Internal grid reactance in Ohms
        name: Name of external grid (optional)
        in_service: Whether in service (default True)
        index: Index of the external grid (optional)
    """
    logger.info(f"Creating external grid at bus: {bus}")
    try:
        net = _get_network()
        
        ext_idx = mc_create.create_ext_grid(
            net, bus=bus, from_phase=from_phase, to_phase=to_phase,
            vm_pu=vm_pu, va_degree=va_degree, r_ohm=r_ohm, x_ohm=x_ohm,
            name=name, in_service=in_service, index=index
        )
        return {
            "status": "success",
            "message": "External grid created successfully",
            "index": int(ext_idx)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create external grid: {str(e)}"
        }


@mcp.tool()
async def run_diagnostic_agent(pkl_path: str, user_message: Optional[str] = None) -> Dict[str, Any]:
    """Run the Multiconductor diagnostic sub-agent on a pickled network file.

    Spawns ``multiconductor.addons.agent.mc_diagnostic_agent.diagnostic_agent``
    (built on the OpenAI Agents SDK) to diagnose convergence and modeling
    issues on the network located at ``pkl_path``. Requires ``OPENAI_API_KEY``
    to be set in the environment and the ``openai-agents`` package installed.

    Args:
        pkl_path: Path to the pickled pandapower-multiconductor network (.pkl).
        user_message: Optional custom prompt for the agent. If omitted, a
            default diagnostic plan is used.

    Returns:
        Dict containing the agent's final output (typically JSON-formatted
        diagnostic report) and status.
    """
    logger.info(f"Launching diagnostic sub-agent for: {pkl_path}")
    try:
        if not os.path.exists(pkl_path):
            return {
                "status": "error",
                "message": f"File not found: {pkl_path}",
            }

        from multiconductor.addons.agent.mc_diagnostic_agent import (
            diagnostic_agent,
        )
        from agents import Runner

        if user_message is None:
            user_message = (
                f"Please diagnose the multiconductor network at: {pkl_path}\n\n"
                "1. Load the network\n"
                "2. Run validation checks\n"
                "3. Attempt power flow\n"
                "4. Check voltage profile\n"
                "5. Analyze topology\n"
                "6. Check phase consistency\n"
                "7. Examine ext_grid and transformer configuration\n"
                "8. Provide your full diagnostic report as JSON."
            )

        result = await Runner.run(diagnostic_agent, user_message)
        return {
            "status": "success",
            "message": "Diagnostic agent completed",
            "pkl_path": pkl_path,
            "final_output": result.final_output,
        }
    except ImportError as ie:
        return {
            "status": "error",
            "message": (
                "Missing dependency for diagnostic agent. "
                "Install with `pip install openai-agents`. "
                f"Details: {ie}"
            ),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Diagnostic agent failed: {str(e)}",
        }


if __name__ == "__main__":
    mcp.run(transport="stdio") 
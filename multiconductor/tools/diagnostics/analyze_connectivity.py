import pickle
import os
import sys
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

NETWORKS_DIR = "networks"


def analyze_network(net_name):
    pkl_path = os.path.join(NETWORKS_DIR, f"{net_name}.pkl")
    if not os.path.exists(pkl_path):
        print(f"Network file not found: {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        net = pickle.load(f)

    print(f"\n{'=' * 60}")
    print(f"Network: {net_name}")
    print(f"{'=' * 60}")
    print(f"Total Buses: {len(net.bus)}")
    print(f"Total Lines: {len(net.line)}")
    print(f"Total Switches: {len(net.switch) if 'switch' in net else 0}")

    # Build connectivity graph
    G = nx.Graph()
    for bus_idx in net.bus.index:
        G.add_node(bus_idx)

    # Add line connections
    for _, line in net.line.iterrows():
        G.add_edge(line["from_bus"], line["to_bus"])

    # Add switch connections (if closed)
    if "switch" in net and len(net.switch) > 0:
        for _, sw in net.switch.iterrows():
            if sw.get("closed", True):
                if sw["et"] == "b":  # Bus-to-bus
                    G.add_edge(sw["bus"], sw["element"])
                # Line switches don't add new connectivity

    # Find connected components
    components = list(nx.connected_components(G))
    print(f"\nConnected Components: {len(components)}")

    # Find slack buses
    if "ext_grid" in net and len(net.ext_grid) > 0:
        slack_buses = set(net.ext_grid.bus.unique())
        print(f"Slack Buses: {slack_buses}")

        # Find which component contains slack
        slack_component = None
        for i, comp in enumerate(components):
            if slack_buses & comp:
                slack_component = i
                print(f"Main grid component #{i}: {len(comp)} buses")
                break

        # Report floating components
        if len(components) > 1:
            print(f"\nFloating Components:")
            for i, comp in enumerate(components):
                if i != slack_component:
                    print(f"  Component #{i}: {len(comp)} buses")
                    # Show a few example buses
                    example_buses = list(comp)[:5]
                    for bus_idx in example_buses:
                        bus_name = net.bus.loc[bus_idx, "name"]
                        print(f"    - Bus {bus_idx}: {bus_name}")
    else:
        print("No external grid found!")
        for i, comp in enumerate(components):
            print(f"  Component #{i}: {len(comp)} buses")


if __name__ == "__main__":
    analyze_network("ALMOND_12KV")
    analyze_network("ANAHURST_4KV")

import sqlite3
import pandas as pd
import os
import pickle
import sys
import numpy as np

# Ensure parent directory is in path to import multiconductor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import multiconductor as mc


class MultconductorTransformer:
    def __init__(self, db_path, networks_dir):
        self.db_path = db_path
        self.networks_dir = networks_dir
        if not os.path.exists(networks_dir):
            os.makedirs(networks_dir)
        self.conn = sqlite3.connect(db_path)
        self.bus_map = {}  # Map NodeId -> Bus Index per network
        self.bus_voltage_map = {}  # Map NodeId -> RatedVoltage (kV)
        self.section_device_map = {}  # Map DeviceNumber -> Section Row

    def build_networks(self):
        if not self._table_exists("MULTINETWORK"):
            # print("MULTINETWORK table not found.")
            return

        networks = pd.read_sql("SELECT * FROM MULTINETWORK", self.conn)
        # print(f"Found {len(networks)} networks.")

        for _, row in networks.iterrows():
            net_id = row["NetworkId"]
            try:
                # print(f"Building {net_id}...")
                net = self.build_network(net_id)
                self.save_network(net, net_id)
                print(f"Successfully built and saved {net_id}")
            except Exception as e:
                print(f"Error building {net_id}: {e}")

    def build_network(self, net_id):
        net = mc.create_empty_network()
        self.bus_map = {}
        self.bus_voltage_map = {}
        self.section_device_map = {}

        # Pre-load section devices for this network to map devices to nodes/sections
        if self._table_exists("MULTISECTIONDEVICE"):
            devs = pd.read_sql(
                f"SELECT * FROM MULTISECTIONDEVICE WHERE NetworkId='{net_id}'",
                self.conn,
            )
            for _, d in devs.iterrows():
                self.section_device_map[d["DeviceNumber"]] = d

        self.load_std_types(net)
        self.create_buses(net, net_id)
        self.create_sections(net, net_id)
        self.create_sources(net, net_id)
        self.create_shunts(net, net_id)
        self.create_generators(net, net_id)

        return net

    def load_std_types(self, net):
        # Overhead Lines (Sequence)
        if self._table_exists("MULTIEQOVERHEADLINE"):
            try:
                df = pd.read_sql("SELECT * FROM MULTIEQOVERHEADLINE", self.conn)
                for _, row in df.iterrows():
                    name = row["EquipmentId"]

                    def get_f(col, default=0.0):
                        val = row.get(col)
                        try:
                            return float(val) if val is not None else default
                        except (ValueError, TypeError):
                            return default

                    data = {
                        "r_ohm_per_km": get_f("PositiveSequenceResistance"),
                        "x_ohm_per_km": get_f("PositiveSequenceReactance"),
                        "r0_ohm_per_km": get_f("ZeroSequenceResistance"),
                        "x0_ohm_per_km": get_f("ZeroSequenceReactance"),
                        "c_nf_per_km": get_f("PosSeqShuntSusceptance", 0) * 1e9,
                        "c0_nf_per_km": 0,
                        "max_i_ka": get_f("NominalRating", 0) / 1000.0,
                    }

                    try:
                        mc.pycci.std_types.create_std_type(
                            net, data, name, element="sequence"
                        )
                    except Exception:
                        pass
            except Exception:
                pass

    def create_buses(self, net, net_id):
        if not self._table_exists("MULTINODE"):
            return

        df = pd.read_sql(
            f"SELECT * FROM MULTINODE WHERE NetworkId='{net_id}'", self.conn
        )
        for _, row in df.iterrows():
            node_id = str(row["NodeId"]).strip()  # Added strip
            vn_kv = row.get("RatedVoltage")
            try:
                vn_kv = float(vn_kv) if vn_kv is not None else 0.4
            except (ValueError, TypeError):
                vn_kv = 0.4

            # Geo
            try:
                geo = (
                    (float(row["X"]), float(row["Y"]))
                    if row["X"] is not None and row["Y"] is not None
                    else None
                )
            except (ValueError, TypeError):
                geo = None

            idx = mc.create_bus(net, vn_kv=vn_kv, name=node_id)
            self.bus_map[node_id] = idx
            self.bus_voltage_map[node_id] = vn_kv

            if geo:
                net.bus_geodata.loc[idx, "x"] = geo[0]
                net.bus_geodata.loc[idx, "y"] = geo[1]

    def create_sections(self, net, net_id):
        if not self._table_exists("MULTISECTION"):
            return

        sections = pd.read_sql(
            f"SELECT * FROM MULTISECTION WHERE NetworkId='{net_id}'", self.conn
        )

        # We also need devices associated with sections
        devices = pd.DataFrame()
        if self._table_exists("MULTISECTIONDEVICE"):
            devices = pd.read_sql(
                f"SELECT * FROM MULTISECTIONDEVICE WHERE NetworkId='{net_id}'",
                self.conn,
            )

        oh_lines = pd.DataFrame()
        if self._table_exists("MULTIOVERHEADLINE"):
            oh_lines = pd.read_sql(
                f"SELECT * FROM MULTIOVERHEADLINE WHERE NetworkId='{net_id}'", self.conn
            )

        dev_by_sec = {}
        if not devices.empty:
            for _, d in devices.iterrows():
                sid = d["SectionId"]
                if sid not in dev_by_sec:
                    dev_by_sec[sid] = []
                dev_by_sec[sid].append(d)

        ohl_by_dev = {}
        if not oh_lines.empty:
            for _, idx_row in oh_lines.iterrows():
                ohl_by_dev[idx_row["DeviceNumber"]] = idx_row

        for _, sec in sections.iterrows():
            sec_id = sec["SectionId"]
            from_node = str(sec["FromNodeId"]).strip()
            to_node = str(sec["ToNodeId"]).strip()

            if from_node not in self.bus_map or to_node not in self.bus_map:
                continue

            from_idx = self.bus_map[from_node]
            to_idx = self.bus_map[to_node]

            # Phases
            phase_code = sec.get("Phase", 7)
            try:
                phase_code = int(phase_code)
            except:
                phase_code = 7

            def get_phases(code):
                if code == 1:
                    return (1,)
                if code == 2:
                    return (2,)
                if code == 3:
                    return (3,)
                if code == 4:
                    return (1, 2)
                if code == 5:
                    return (1, 3)
                if code == 6:
                    return (2, 3)
                if code == 7:
                    return (1, 2, 3)
                return (1, 2, 3)

            line_phases = get_phases(phase_code)

            sec_devs = dev_by_sec.get(sec_id, [])

            line_dev = None
            switch_dev = None

            for d in sec_devs:
                dtype = d["DeviceType"]
                if dtype in [13, 1]:  # Overhead Line or Cable
                    line_dev = d
                if dtype in [8, 3]:  # Switch or Fuse
                    switch_dev = d

            line_idx = None

            # Logic:
            # 1. If Line/Cable -> Create Line.
            # 2. If NO Line/Cable -> Check for Switch.
            #    If switch -> Create Bus-Bus Switch?
            #    BUT if only switch, we might want parallel line? No, switch IS the connection.
            # 3. If NO Line/Cable AND NO Switch -> Create "Connector" line (Service Drop).
            #    This handles sections that only have Loads/Shunts.

            if line_dev is not None:
                dev_num = line_dev["DeviceNumber"]
                dtype = line_dev["DeviceType"]

                length = 0.1
                std_type = None

                if dtype == 13:
                    ohl = ohl_by_dev.get(dev_num)
                    if ohl is not None:
                        length = float(ohl["Length"]) if ohl["Length"] else 0.1
                        std_type = ohl["LineId"]
                elif dtype == 1:
                    length = 0.05
                    std_type = "fallback_cable"

                created = False
                if std_type and mc.pycci.std_types.std_type_exists(
                    net, std_type, element="sequence"
                ):
                    line_idx = mc.create_line(
                        net,
                        from_bus=from_idx,
                        to_bus=to_idx,
                        length_km=length / 1000.0,
                        std_type=std_type,
                        model_type="sequence",
                        from_phase=line_phases,
                        to_phase=line_phases,
                        name=sec_id,
                    )
                    created = True

                if not created:
                    fallback_type = "fallback_oh"
                    try:
                        if not mc.pycci.std_types.std_type_exists(
                            net, fallback_type, element="sequence"
                        ):
                            fallback_data = {
                                "r_ohm_per_km": 0.1,
                                "x_ohm_per_km": 0.1,
                                "r0_ohm_per_km": 0.1,
                                "x0_ohm_per_km": 0.1,
                                "c_nf_per_km": 0,
                                "c0_nf_per_km": 0,
                                "max_i_ka": 0.4,
                            }
                            mc.pycci.std_types.create_std_type(
                                net, fallback_data, fallback_type, element="sequence"
                            )
                    except Exception:
                        pass

                    line_idx = mc.create_line(
                        net,
                        from_bus=from_idx,
                        to_bus=to_idx,
                        length_km=length / 1000.0,
                        std_type=fallback_type,
                        model_type="sequence",
                        from_phase=line_phases,
                        to_phase=line_phases,
                        name=sec_id,
                    )

            # Check if we need a Connector (Service Line)
            if line_idx is None and switch_dev is None:
                # Disconnected section! Create a connector.
                connector_type = "connector_line"
                try:
                    if not mc.pycci.std_types.std_type_exists(
                        net, connector_type, element="sequence"
                    ):
                        conn_data = {
                            "r_ohm_per_km": 0.001,  # Low impedance
                            "x_ohm_per_km": 0.001,
                            "r0_ohm_per_km": 0.001,
                            "x0_ohm_per_km": 0.001,
                            "c_nf_per_km": 0,
                            "c0_nf_per_km": 0,
                            "max_i_ka": 10.0,  # High rating
                        }
                        mc.pycci.std_types.create_std_type(
                            net, conn_data, connector_type, element="sequence"
                        )
                except Exception:
                    pass

                line_idx = mc.create_line(
                    net,
                    from_bus=from_idx,
                    to_bus=to_idx,
                    length_km=0.01,  # Short
                    std_type=connector_type,
                    model_type="sequence",
                    from_phase=line_phases,
                    to_phase=line_phases,
                    name=f"CONN_{sec_id}",
                )

            # Switch Logic
            if switch_dev is not None:
                is_closed = bool(switch_dev.get("Status", 1))

                if line_idx is not None:
                    mc.create_switch(
                        net,
                        bus=from_idx,
                        element=line_idx,
                        et="l",
                        closed=is_closed,
                        phase=line_phases,
                        name=switch_dev["DeviceNumber"],
                    )
                else:
                    # Only switch, Bus-Bus
                    mc.create_switch(
                        net,
                        bus=from_idx,
                        element=to_idx,
                        et="b",
                        closed=is_closed,
                        phase=line_phases,
                        name=switch_dev["DeviceNumber"],
                    )

            # Loads (Type 20, 2?)
            for d in sec_devs:
                if d["DeviceType"] in [20, 2]:
                    mc.create_asymmetric_load(
                        net,
                        bus=to_idx,
                        from_phase=(1, 2, 3),
                        to_phase=0,
                        p_mw=0,
                        q_mvar=0,
                        name=d["DeviceNumber"],
                    )

    def create_sources(self, net, net_id):
        if not self._table_exists("MULTISOURCE"):
            return

        try:
            sources = pd.read_sql("SELECT * FROM MULTISOURCE", self.conn)
        except Exception:
            return

        for _, src in sources.iterrows():
            node = str(src["NodeId"]).strip()
            if node not in self.bus_map:
                continue

            bus_idx = self.bus_map[node]

            r_ohm = 0.1
            x_ohm = 0.1

            try:
                r = float(src.get("PositiveSequenceResistance", 0))
                x = float(src.get("PositiveSequenceReactance", 0))
                if r > 0 or x > 0:
                    r_ohm = r
                    x_ohm = x
            except (ValueError, TypeError):
                pass

            try:
                mc.create_ext_grid(
                    net,
                    bus=bus_idx,
                    from_phase=(1, 2, 3),
                    to_phase=0,
                    vm_pu=1.0,
                    va_degree=0.0,
                    r_ohm=r_ohm,
                    x_ohm=x_ohm,
                    name=node,
                )
            except Exception as e:
                print(f"Error creating ext_grid for {node}: {e}")

    def create_shunts(self, net, net_id):
        if not self._table_exists("MULTISHUNTCAPACITOR"):
            return

        shunts = pd.read_sql(
            f"SELECT * FROM MULTISHUNTCAPACITOR WHERE NetworkId='{net_id}'", self.conn
        )

        sections_df = pd.DataFrame()
        if self._table_exists("MULTISECTION"):
            sections_df = pd.read_sql(
                f"SELECT SectionId, ToNodeId FROM MULTISECTION WHERE NetworkId='{net_id}'",
                self.conn,
            )
            sec_map = sections_df.set_index("SectionId")["ToNodeId"].to_dict()
        else:
            sec_map = {}

        for _, shunt in shunts.iterrows():
            dnum = shunt["DeviceNumber"]
            if dnum not in self.section_device_map:
                continue

            sdev = self.section_device_map[dnum]
            sec_id = sdev["SectionId"]

            node_id = sec_map.get(sec_id)
            if not node_id:
                if sec_id in self.bus_map:
                    node_id = sec_id

            if node_id:
                node_id = str(node_id).strip()

            if not node_id or node_id not in self.bus_map:
                continue

            bus_idx = self.bus_map[node_id]

            def get_val(col):
                try:
                    return float(shunt.get(col, 0))
                except:
                    return 0.0

            qa = get_val("KVARA") / 1000.0
            qb = get_val("KVARB") / 1000.0
            qc = get_val("KVARC") / 1000.0
            pa = get_val("LossesA") / 1000.0
            pb = get_val("LossesB") / 1000.0
            pc = get_val("LossesC") / 1000.0

            try:
                mc.create_asymmetric_shunt(
                    net,
                    bus=bus_idx,
                    from_phase=(1, 2, 3),
                    to_phase=0,
                    q_mvar=[qa, qb, qc],
                    p_mw=[pa, pb, pc],
                    name=dnum,
                    in_service=bool(shunt.get("Status", 1)),
                )
            except Exception as e:
                print(f"Error creating shunt {dnum}: {e}")

    def create_generators(self, net, net_id):
        if not self._table_exists("MULTISYNCHRONOUSGENERATOR"):
            return

        gens = pd.read_sql(
            f"SELECT * FROM MULTISYNCHRONOUSGENERATOR WHERE NetworkId='{net_id}'",
            self.conn,
        )

        sections_df = pd.DataFrame()
        if self._table_exists("MULTISECTION"):
            sections_df = pd.read_sql(
                f"SELECT SectionId, ToNodeId FROM MULTISECTION WHERE NetworkId='{net_id}'",
                self.conn,
            )
            sec_map = sections_df.set_index("SectionId")["ToNodeId"].to_dict()
        else:
            sec_map = {}

        for _, gen in gens.iterrows():
            dnum = gen["DeviceNumber"]
            if dnum not in self.section_device_map:
                continue

            sdev = self.section_device_map[dnum]
            sec_id = sdev["SectionId"]

            node_id = sec_map.get(sec_id)
            if not node_id:
                if sec_id in self.bus_map:
                    node_id = sec_id

            if node_id:
                node_id = str(node_id).strip()

            if not node_id or node_id not in self.bus_map:
                continue

            bus_idx = self.bus_map[node_id]

            p_kw = 0
            try:
                p_kw = float(gen.get("MaxDispatchablePower", 0))
            except:
                pass
            p_mw = p_kw / 1000.0

            kv_set = 1.0
            try:
                kv_set = float(gen.get("KVSet", 1.0))
            except:
                pass

            base_kv = self.bus_voltage_map.get(node_id, 1.0)
            if base_kv <= 0:
                base_kv = 1.0
            vm_pu = kv_set / base_kv

            phase_code = gen.get("Phase", 7)
            try:
                phase_code = int(phase_code)
            except:
                phase_code = 7

            def get_phases(code):
                if code == 1:
                    return (1,)
                if code == 2:
                    return (2,)
                if code == 3:
                    return (3,)
                if code == 4:
                    return (1, 2)
                if code == 5:
                    return (1, 3)
                if code == 6:
                    return (2, 3)
                if code == 7:
                    return (1, 2, 3)
                return (1, 2, 3)

            g_phases = get_phases(phase_code)

            num_phases = len(g_phases)
            if num_phases > 0:
                p_mw_per_phase = p_mw / num_phases
            else:
                p_mw_per_phase = 0
            p_mw_list = [p_mw_per_phase] * num_phases

            try:
                mc.create_asymmetric_gen(
                    net,
                    bus=bus_idx,
                    from_phase=g_phases,
                    to_phase=0,
                    p_mw=p_mw_list,
                    vm_pu=vm_pu,
                    name=dnum,
                    in_service=bool(gen.get("Status", 1)),
                )
            except Exception as e:
                print(f"Error creating gen {dnum}: {e}")

    def _table_exists(self, table_name):
        res = self.conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        ).fetchone()
        return res is not None

    def save_network(self, net, net_id):
        safe_name = str(net_id).replace(" ", "_").replace("/", "_").replace("\\", "_")
        path = os.path.join(self.networks_dir, f"{safe_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(net, f)


if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), "cyme2multi.db")
    net_dir = os.path.join(os.path.dirname(__file__), "networks")

    transformer = MultconductorTransformer(db_path, net_dir)
    transformer.build_networks()

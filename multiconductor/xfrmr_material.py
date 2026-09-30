"""
Utility to parse transformer MATERIAL_CODE strings and return the correct
multiconductor ``trafo1ph`` field values.

Supports all HV/LV connection combinations: Yy, Yd, Dy, Dd.

Usage::

    from multiconductor.xfrmr_material import populate_xfrmr_fields
    x_df = populate_xfrmr_fields(x_df)
"""

import re
import math
import numpy as np


# ── Secondary voltage → winding kV ──────────────────────────────────────────
#   Maps the secondary label from the material code to a line-to-line kV value.
#   The winding voltage is then computed from the connection type (Y or D).
_SECONDARY_LL_KV = {
    "120/240V":     0.24,
    "240/120VV":    0.24,
    "240/120V":     0.24,
    "120V":         0.208,    # 3φ Y → 120 L-N = 208 L-L
    "240V":         0.24,
    "208Y/120VV":   0.208,
    "208Y/120V":    0.208,
    "277V":         0.48,     # 277 L-N = 480 L-L
    "277/480V":     0.48,
    "480V":         0.48,
}

_DEFAULT_LV_LL_KV = 0.24     # default for unknown: assume 120/240


# ── Connection-type helpers ──────────────────────────────────────────────────

def _winding_kv(ll_kv, conn, phase_count=3, is_secondary=False):
    """Return trafo1ph winding voltage for a given system voltage and connection."""
    # For single-phase center-tapped or dedicated secondary windings, the nameplate 
    # voltage is the physical winding voltage; we do not divide by sqrt(3).
    if is_secondary and phase_count == 1:
        return round(ll_kv, 4)

    if conn.upper() == "Y":
        return round(ll_kv / math.sqrt(3), 4)
    else:  # Delta
        return round(ll_kv, 4)


def _phase_assignments(conn, phase_count):
    """
    Return (from_phase, to_phase) for a connection type.

    Y  → from=[1,2,3], to=0
    D  → from=[1,2,3], to=[2,3,1]
    1P → from=None (per-install), to=0 or to=None
    """
    if phase_count == 1:
        if conn.upper() == "Y":
            return None, 0          # phase-to-neutral; phase set per install
        else:
            return None, None       # phase-to-phase; both set per install
    else:
        if conn.upper() == "Y":
            return [1, 2, 3], 0
        else:
            return [1, 2, 3], [2, 3, 1]


# ── Vector group → (hv_conn, lv_conn) ───────────────────────────────────────

_VECTOR_GROUPS = {
    "Yy":   ("Y", "Y"),
    "Yd":   ("Y", "D"),
    "Dy":   ("D", "Y"),
    "Dd":   ("D", "D"),
    "YNyn": ("Y", "Y"),
    "Dyn":  ("D", "Y"),
    "YNd":  ("Y", "D"),
}


def _infer_vector_group(material_code, phase_count):
    """Infer a default vector group from the material code."""
    code_upper = material_code.upper()
    # 208Y explicitly says Y secondary
    if "208Y" in code_upper:
        return "YNyn" if phase_count == 3 else "Yy"
    # Phase-to-phase secondary voltages (240V, 480V without Y marker) could be delta
    # but default to Yy for North American distribution
    return "YNyn" if phase_count == 3 else "Yy"


# ── Core parser ──────────────────────────────────────────────────────────────

def parse_material_code(material_code, vector_group=None, hv_to_phase=None):
    """
    Parse a transformer MATERIAL_CODE and return multiconductor trafo1ph fields.

    Parameters
    ----------
    material_code : str
        e.g. ``"UG_1P_12KV_120/240V_50KVA_1.80%"``
    vector_group : str, optional
        e.g. ``"Yy"``, ``"Dy"``, ``"Dd"``.  If None, inferred from code.

    Returns
    -------
    dict with all XFRMR_ fields.
    """
    code = material_code.strip().upper().replace("%", "")
    parts = code.split("_")

    # ── Phase count ──
    phase_count = 3 if "3P" in parts else 1

    # ── Primary system voltage (line-to-line kV) ──
    primary_kv_ll = 12.0  # default
    for p in parts:
        m = re.match(r"^(\d+\.?\d*)KV$", p)
        if m:
            primary_kv_ll = float(m.group(1))
            break

    # ── Secondary voltage label ──
    secondary_label = "UNKNV"
    for p in parts:
        if re.match(r"^\d+.*V+$", p) and "KV" not in p:
            secondary_label = p
            break

    # ── Secondary L-L kV ──
    secondary_ll_kv = _SECONDARY_LL_KV.get(secondary_label, _DEFAULT_LV_LL_KV)

    # ── KVA → MVA ──
    sn_mva = None
    for p in parts:
        m = re.match(r"^(\d+\.?\d*)KVA$", p)
        if m:
            sn_mva = float(m.group(1)) / 1000.0
            break

    # ── Impedance % ──
    vk_percent = None
    for p in parts:
        try:
            val = float(p)
            if 0 < val < 100:
                vk_percent = val
                break
        except ValueError:
            continue

    # ── Vector group & connection types ──
    if vector_group is None:
        vector_group = _infer_vector_group(material_code, phase_count)

    vg_key = vector_group.replace("N", "").replace("n", "")
    if vg_key in _VECTOR_GROUPS:
        hv_conn, lv_conn = _VECTOR_GROUPS[vg_key]
    else:
        hv_conn, lv_conn = "Y", "Y"

    # For 1P transformers, if a valid working phase is explicitly provided
    # for the to_phase (e.g., connected phase-to-phase), override to Delta (D)
    if phase_count == 1 and hv_to_phase is not None:
        tp = str(hv_to_phase).strip()
        if tp in ("1", "2", "3"):
            hv_conn = "D"

    # ── Winding voltages ──
    vn_hv_kv = _winding_kv(primary_kv_ll, hv_conn, phase_count, is_secondary=False)
    vn_lv_kv = _winding_kv(secondary_ll_kv, lv_conn, phase_count, is_secondary=True)

    # ── Phase assignments ──
    hv_from, hv_to = _phase_assignments(hv_conn, phase_count)
    lv_from, lv_to = _phase_assignments(lv_conn, phase_count)

    # ── Connected phase ──
    if phase_count == 1:
        connected_phase = np.nan  # set per installation
    else:
        connected_phase = "1,2,3"

    return {
        "XFRMR_SN_MVA":              round(sn_mva, 6) if sn_mva else None,
        "XFRMR_VN_HV_KV":            vn_hv_kv,
        "XFRMR_VN_LV_KV":            vn_lv_kv,
        "XFRMR_VK_PERCENT":           vk_percent,
        "XFRMR_VECTOR_GROUP":         vector_group,
        "XFRMR_HV_CONNECTION_TYPE":   hv_conn,
        "XFRMR_LV_CONNECTION_TYPE":   lv_conn,
        "XFRMR_CONNECTED_PHASE":      connected_phase,
        "XFRMR_HV_FROM_PHASE":        hv_from,
        "XFRMR_HV_TO_PHASE":          hv_to,
        "XFRMR_LV_FROM_PHASE":        lv_from,
        "XFRMR_LV_TO_PHASE":          lv_to,
    }


# ── DataFrame populator ─────────────────────────────────────────────────────

def populate_xfrmr_fields(df, material_col="MATERIAL_CODE",
                          vector_group_col="XFRMR_VECTOR_GROUP"):
    """
    Populate transformer fields in a DataFrame based on MATERIAL_CODE.

    If the DataFrame already has a ``XFRMR_VECTOR_GROUP`` column with a
    non-empty value for a row, that value is used.  Otherwise the vector
    group is inferred from the material code (defaults to Yy / YNyn).

    Supports all connection combinations: **Yy, Yd, Dy, Dd**.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``material_col``.
    material_col : str
        Column with material code strings.
    vector_group_col : str
        Column with vector group (optional; used if present).

    Returns
    -------
    pandas.DataFrame
        Modified in-place and returned.
    """
    has_vg_col = vector_group_col in df.columns

    for idx, row in df.iterrows():
        code = str(row[material_col]).strip()
        if not code or code == "nan":
            continue

        # Use existing vector group if present and non-empty
        vg = None
        if has_vg_col:
            vg_val = row.get(vector_group_col)
            if isinstance(vg_val, str) and vg_val.strip():
                vg = vg_val.strip()
                
        # In pandas, missing values might be NaN
        tp_val = row.get("XFRMR_HV_TO_PHASE")
        if pd.isna(tp_val):
            tp_val = None

        p = parse_material_code(code, vector_group=vg, hv_to_phase=tp_val)

        df.at[idx, "XFRMR_SN_MVA"]              = p["XFRMR_SN_MVA"]
        df.at[idx, "XFRMR_VN_HV_KV"]            = p["XFRMR_VN_HV_KV"]
        df.at[idx, "XFRMR_VN_LV_KV"]            = p["XFRMR_VN_LV_KV"]
        df.at[idx, "XFRMR_VK_PERCENT"]           = p["XFRMR_VK_PERCENT"]
        df.at[idx, "XFRMR_VECTOR_GROUP"]         = p["XFRMR_VECTOR_GROUP"]
        df.at[idx, "XFRMR_HV_CONNECTION_TYPE"]   = p["XFRMR_HV_CONNECTION_TYPE"]
        df.at[idx, "XFRMR_LV_CONNECTION_TYPE"]   = p["XFRMR_LV_CONNECTION_TYPE"]
        df.at[idx, "XFRMR_CONNECTED_PHASE"]      = p["XFRMR_CONNECTED_PHASE"]

        hv_from = p["XFRMR_HV_FROM_PHASE"]
        df.at[idx, "XFRMR_HV_FROM_PHASE"] = str(hv_from) if hv_from is not None else np.nan

        hv_to = p["XFRMR_HV_TO_PHASE"]
        df.at[idx, "XFRMR_HV_TO_PHASE"] = str(hv_to) if hv_to is not None else np.nan

        lv_from = p["XFRMR_LV_FROM_PHASE"]
        df.at[idx, "XFRMR_LV_FROM_PHASE"] = str(lv_from) if lv_from is not None else np.nan

        lv_to = p["XFRMR_LV_TO_PHASE"]
        df.at[idx, "XFRMR_LV_TO_PHASE"] = str(lv_to) if lv_to is not None else np.nan

    return df


# ── Reference table generator ───────────────────────────────────────────────

def generate_connection_reference(primary_kv_ll=12.0, secondary_kv_ll=0.24):
    """
    Generate a reference table showing all valid HV/LV connection types.

    Returns a list of dicts for every combination of:
      - Phase count (1P, 3P)
      - Vector group (Yy, Yd, Dy, Dd)

    Parameters
    ----------
    primary_kv_ll : float
        Primary line-to-line voltage in kV.
    secondary_kv_ll : float
        Secondary line-to-line voltage in kV.

    Returns
    -------
    list[dict]
        One record per (phase_count, vector_group) combination.
    """
    records = []
    for phase_count, phase_label in [(1, "1P"), (3, "3P")]:
        for vg in ["Yy", "Yd", "Dy", "Dd"]:
            hv_conn, lv_conn = _VECTOR_GROUPS[vg]
            vn_hv = _winding_kv(primary_kv_ll, hv_conn, phase_count, is_secondary=False)
            vn_lv = _winding_kv(secondary_kv_ll, lv_conn, phase_count, is_secondary=True)
            hv_from, hv_to = _phase_assignments(hv_conn, phase_count)
            lv_from, lv_to = _phase_assignments(lv_conn, phase_count)

            records.append({
                "PHASES":                      phase_label,
                "XFRMR_VECTOR_GROUP":         vg,
                "XFRMR_HV_CONNECTION_TYPE":   hv_conn,
                "XFRMR_LV_CONNECTION_TYPE":   lv_conn,
                "XFRMR_VN_HV_KV":            vn_hv,
                "XFRMR_VN_LV_KV":            vn_lv,
                "XFRMR_HV_FROM_PHASE":        str(hv_from) if hv_from else "per install",
                "XFRMR_HV_TO_PHASE":          str(hv_to) if hv_to is not None else "per install",
                "XFRMR_LV_FROM_PHASE":        str(lv_from) if lv_from else "per install",
                "XFRMR_LV_TO_PHASE":          str(lv_to) if lv_to is not None else "per install",
            })
    return records


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    print("=" * 100)
    print("CONNECTION REFERENCE TABLE (12kV / 240V)")
    print("=" * 100)
    ref = generate_connection_reference(12.0, 0.24)
    ref_df = pd.DataFrame(ref)
    print(ref_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("MATERIAL CODE PARSING (sample)")
    print("=" * 100)
    test_codes = [
        "UG_1P_12KV_120/240V_50KVA_1.80%",
        "UG_3P_12KV_208Y/120VV_300KVA_2.90%",
        "UG_1P_12KV_277/480V_750KVA_5.64%",
        "UG_1P_12KV_UNKNV_75KVA_UNK%",
    ]
    for code in test_codes:
        r = parse_material_code(code)
        print(f"\n{code}")
        for k, v in r.items():
            print(f"  {k:30s} = {v}")

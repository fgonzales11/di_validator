"""Build the outage-aware Kepler map for CKT 245."""

from pathlib import Path
import pickle
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multiconductor.addons.tools.network_geoplot import build_outage_kepler_map


NETWORK_PATH = ROOT / "data" / "CKT_245_16925.pkl"
AMI_PATH = ROOT / "data" / "AMI_CKT_245_16925.csv"
OUTPUT_PATH = ROOT / "data" / "CKT_245_16925_outages.html"

# Endpoint labels visible in the supplied circuit diagram. Matching an endpoint
# marks the associated model line records, including their conductor phases.
DEENERGIZED_LINE_ENDPOINTS = (
    "4052842E",
    "2156053E",
    "4071439E",
    "2293256E",
    "2293259E",
    "4019658E",
    "4019652E",
    "2290334E",
    "BF74072",
    "BF74073",
    "BF75604",
    "P5588509",
)

# The AMI/load identifier is used to resolve the transformer's low-voltage bus.
DEENERGIZED_TRANSFORMERS = ("5588509",)


if __name__ == "__main__":
    with NETWORK_PATH.open("rb") as stream:
        network = pickle.load(stream)
    kepler_map = build_outage_kepler_map(
        network,
        AMI_PATH,
        title="CKT 245 16925 outages",
        deenergized_lines=DEENERGIZED_LINE_ENDPOINTS,
        deenergized_transformers=DEENERGIZED_TRANSFORMERS,
    )
    kepler_map.save_to_html(file_name=str(OUTPUT_PATH), read_only=False)
    print(OUTPUT_PATH)

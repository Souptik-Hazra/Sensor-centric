"""Shared backend runtime state and model initialization."""

import json
import os
import sys
from typing import Any

from .services.chatbot_service import TrafficChatbotService
from .services.routing_service import plan_smart_route
from .llm_engine import llm_engine

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

_gwnet_candidates=[
    os.path.join(BASE_DIR, "gwnet"),
    os.path.abspath(os.path.join(BASE_DIR, "..", "gwnet")),
    os.path.abspath(os.path.join(BASE_DIR, "EquiTrafficAI", "gwnet")),
]
GWNET_DIR=next((path for path in _gwnet_candidates if os.path.exists(path)), _gwnet_candidates[0])
if GWNET_DIR not in sys.path:
    sys.path.insert(0, GWNET_DIR)

try:
    from gwnet_adapter import UniversalPeMSAdapter

    gwnet_adapters={
        "la": UniversalPeMSAdapter("metr_la"),
        "sd": UniversalPeMSAdapter("sd400"),
    }
    print("[+] PyTorch 2.x Graph WaveNet (GWNet) GNN Inference Adapter Loaded Successfully!")
except Exception as error:
    gwnet_adapters={}
    print(f"[!] GWNet PyTorch Adapter Init Notice: {error}")

data_dir_candidates=[
    os.path.join(BASE_DIR, "EquiTrafficAI", "data"),
    os.path.abspath(os.path.join(BASE_DIR, "..", "data")),
    os.path.join(BASE_DIR, "data"),
]
data_dir=next((path for path in data_dir_candidates if os.path.exists(path)), data_dir_candidates[0])

la_location_map, sd_location_map={}, {}
la_loc_path=os.path.join(data_dir, "la_sensor_location_map.json")
sd_loc_path=os.path.join(data_dir, "sd_sensor_location_map.json")
if os.path.exists(la_loc_path):
    with open(la_loc_path, "r", encoding="utf-8") as file:
        la_location_map=json.load(file)
if os.path.exists(sd_loc_path):
    with open(sd_loc_path, "r", encoding="utf-8") as file:
        sd_location_map=json.load(file)

state_data: dict[str, Any]={
    "la": {},
    "sd": {},
    "pems04": {},
    "pems08": {},
    "pems_bay": {},
    "pems03": {},
    "pems07": {},
    "graph_neighbors": {},
}
route_graph_data: dict[str, Any]={"pems_to_idx": {}, "edges": []}
chatbot_service=TrafficChatbotService(llm_engine)
route_planner=plan_smart_route

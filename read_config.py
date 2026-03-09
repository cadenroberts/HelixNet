#!/usr/bin/env python3
import json, sys, os

config_dir = os.environ.get("HELIXNET_CONFIG_DIR", os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(config_dir, "config.json")
with open(config_path) as f:
    cfg = json.load(f)

keys = sys.argv[1].split(".")
v = cfg
for k in keys:
    v = v[k]
print(v if isinstance(v, str) else json.dumps(v))

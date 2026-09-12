"""Translate a bot-lab-21 style DeepSeek V4.1 EXL3 pack config.json to the metadata contract of the
vllm-exl3 plugin (non_routed_quantization, scope, mtp_experts, layer_bits, layer_bits_down).
Keeps a backup as config.json.orig. usage: translate_v41_pack_config.py MODEL_DIR"""
import ast, json, shutil, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "config.json"
bak = p.with_suffix(".json.orig")
if not bak.exists(): shutil.copy(p, bak)
c = json.loads(bak.read_text()); q = c["quantization_config"]
orig = q.get("original_quantization_config") or {}
orig = ast.literal_eval(orig) if isinstance(orig, str) else orig
q["scope"] = "deepseek_v41_routed_experts"
q["non_routed_quantization"] = {
    "quant_method": "deepseek_v4_fp8",
    "weight_block_size": list(orig.get("weight_block_size", [32, 32])),
    "activation_scheme": orig.get("activation_scheme", "dynamic"),
    "scale_fmt": orig.get("scale_fmt", "ue8m0"),
}
q["mtp_experts"] = "source"
q["mtp_experts_start_layer"] = int(c.get("text_config", c).get("num_hidden_layers", 40))
eb = q.get("expert_bits"); eb = ast.literal_eval(eb) if isinstance(eb, str) else (eb or {})
q["layer_bits"] = {str(k): int(v["gu"]) for k, v in eb.items()}          # gate/up K per layer
q["layer_bits_down"] = {str(k): int(v["down"]) for k, v in eb.items()}   # down K per layer (plugin_down_bits.py)
p.write_text(json.dumps(c, indent=2))
mixed = {k: (v["gu"], v["down"]) for k, v in eb.items() if v["gu"] != v["down"]}
print(f"translated {p}: {len(eb)} layers, mixed gate-up/down K in layers {sorted(mixed, key=int)}")

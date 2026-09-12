"""Patch vllm_exl3/exl3.py: per-layer down-projection K (`layer_bits_down` in quantization_config) so packs
with gate/up K != down K in some layers (V4.1 Pollard: layers 20/22/24 = K3/K4) allocate w2_trellis with
the right word count. Applies on top of plugin_mixed_k.py (fused calls already pass K_down)."""
import pathlib, sys
p = pathlib.Path("/work/upstream-vllm-exl3/src/vllm_exl3/exl3.py"); s = p.read_text()
if "layer_bits_down" in s: print("already patched"); sys.exit(0)
old = '''        raw_layer_bits = kwargs.pop("layer_bits", None) or {}
        self.layer_bits: dict[int, int] = {
            int(k): int(v) for k, v in dict(raw_layer_bits).items()
        }
'''
new = old + '''        raw_down = kwargs.pop("layer_bits_down", None) or {}
        self.layer_bits_down: dict[int, int] = {
            int(k): int(v) for k, v in dict(raw_down).items()
        }
'''
assert old in s; s = s.replace(old, new, 1)
old = '''    def bits_for_prefix(self, prefix: str) -> int:'''
new = '''    def down_bits_for_prefix(self, prefix: str) -> int:
        """Per-layer down-projection K (`layer_bits_down`), else the layer's gate/up K."""
        base = self.bits_for_prefix(prefix)
        if not getattr(self, "layer_bits_down", None):
            return base
        m = self._LAYER_RE.search(prefix or "")
        return base if m is None else self.layer_bits_down.get(int(m.group(1)), base)

    def bits_for_prefix(self, prefix: str) -> int:'''
assert old in s; s = s.replace(old, new, 1)
old = '''                layer.moe_config, self, bits=self.bits_for_prefix(prefix)
            )'''
new = '''                layer.moe_config, self, bits=self.bits_for_prefix(prefix),
                bits_down=self.down_bits_for_prefix(prefix),
            )'''
assert old in s; s = s.replace(old, new, 1)
old = '''        self, moe, quant_config: Exl3Config, bits: int | None = None
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        # One method instance per RoutedExperts layer, so this is per-layer K.
        self.bits = int(bits) if bits is not None else quant_config.bits
'''
new = '''        self, moe, quant_config: Exl3Config, bits: int | None = None,
        bits_down: int | None = None,
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        # One method instance per RoutedExperts layer, so this is per-layer K.
        self.bits = int(bits) if bits is not None else quant_config.bits
        self.bits_down = int(bits_down) if bits_down is not None else self.bits
'''
assert old in s; s = s.replace(old, new, 1)
old = '''        w2_trellis = Parameter(
            torch.empty(
                num_experts, out_tiles, in_tiles, k_words, dtype=torch.int16
            ),'''
new = '''        w2_trellis = Parameter(
            torch.empty(
                num_experts, out_tiles, in_tiles, self.bits_down * 16, dtype=torch.int16
            ),'''
assert old in s; s = s.replace(old, new, 1)
old = '''        layer._exl3_bits = self.bits
'''
new = '''        layer._exl3_bits = self.bits
        layer._exl3_bits_down = self.bits_down
'''
assert s.count(old) >= 1; s = s.replace(old, new, 1)
p.write_text(s); print("patched:", p)

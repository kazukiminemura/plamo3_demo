"""Convert PFN's original PLaMo3 implementation, including its unrotated KV cache."""

import json
from pathlib import Path

import torch

MODEL_ID = "pfnet/plamo-3-translate-8b"


class ExportCache:
    """Tensor-only equivalent of DynamicCache for tracing a single request."""

    def __init__(self, past, config):
        self.past = past
        self.present = []
        self.config = config

    def get_seq_length(self):
        # A full-attention layer retains the complete prefix.
        index = self.config.layer_types.index("full_attention")
        return self.past[2 * index].shape[2]

    def update(self, key, value, layer_idx):
        key = torch.cat((self.past[2 * layer_idx], key), dim=2)
        value = torch.cat((self.past[2 * layer_idx + 1], value), dim=2)
        if self.config.layer_types[layer_idx] == "sliding_attention":
            keep = self.config.window_size - 1
            self.present.extend((key[:, :, -keep:, :], value[:, :, -keep:, :]))
        else:
            self.present.extend((key, value))
        return key, value


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, *past):
        cache = ExportCache(past, self.model.config)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions + attention_mask.shape[1] - input_ids.shape[1]
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=positions,
            use_cache=True,
            logits_to_keep=1,
        )
        return (output.logits, *cache.present)


class RotatedExportModel(torch.nn.Module):
    """Equivalent cached attention while the context fits inside the SWA window."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids, *past):
        hidden = self.model.model.embed_tokens(input_ids)
        batch, query_length, _ = hidden.shape
        key_length = attention_mask.shape[1]
        query_positions = torch.arange(query_length) + past[0].shape[2]
        mask = torch.arange(key_length)[None, :] <= query_positions[:, None]
        mask = mask[None, None] & attention_mask[:, None, None, :].bool()
        mask = torch.where(mask.any(dim=-1, keepdim=True), mask, torch.ones_like(mask))
        present = []
        for index, layer in enumerate(self.model.model.layers.layers):
            attention = layer.mixer
            residual = hidden
            projected = attention.qkv_proj(layer.pre_mixer_norm(hidden))
            query, key, value = torch.split(projected,
                [attention.q_proj_dim, attention.k_proj_dim, attention.v_proj_dim], dim=-1)
            query = attention.q_norm(query.view(batch, query_length, attention.q_num_heads, attention.qk_dim).transpose(1, 2))
            key = attention.k_norm(key.view(batch, query_length, attention.k_num_heads, attention.qk_dim).transpose(1, 2))
            value = value.view(batch, query_length, attention.v_num_heads, attention.v_dim).transpose(1, 2)
            cos, sin = attention.rotary_emb(value, seq_len=key_length)
            cos = cos[0, 0][position_ids].unsqueeze(1)
            sin = sin[0, 0][position_ids].unsqueeze(1)
            half = attention.qk_dim // 2
            query = query * cos + torch.cat((-query[..., half:], query[..., :half]), -1) * sin
            key = key * cos + torch.cat((-key[..., half:], key[..., :half]), -1) * sin
            key = torch.cat((past[2 * index], key), dim=2)
            value = torch.cat((past[2 * index + 1], value), dim=2)
            present.extend((key, value))
            key = key[:, :, None].expand(batch, attention.k_num_heads, attention.n_group, key_length,
                                         attention.qk_dim).reshape(batch, attention.q_num_heads, key_length, attention.qk_dim)
            value = value[:, :, None].expand(batch, attention.v_num_heads, attention.n_group, key_length,
                                             attention.v_dim).reshape(batch, attention.q_num_heads, key_length, attention.v_dim)
            mixed = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
            mixed = mixed.transpose(1, 2).reshape(batch, query_length, attention.q_proj_dim)
            hidden = residual + layer.post_mixer_norm(attention.o_proj(mixed))
            hidden = hidden + layer.post_mlp_norm(layer.mlp(layer.pre_mlp_norm(hidden)))
        hidden = self.model.model.norm(hidden[:, -1:])
        return (self.model.lm_head(hidden)[..., :self.model.config.vocab_size], *present)


def convert(model, rotated_cache=False):
    import openvino as ov
    from openvino.frontend.pytorch import ConversionExtension

    config = model.config
    dtype = next(model.parameters()).dtype
    past = tuple(
        torch.zeros(1, config.num_key_value_heads, 2, config.head_dim, dtype=dtype)
        for _ in range(2 * config.num_hidden_layers)
    )
    example = (torch.ones(1, 3, dtype=torch.long), torch.ones(1, 5, dtype=torch.long), *past)
    wrapper = RotatedExportModel(model.eval()) if rotated_cache else ExportModel(model.eval())
    if rotated_cache:
        example = (*example[:2], torch.tensor([[2, 3, 4]]), *past)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, check_trace=False)
        converted = ov.convert_model(traced, example_input=example, extension=[
            ConversionExtension("aten::isneginf", lambda context: [
                ov.opset13.is_inf(context.get_input(0), {"detect_negative": True, "detect_positive": False}).output(0)
            ]),
        ])
    base_names = ["input_ids", "attention_mask"] + (["position_ids"] if rotated_cache else [])
    names = base_names + [
        f"past_{i}_{kind}" for i in range(config.num_hidden_layers) for kind in ("key", "value")
    ]
    shapes = {}
    for index, (port, name) in enumerate(zip(converted.inputs, names, strict=True)):
        port.get_tensor().set_names({name})
        shapes[name] = [1, -1] if index < len(base_names) else [1, config.num_key_value_heads, -1, config.head_dim]
    converted.reshape(shapes)
    for port, name in zip(converted.outputs, ["logits"] + [n.replace("past_", "present_") for n in names[len(base_names):]], strict=True):
        port.get_tensor().set_names({name})
    return converted


def export_model(args):
    import openvino as ov
    from huggingface_hub import model_info, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(args.model_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"出力先が空ではありません: {output}。別の --model-dir を指定してください。")
    print(f"Downloading {MODEL_ID} ({args.revision}) ...", flush=True)
    revision = model_info(MODEL_ID, revision=args.revision).sha
    snapshot = snapshot_download(
        MODEL_ID, revision=revision, local_dir=output.parent / ".source" / revision,
        allow_patterns=["*.json", "*.jsonl", "*.jinja", "*.py", "*.safetensors", "LICENSE/*"],
    )
    print("Loading original model on CPU ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).eval()
    optimized = getattr(args, "npu_optimized", False)
    print(f"Converting to OpenVINO ({'rotated' if optimized else 'unrotated'} KV cache) ...", flush=True)
    converted = convert(model, rotated_cache=optimized)
    if args.weight_format != "fp16":
        import nncf

        mode = nncf.CompressWeightsMode.INT4_SYM if args.weight_format == "int4" else nncf.CompressWeightsMode.INT8_SYM
        options = {"ratio": 1.0, "group_size": getattr(args, "group_size", 128)} if args.weight_format == "int4" else {}
        print(f"Compressing weights: {args.weight_format} ...", flush=True)
        converted = nncf.compress_weights(converted, mode=mode, **options)
    output.mkdir(parents=True, exist_ok=True)
    ov.save_model(converted, output / "openvino_model.xml")
    tokenizer.save_pretrained(output)
    # save_pretrained copies the tokenizer's custom Python implementation.
    model.config.save_pretrained(output)
    import shutil

    for name in ("modeling_plamo.py", "tokenization_plamo.py", "chat_template.jinja"):
        source = Path(snapshot) / name
        if source.exists():
            shutil.copyfile(source, output / name)
    if (Path(snapshot) / "LICENSE").is_dir():
        shutil.copytree(Path(snapshot) / "LICENSE", output / "LICENSE", dirs_exist_ok=True)
    metadata = {
        "model_id": MODEL_ID,
        "revision": revision,
        "weight_format": args.weight_format,
        "context_length": min(32768, model.config.window_size) if optimized else 32768,
        "cache_mode": "rotated" if optimized else "unrotated",
        "group_size": getattr(args, "group_size", 128) if args.weight_format == "int4" else None,
        "default_max_new_tokens": 256,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_key_value_heads": model.config.num_key_value_heads,
        "head_dim": model.config.head_dim,
    }
    (output / "generation_config.json").write_text(json.dumps({
        "max_new_tokens": 256, "do_sample": False, "bos_token_id": 1,
        "eos_token_id": [1, 2, 16], "pad_token_id": 3,
    }, indent=2), encoding="utf-8")
    # Written last: inference rejects incomplete exports.
    (output / "plamo_openvino.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Exported: {output}")

"""The only way tuna loads a model that ships its own Python code.

Models whose repos carry `auto_map` code (the Liquid LFM2.5 encoders, LFM3) run
that code with trust_remote_code. tuna allows it only for the models in
pins.toml, only at the pinned commit (`revision` and `code_revision` are both
the sha), and only with safetensors weights. Everything else in tuna that loads
a model calls `refuse_unpinned()` first, so a Liquid model cannot slip in
through another loader.

    from tuna import pinned
    model = pinned.load("LiquidAI/LFM2.5-Embedding-350M", device="cuda")
    tok, lm = pinned.load("LiquidAI/LFM3-1.7B", device="cuda")
"""
import dataclasses
import os
import tomllib

from . import common

PINS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pins.toml")
# Weight formats that unpickle on load; a pinned revision must ship none of them.
PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt")
# Orgs whose models tuna never loads outside this module.
GUARDED_PREFIXES = ("liquidai/",)

_lfm2_patcher_loaded = None  # model id whose code patched transformers.models.lfm2 in this process


@dataclasses.dataclass(frozen=True)
class Pin:
    model_id: str
    sha: str
    date: str
    kind: str              # sentence-transformer | multi-vector | causal-lm
    patches_lfm2: bool
    audit: str
    compat: tuple = ()


def _lfm2_conv_drop_none_seq_idx():
    """transformers 5.x passes `seq_idx` (packed-sequence boundaries) to every
    LFM2 short conv; code written for 4.x rejects the keyword. Without packing it
    is None and carries nothing, so drop it then and refuse packed inputs."""
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv
    inner = Lfm2ShortConv.slow_forward
    if getattr(inner, "_tuna_compat", False):
        return

    def slow_forward(self, *args, seq_idx=None, **kwargs):
        if seq_idx is not None:
            raise RuntimeError("packed inputs (seq_idx) are not supported by this model's conv")
        return inner(self, *args, **kwargs)

    slow_forward._tuna_compat = True
    Lfm2ShortConv.slow_forward = slow_forward


# Wrappers applied after a pinned model's own code has loaded. They adapt calling
# conventions only; the model's code at the pinned revision is never edited.
COMPAT = {"lfm2-conv-drop-none-seq_idx": _lfm2_conv_drop_none_seq_idx}


def pins():
    with open(PINS_FILE, "rb") as fh:
        table = tomllib.load(fh)["models"]
    out = {}
    for model_id, row in table.items():
        if len(row["sha"]) != 40:
            raise ValueError(f"{PINS_FILE}: {model_id} needs a full 40-char commit sha")
        row["compat"] = tuple(row.get("compat", ()))
        unknown = set(row["compat"]) - set(COMPAT)
        if unknown:
            raise ValueError(f"{PINS_FILE}: {model_id} names unknown compat shims {sorted(unknown)}")
        out[model_id] = Pin(model_id=model_id, **row)
    return out


def pin_for(model_id):
    table = pins()
    if model_id not in table:
        raise PermissionError(f"{model_id} is not in {PINS_FILE}; it needs a pinned, read revision first")
    return table[model_id]


def refuse_unpinned(model_id_or_path):
    """Call before any non-pinned load: refuses Liquid models and any local
    checkout whose config.json asks for remote code."""
    name = str(model_id_or_path)
    if name.lower().startswith(GUARDED_PREFIXES):
        raise PermissionError(f"{name}: load it through tuna.pinned.load(), not directly")
    cfg = os.path.join(name, "config.json")
    if os.path.isfile(cfg):
        import json
        with open(cfg) as fh:
            if "auto_map" in json.load(fh):
                raise PermissionError(f"{name} ships remote code (auto_map); pin it in pins.toml and use tuna.pinned")


def _environment():
    # Hub kernels would fetch compiled code from another repo; the Triton cache stays under tuna's data dir.
    os.environ["USE_HUB_KERNELS"] = "0"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRITON_CACHE_DIR",
                          os.path.join(common.data_root(), "cache", "triton"))


def check_revision_files(pin):
    """List the files at the pinned commit and refuse pickle-format weights.

    Offline (HF_HUB_OFFLINE=1, e.g. the MCP search server) the list is the local
    cache snapshot at exactly the pinned sha: the files that can actually load."""
    from huggingface_hub import HfApi, constants
    if constants.HF_HUB_OFFLINE:
        from huggingface_hub import snapshot_download
        root = snapshot_download(pin.model_id, revision=pin.sha, local_files_only=True)
        files = [os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs]
    else:
        files = HfApi().list_repo_files(pin.model_id, revision=pin.sha)
    bad = [f for f in files if f.lower().endswith(PICKLE_SUFFIXES)]
    if bad:
        raise PermissionError(f"{pin.model_id}@{pin.sha} ships pickle-format files {bad}; refusing to load")
    if not any(f.endswith(".safetensors") for f in files):
        raise PermissionError(f"{pin.model_id}@{pin.sha} has no safetensors weights")
    return files


def _claim_lfm2_patches(pin):
    global _lfm2_patcher_loaded
    if not pin.patches_lfm2:
        return
    if _lfm2_patcher_loaded not in (None, pin.model_id):
        raise RuntimeError(f"{pin.model_id} and {_lfm2_patcher_loaded} both patch transformers.models.lfm2 "
                           "process-wide; load them in separate processes")
    _lfm2_patcher_loaded = pin.model_id


def load(model_id, device="cuda", dtype="bfloat16", **kwargs):
    """Load a pinned remote-code model at its exact revision.

    sentence-transformer -> SentenceTransformer, multi-vector -> MultiVectorEncoder,
    causal-lm -> (tokenizer, AutoModelForCausalLM).
    """
    import torch
    pin = pin_for(model_id)
    _environment()
    check_revision_files(pin)
    _claim_lfm2_patches(pin)
    torch_dtype = getattr(torch, dtype)
    common = {"revision": pin.sha, "trust_remote_code": True}
    if pin.kind in ("sentence-transformer", "multi-vector"):
        import sentence_transformers as st
        cls = st.SentenceTransformer if pin.kind == "sentence-transformer" else st.MultiVectorEncoder
        model = cls(pin.model_id, device=device, **common,
                    model_kwargs={"dtype": torch_dtype, "use_safetensors": True, "code_revision": pin.sha},
                    config_kwargs={"code_revision": pin.sha}, **kwargs)
        for name in pin.compat:
            COMPAT[name]()
        return model
    if pin.kind == "causal-lm":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(pin.model_id, **common, code_revision=pin.sha)
        model = AutoModelForCausalLM.from_pretrained(pin.model_id, **common, code_revision=pin.sha,
                                                     dtype=torch_dtype, use_safetensors=True, **kwargs)
        for name in pin.compat:
            COMPAT[name]()
        return tok, model.to(device).eval()
    raise ValueError(f"{pin.model_id}: unknown kind {pin.kind!r}")

#!/usr/bin/env python3
"""Ready-to-run downloads for the one-step setup, and which of them fit a machine.

family_registry.py knows every engine and how to plan a model it can see on
disk. It has no notion of a model that is not downloaded yet: where to fetch it,
how big it is, how much RAM it needs before its shards exist to be measured.
That is this table. Each entry names a registry family, so the engine, build
target and accelerator support come from the registry, never from here
(tests/test_setup_catalog.py holds the two together).

Only checkpoints that run straight after a plain download are listed: a
pre-converted colibri container, or an official checkpoint the engine reads
natively. Models that need a conversion (GLM-5.3-Flash, OLMoE, Qwen3.8-27B) or a
post-download step with third-party packages (DeepSeek V4.1 Flash) stay in the
README's manual path.

Sizes are the repository listings (with the excludes applied), in GB of 10^9
bytes, checked against the Hugging Face API on 2026-10-02. The RAM figures come
from the measurements quoted in README.md and the per-model docs:

- ram_min_gb: the always-resident weights (the dense part) plus the smallest
  expert cache the engine runs with, plus a little for the operating system.
  Below it the model does not start or swaps.
- ram_good_gb: where it runs as the docs measured it: the whole expert set in
  RAM for the small models, a comfortable cache for the large ones.

After the download, `regesarc setup` runs the real planner (resource_plan.py) on the
files, which is the authority; these numbers only decide what to offer.
"""
import json
import os
from dataclasses import dataclass, field, fields

GB = 1_000_000_000


@dataclass(frozen=True)
class CatalogModel:
    id: str
    family: str                 # family_registry id
    name: str
    repo: str
    disk_gb: float
    ram_min_gb: float
    ram_good_gb: float
    dense_gb: float             # always-resident part, for the "what fits" line
    rank: int                   # higher = more capable
    size_class: str             # "small": fits in RAM on a normal PC; "large": streams from the SSD
    summary: str
    revision: str = "main"
    include: tuple = ()
    exclude: tuple = ()
    modality: str = "text"
    license_note: str = ""
    # (tool under tools/ run with the model folder, the file it creates)
    post_install: tuple = ()
    # The first published release whose prebuilt engines run this checkpoint.
    # Older release archives predate the engine support; build from source then.
    prebuilt_since: str = "1.12.1"
    doc: str = ""
    notes: tuple = field(default_factory=tuple)
    # RegesARC's own model. recommend() offers it ahead of equally-fitting
    # third-party models, the way Strata defaults to Qwen. This is a product
    # decision, not a capability claim, so it is kept separate from `rank`.
    preferred: bool = False


CATALOG = (
    CatalogModel(
        id="qwen36-35b", family="qwen36", name="Qwen3.6-35B-A3B",
        repo="Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64", disk_gb=23.1,
        # docs/qwen36.md: peak RSS 17 GB with every expert cached (cap 256);
        # the minimum follows the same engine's measured 6.5 GB at 32 experts/layer.
        ram_min_gb=10, ram_good_gb=20, dense_gb=3.5, rank=3, size_class="small",
        summary="general chat with thinking and tools; int4-gs64 container",
        doc="docs/qwen36.md"),
    CatalogModel(
        id="qwen3-coder-30b", family="qwen36", name="Qwen3-Coder-30B-A3B",
        repo="Justvugg/Qwen3-Coder-30B-A3B-colibri-int4", disk_gb=19.4,
        # docs/qwen36.md: 6.5 GB resident at 32 experts/layer, 15.2 GB with all 128.
        ram_min_gb=8, ram_good_gb=18, dense_gb=3.6, rank=2, size_class="small",
        summary="coding model with tool calls, no thinking; int4-gs64 container",
        prebuilt_since="2.0.0", doc="docs/qwen36.md#qwen3-coder-30b-a3b"),
    CatalogModel(
        id="deepseek-v4-flash-reap", family="deepseek_v4", name="DeepSeek V4 Flash REAP 150B",
        repo="puwaer/DeepSeek-V4-Flash-0731-reap-150b", disk_gb=84.7,
        ram_min_gb=16, ram_good_gb=32, dense_gb=7.9, rank=4, size_class="large",
        summary="pruned DeepSeek V4 Flash (132 of 256 experts), official fp4/fp8 weights",
        doc="docs/deepseek-v4.md"),
    CatalogModel(
        id="deepseek-v4-flash", family="deepseek_v4", name="DeepSeek V4 Flash",
        repo="deepseek-ai/DeepSeek-V4-Flash-0731", disk_gb=166.9,
        ram_min_gb=16, ram_good_gb=32, dense_gb=7.9, rank=5, size_class="large",
        summary="284B, official checkpoint, no conversion",
        doc="docs/deepseek-v4.md"),
    CatalogModel(
        id="qwen38-flash-next", family="qwen38", name="Qwen3.8-Flash-Next",
        repo="Qwen/Qwen3.8-Flash-Next-FP8", revision="bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
        disk_gb=185.6, ram_min_gb=24, ram_good_gb=32, dense_gb=8.0, rank=5, size_class="large",
        summary="125B + 51B n-gram, official FP8 checkpoint (pinned revision)",
        doc="docs/qwen38.md"),
    CatalogModel(
        id="mimo-v2.6-flash", family="mimo", name="MiMo-V2.6 Flash",
        repo="XiaomiMiMo/MiMo-V2.6-Flash-MOPD", disk_gb=171.8,
        # docs/mimo.md: 30.1 GB resident with 32 experts cached per layer, 49.8 GB with 64.
        ram_min_gb=32, ram_good_gb=52, dense_gb=24.0, rank=6, size_class="large",
        summary="309B with vision and tools, official checkpoint",
        # The engine never loads these three (docs/mimo.md, measured on Pro,
        # the same engine and layout).
        exclude=("dflash/*", "audio_tokenizer/*", "model_mtp.safetensors"),
        prebuilt_since="2.0.0", doc="docs/mimo.md"),
    CatalogModel(
        id="glm-5.2", family="glm", name="GLM-5.2",
        repo="mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp", disk_gb=429.3,
        ram_min_gb=16, ram_good_gb=24, dense_gb=9.9, rank=8, size_class="large",
        summary="744B, the reference model; int4-gs64 with the int8 MTP head",
        doc="README.md#2-get-the-model"),
    CatalogModel(
        id="glm-5.3", family="glm", name="GLM-5.3",
        repo="Justvugg/GLM-5.3-colibri-int4-g64", disk_gb=419.3,
        ram_min_gb=16, ram_good_gb=24, dense_gb=9.9, rank=8, size_class="large",
        summary="744B, int4-gs64, no MTP head (no speculative decoding)",
        doc="README.md#2-get-the-model"),
    CatalogModel(
        id="reges-core-1.0-35", family="qwen36", name="RegesCore 1.0 35B",
        repo="iBossonline/RegesCore-1.0-35",
        # Corrected 2026-10-07 against the repository listing. The earlier 35 GB was
        # wrong: it matched one of the quantised containers, not the model. The BF16
        # GGUF is 69.4 GB (two shards); the repo also carries UD-Q8_K_XL at 38.2 GB,
        # Q8_0 at 36.9, UD-Q6_K_XL at 31.8 and smaller. disk_gb is what the engine
        # downloads, which is the BF16 container; the quantised ones are listed so the
        # choice is visible rather than hidden in a number.
        disk_gb=69.4,
        ram_min_gb=10, ram_good_gb=78, dense_gb=3.6, rank=3, size_class="medium",
        summary="Qwen3_5_Moe architecture, GGUF, MIT. BF16 69.4 GB; Q8 38 GB and "
                "Q6 32 GB containers in the same repo",
        # RegesARC's own model, and the one the project is named around: it is
        # offered first among the models that fit, ahead of the third-party
        # entries of equal capability.
        preferred=True,
        doc="docs/qwen36.md"),
    CatalogModel(
        id="ornith-ai-1.0", family="qwen36", name="Ornith AI 1.0",
        repo="ornith-ai/Ornith-1.0-35B", disk_gb=28.0,
        ram_min_gb=10, ram_good_gb=18, dense_gb=3.6, rank=2, size_class="small",
        summary="Qwen3_5_Moe architecture, safetensors format, MIT license",
        doc="docs/qwen36.md"),
    CatalogModel(
        id="ornith-ai-1.5", family="qwen36", name="Ornith AI 1.5",
        repo="ornith-ai/Ornith-1.5-35B", disk_gb=28.0,
        ram_min_gb=10, ram_good_gb=18, dense_gb=3.6, rank=2, size_class="small",
        summary="Qwen3_5_Moe architecture, safetensors format, MIT license",
        doc="docs/qwen36.md"),
    # The 397B Ornith checkpoints: the same qwen3_5_moe family as the 35B above,
    # so no new engine - see the (60, 512, 4096) DisplayVariant in
    # family_registry.py. Sizes are the repository listings on 2026-10-07
    # (HF tree API): 793.6 GB of weights in 122 shards, plus a separate
    # 13.2 GB model-mtp.safetensors for 1.5. Parameter counts are the API's
    # own: 403,397,928,944 and 396,802,360,816.
    # The RAM figures below are DERIVED, not measured: config.json gives 60
    # layers x 512 experts x 3 x (4096x1024) = 386B routed-expert params, so
    # the always-resident dense part is ~11B params. They are placeholders to
    # make the offer and the fit test work; replace them with measured peak RSS
    # the first time the engine runs this checkpoint (Phase 1 baseline).
    CatalogModel(
        id="ornith-ai-1.5-397b", family="qwen36", name="Ornith AI 1.5 397B",
        repo="ornith-ai/Ornith-1.5-397B", disk_gb=806.8,
        ram_min_gb=24, ram_good_gb=64, dense_gb=12.0, rank=7, size_class="large",
        summary="403B MoE, 512 experts top-10, with its MTP draft head",
        doc="docs/qwen36.md"),
    CatalogModel(
        # The same checkpoint in FP8, verified against the repository listing on
        # 2026-10-07: 418.4 GB against 806.8 GB for BF16. It is a different download,
        # not a smaller model - same 512-expert geometry, so the same family entry.
        # This is the one that makes a 397B reachable on a normal NVMe.
        id="ornith-ai-1.5-397b-fp8", family="qwen36", name="Ornith AI 1.5 397B (FP8)",
        repo="ornith-ai/Ornith-1.5-397B-FP8", disk_gb=418.4,
        ram_min_gb=24, ram_good_gb=64, dense_gb=6.0, rank=7, size_class="large",
        summary="the 397B in FP8 - same model, half the disk, and it fits a 500 GB drive",
        doc="docs/qwen36.md"),
    CatalogModel(
        id="ornith-ai-1.0-397b", family="qwen36", name="Ornith AI 1.0 397B",
        repo="ornith-ai/Ornith-1.0-397B", disk_gb=793.6,
        ram_min_gb=24, ram_good_gb=64, dense_gb=12.0, rank=7, size_class="large",
        summary="397B MoE, 512 experts top-10; no MTP head in this checkpoint",
        doc="docs/qwen36.md"),
    CatalogModel(
        id="mimo-v2.6-pro", family="mimo", name="MiMo-V2.6 Pro",
        repo="XiaomiMiMo/MiMo-V2.6-Pro-MOPD", disk_gb=563.6,
        ram_min_gb=54, ram_good_gb=64, dense_gb=32.4, rank=9, size_class="large",
        summary="1.02T with vision and tools, official checkpoint",
        exclude=("dflash/*", "audio_tokenizer/*", "model_mtp.safetensors"),
        prebuilt_since="2.0.0", doc="docs/mimo.md#pro"),
    CatalogModel(
        id="inkling", family="inkling", name="Inkling",
        repo="nbeerbower/Inkling-colibri-int4", disk_gb=514.1,
        # README: ~120 GB of RAM as downloaded (bf16 dense, 49.4 GB resident);
        # the 25 GB path needs docs/inkling.md's dense conversion first.
        ram_min_gb=120, ram_good_gb=128, dense_gb=49.4, rank=9, size_class="large",
        summary="975B, int4 experts with bf16 dense weights",
        doc="docs/inkling.md"),
    CatalogModel(
        id="kimi-k3", family="kimi", name="Kimi K3",
        repo="moonshotai/Kimi-K3", disk_gb=1561.0,
        ram_min_gb=32, ram_good_gb=64, dense_gb=20.0, rank=10, size_class="large",
        summary="2.8T, original checkpoint with native MXFP4 experts",
        # The repo ships tiktoken.model only; the engines read tokenizer.json.
        post_install=(("k3_tokenizer", "tokenizer.json"),),
        doc="docs/kimi_k3.md"),
    CatalogModel(
        id="qwen-image-2.1", family="qwen_image", name="Qwen-Image-2.1",
        repo="Qwen/Qwen-Image-2.1", disk_gb=33.1,
        # docs/qwen-image.md: 9.0 GB peak for one 768x512 image with the text
        # encoder loaded per prompt, 16.0 GB with everything resident.
        ram_min_gb=12, ram_good_gb=18, dense_gb=8.5, rank=1, size_class="small",
        modality="image", summary="text to image (pictures, not chat)",
        license_note="Qwen Research License: non-commercial use only",
        prebuilt_since="2.0.0", doc="docs/qwen-image.md"),
)


def catalog():
    """The built-in entries, plus any from the JSON list REGESARC_SETUP_CATALOG names:
    a mirror, a private repository, or a test fixture served by a fake hub. An
    extra entry with a built-in id replaces it."""
    entries = {entry.id: entry for entry in CATALOG}
    path = os.environ.get("REGESARC_SETUP_CATALOG")
    if path:
        with open(path, encoding="utf-8") as handle:
            extra = json.load(handle)
        known = {f.name for f in fields(CatalogModel)}
        for item in extra if isinstance(extra, list) else []:
            values = {k: tuple(v) if isinstance(v, list) else v
                      for k, v in item.items() if k in known}
            entry = CatalogModel(**values)
            entries[entry.id] = entry
    return tuple(entries.values())


def by_id(model_id):
    for entry in catalog():
        if entry.id == model_id:
            return entry
    raise KeyError(model_id)


def version_tuple(text):
    parts = []
    for piece in str(text or "").lstrip("v").split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple((parts + [0, 0, 0])[:3])


def platform_supported(entry, os_name, machine):
    """Engine builds the Makefile itself refuses (REGESARC_V4_SUPPORTED)."""
    machine = (machine or "").lower()
    if entry.family == "deepseek_v4":
        x86 = machine in ("x86_64", "amd64")
        arm = machine in ("aarch64", "arm64")
        return (x86 and os_name in ("linux", "win32")) or (arm and os_name in ("linux", "darwin"))
    return True


def assess(entry, ram_total, disk_free, *, os_name="linux", machine="x86_64",
           already_downloaded=0):
    """One entry against one machine. Returns (status, reason).

    status is one of: good, tight, needs-ram, needs-disk, unsupported."""
    if not platform_supported(entry, os_name, machine):
        return "unsupported", f"no {entry.family} engine for {os_name}/{machine}"
    need_disk = max(0.0, entry.disk_gb * GB * 1.02 + 2 * GB - (already_downloaded or 0))
    if ram_total is not None and ram_total < entry.ram_min_gb * GB:
        return "needs-ram", (f"needs {entry.ram_min_gb:g} GB of RAM "
                             f"(dense part {entry.dense_gb:g} GB plus the expert cache); "
                             f"this machine has {ram_total / GB:.1f} GB")
    if disk_free is not None and disk_free < need_disk:
        return "needs-disk", (f"needs {need_disk / GB:.0f} GB free for the download; "
                              f"{disk_free / GB:.0f} GB free there")
    if ram_total is not None and ram_total < entry.ram_good_gb * GB:
        return "tight", (f"runs with a small expert cache (needs {entry.ram_min_gb:g} GB, "
                         f"runs best with {entry.ram_good_gb:g} GB)")
    if entry.size_class == "small":
        return "good", f"the whole model fits in RAM ({entry.ram_good_gb:g} GB)"
    return "good", (f"dense part {entry.dense_gb:g} GB stays in RAM, the experts "
                    "stream from the SSD (a fast NVMe drive makes it faster)")


def recommend(ram_total, disk_free, *, os_name="linux", machine="x86_64",
              downloaded=None, include_images=True):
    """Every catalog entry assessed, the default marked.

    The default is the most capable model that runs from RAM on this machine;
    if no small model fits, the smallest download among the large ones that do.
    A 20 GB download that answers at several tokens per second is the better
    first experience than a 400 GB one at a fraction of a token; the larger
    ones are listed right below it.

    Among models that fit equally well, one marked `preferred` wins: RegesARC
    offers RegesCore first the way Strata defaults to Qwen. `preferred` breaks
    ties, it does not override fit - a model that does not fit is never picked
    for being ours."""
    downloaded = downloaded or {}
    rows = []
    for entry in catalog():
        if entry.modality != "text" and not include_images:
            continue
        status, reason = assess(entry, ram_total, disk_free, os_name=os_name, machine=machine,
                                already_downloaded=downloaded.get(entry.id, 0))
        rows.append({"entry": entry, "status": status, "reason": reason,
                     "fits": status in ("good", "tight"), "recommended": False})
    text = [r for r in rows if r["fits"] and r["entry"].modality == "text"]
    pick = None
    for wanted in ("good", "tight"):
        small = [r for r in text if r["entry"].size_class == "small" and r["status"] == wanted]
        if small:
            pick = max(small, key=lambda r: (r["entry"].preferred, r["entry"].rank))
            break
    if pick is None:
        large = [r for r in text if r["entry"].size_class == "large"]
        if large:
            pick = min(large, key=lambda r: r["entry"].disk_gb)
    if pick is not None:
        pick["recommended"] = True
    fitting = [r for r in rows if r["fits"]]
    fitting.sort(key=lambda r: (not r["recommended"], r["entry"].modality != "text",
                                r["entry"].disk_gb))
    others = [r for r in rows if not r["fits"]]
    others.sort(key=lambda r: r["entry"].disk_gb)
    return fitting + others


def as_dict(row):
    """A recommendation row as plain data (the MCP server and --json)."""
    entry = row["entry"]
    return {
        "id": entry.id, "name": entry.name, "family": entry.family, "repo": entry.repo,
        "revision": entry.revision, "download_gb": entry.disk_gb,
        "ram_min_gb": entry.ram_min_gb, "ram_good_gb": entry.ram_good_gb,
        "dense_gb": entry.dense_gb, "kind": entry.modality,
        "runs": "from RAM" if entry.size_class == "small" else "streams experts from the SSD",
        "summary": entry.summary, "status": row["status"], "fits": row["fits"],
        "reason": row["reason"], "recommended": row["recommended"],
        "license_note": entry.license_note, "doc": entry.doc,
    }


FITS_EXPLAINED = ("fits = the dense part, which always stays in RAM, plus a minimum expert "
                  "cache fit in RAM, and the download fits on the disk")

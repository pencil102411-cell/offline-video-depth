# PyInstaller one-folder build (depth-video.exe + _internal/). The model cache
# is intentionally external so
# the executable remains small; copy hf_cache/ beside the executable or pass
# --cache-dir to a cache on the target machine.
hiddenimports = [
    "transformers.models.depth_anything",
    "transformers.models.depth_anything.configuration_depth_anything",
    "transformers.models.depth_anything.image_processing_depth_anything",
    "transformers.models.depth_anything.modeling_depth_anything",
    "transformers.models.auto",
    "transformers.pipelines",
    "transformers.pipelines.depth_estimation",
    "transformers.image_processing_utils",
    "transformers.image_processing_base",
    "transformers.processing_utils",
    "transformers.tokenization_utils_base",
]
a = Analysis(
    ["depth_video.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Unused packages that the global environment would otherwise drag in.
    excludes=["tensorflow", "flax", "jax", "pyarrow", "av", "onnxruntime", "pandas", "lxml", "hf_xet", "tkinter", "matplotlib", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)
# exclude_binaries keeps this a true one-folder build: embedding binaries in
# the EXE turns it into a one-file bootloader that unpacks ~1 GB to %TEMP% on
# every run and leaves it behind when a conversion is cancelled.
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="depth-video", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="depth-video")

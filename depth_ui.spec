# PyInstaller one-folder build for the optional local browser UI.
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
    ["depth_ui.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tensorflow", "flax", "jax"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="depth-ui", console=False)
coll = COLLECT(exe, a.binaries, a.datas, a.zipfiles, name="depth-ui")



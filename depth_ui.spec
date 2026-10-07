# PyInstaller one-folder build for the optional local browser UI.
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("transformers")
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
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="depth-ui", console=True)
coll = COLLECT(exe, a.binaries, a.datas, a.zipfiles, name="depth-ui")

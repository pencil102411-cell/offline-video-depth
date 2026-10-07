# PyInstaller one-folder build. The model cache is intentionally external so
# the executable remains small; copy hf_cache/ beside the executable or pass
# --cache-dir to a cache on the target machine.
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("transformers")

a = Analysis(
    ["depth_video.py"],
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
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="depth-video", console=True)
coll = COLLECT(exe, a.binaries, a.datas, a.zipfiles, name="depth-video")

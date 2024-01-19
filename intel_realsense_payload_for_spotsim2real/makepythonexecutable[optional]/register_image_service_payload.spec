# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['register_image_service_payload.py'],
    pathex=[],
    binaries=[],
    datas=[('/home/spot/miniconda3/envs/spot_ros/lib/python3.9/site-packages/bosdyn/client/resources/robot.pem', 'bosdyn/client/resources')],
    hiddenimports=['bosdyn.client.resources'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='register_image_service_payload',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

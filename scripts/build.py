import os
import platform
import subprocess
from build_config import *

def get_platform_name():
    system = platform.system().lower()
    if system == "windows":
        return "windows.exe"
    elif system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    else:
        return system

def create_spec_file(asset_name):
    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['src/zoom_earth_cli/main.py'],
    pathex=['{ROOT_DIR}'],
    binaries=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={{}},
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
    name='{asset_name}',
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
"""
    with open('my_cli.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)

def main():
    # 确保在项目根目录
    if not os.path.exists('src/zoom_earth_cli'):
        print("请在项目根目录运行此脚本")
        return

    # 获取平台特定的可执行文件名
    asset_name = f"my-cli-{get_platform_name()}"
    print(f"正在为 {platform.system()} 平台构建 {asset_name}")

    # 创建spec文件
    create_spec_file(asset_name)
    print("已创建 spec 文件")

    # 运行 PyInstaller
    try:
        subprocess.run(['pyinstaller', 'my_cli.spec'], check=True)
        print(f"构建成功！可执行文件位于 dist/{asset_name}")
    except subprocess.CalledProcessError as e:
        print(f"构建失败：{str(e)}")
        return

    # 测试运行
    executable = os.path.join('dist', asset_name)
    if os.path.exists(executable):
        print("\n测试运行:")
        try:
            subprocess.run([executable, '--help'], check=True)
            print("\n测试成功！")
        except subprocess.CalledProcessError as e:
            print(f"\n测试失败：{str(e)}")

if __name__ == '__main__':
    main()
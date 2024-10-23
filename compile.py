# compile.py
import os
import subprocess
import shutil
import sys
import pkg_resources

def compile_driveworm():
    print("🪱 Starting DriveWorm compilation process...")
    
    # Create dist and build directories if they don't exist
    os.makedirs('dist', exist_ok=True)
    os.makedirs('build', exist_ok=True)

    # Check for required directories and files
    required_files = {
        'images/dw_ico.ico': 'Program icon',
        'worm_drive.py': 'Main program file',
        'drive_crawler.py': 'Crawler module'
    }

    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            print(f"Error: Required {description} not found at '{file_path}'!")
            return False

    try:
        # Install PyInstaller if not already installed
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
        
        # Create spec file content
        spec_content = f'''
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['worm_drive.py'],
    pathex=[],
    binaries=[],
    datas=[('images', 'images')],
    hiddenimports=['torch', 'sentence_transformers', 'faiss', 'pandas', 'PyPDF2', 
                   'transformers', 'PIL', 'numpy', 'tqdm', 'huggingface_hub'],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DriveWorm',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['images/dw_ico.ico'],
)
'''
        
        # Write spec file
        with open('DriveWorm.spec', 'w') as f:
            f.write(spec_content)
        
        # Run PyInstaller with the spec file
        subprocess.check_call(['pyinstaller', 'DriveWorm.spec', '--clean'])
        
        print("\n✅ Compilation completed successfully!")
        print("📁 Your executable can be found in the 'dist' directory")
        
        # Verify the executable was created
        exe_path = os.path.join('dist', 'DriveWorm.exe')
        if os.path.exists(exe_path):
            print(f"✨ Created: {exe_path}")
            print(f"📦 Size: {os.path.getsize(exe_path) / (1024*1024):.1f} MB")
        else:
            print("⚠️ Warning: Executable not found in expected location")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during compilation: {str(e)}")
        return False

if __name__ == "__main__":
    compile_driveworm()
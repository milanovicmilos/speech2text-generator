import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
print('Workspace root:', root)

out1 = root / 'speech_recognation_code_for_kaggle.zip'
out2 = root / 'kaggle_bundle_minimal.zip'

full_includes = [
    'kaggle',
    'src',
    'cli',
    'tools',
    'configs',
    'requirements.txt',
    'README.md',
    'spec.txt',
]

minimal_includes = [
    'kaggle',
    'README.md',
]

def add_entries(zf, items):
    for p in items:
        pth = root / p
        if not pth.exists():
            print('  skipping missing:', p)
            continue
        if pth.is_dir():
            for f in pth.rglob('*'):
                if f.is_file():
                    arcname = f.relative_to(root).as_posix()
                    zf.write(f, arcname)
        else:
            zf.write(pth, pth.relative_to(root).as_posix())

print('Building', out1.name)
with zipfile.ZipFile(out1, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    add_entries(zf, full_includes)

print('Building', out2.name)
with zipfile.ZipFile(out2, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    add_entries(zf, minimal_includes)

print('Verifying contents:')
for zp in (out1, out2):
    try:
        with zipfile.ZipFile(zp, 'r') as zf:
            names = zf.namelist()
            found = [n for n in names if n.endswith('kaggle/Kaggle_Full_ASR_Run.ipynb')]
            if found:
                info = zf.getinfo(found[0])
                print(f" - {zp.name}: contains notebook ({found[0]}) size={info.file_size}")
            else:
                print(f" - {zp.name}: NOTEBOOK NOT FOUND")
    except Exception as e:
        print(' -', zp.name, 'error:', e)

print('Done.')

"""Build a manual-install ZIP. HACS installs the tagged custom_components tree."""
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
component = root / 'custom_components' / 'homeguide'
manifest = json.loads((component / 'manifest.json').read_text())
assert manifest['domain'] == component.name
assert all(manifest[k] for k in ('version','documentation','issue_tracker','codeowners'))
output = root / 'dist' / 'homeguide.zip'
output.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(component.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            archive.write(path, path.relative_to(root))
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
    assert 'custom_components/homeguide/manifest.json' in archive.namelist()
print(f'{output}: HomeGuide {manifest["version"]}')

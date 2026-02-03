import os,glob,shutil,subprocess
import modal
VOL = os.environ.get('ZJU_VOL', 'vggt-zju-data')
SEQ = os.environ.get('ZJU_SEQ', 'CoreView_390')
ARCH = os.environ.get('ARCHIVES_DIR', '/mnt/data/archives')
ROOT = os.environ.get('EXTRACT_ROOT', '/mnt/data/zju_mocap')
vol = modal.Volume.from_name(VOL, create_if_missing=False)
image = modal.Image.debian_slim(python_version='3.10').apt_install('tar')
app = modal.App('vggt-prepare-zju')
@app.function(image=image, volumes={'/mnt/data': vol}, timeout=21600)
def prepare():
    need = os.path.join(ROOT, SEQ, 'vggt_geom')
    if os.path.isdir(need):
        print('[prep] ok:', need)
    else:
        parts = sorted(glob.glob(os.path.join(ARCH, SEQ + '.tar.part*')))
        tar_path = os.path.join(ARCH, SEQ + '.tar')
        if (not os.path.exists(tar_path)) and parts:
            print('[prep] assembling', tar_path, 'from', len(parts), 'parts')
            os.makedirs(ARCH, exist_ok=True)
            with open(tar_path, 'wb') as w:
                for p in parts:
                    with open(p, 'rb') as r:
                        shutil.copyfileobj(r, w, 64 * 1024 * 1024)
        if not os.path.exists(tar_path):
            raise FileNotFoundError('[prep] missing tar and parts under: ' + ARCH)
        os.makedirs(ROOT, exist_ok=True)
        print('[prep] extracting', tar_path, '->', ROOT)
        subprocess.check_call(['tar', '-xf', tar_path, '-C', ROOT])
        if not os.path.isdir(need):
            raise FileNotFoundError('[prep] extracted but still missing: ' + need)
    subprocess.call(['bash', '-lc', 'ls -la ' + os.path.join(ROOT, SEQ) + ' | head -200'])
    try:
        vol.commit()
        print('[prep] committed volume')
    except Exception as e:
        print('[prep] vol.commit skipped:', e)
@app.local_entrypoint()
def main():
    prepare.remote()

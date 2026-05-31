"""Pack local V8 code, upload to N150, and extract."""
import paramiko
import os
import tarfile
import io

V8_LOCAL = r"C:\Users\jerry\Desktop\V8"
V8_REMOTE = "/home/jerry/V8"
TAR_PATH = r"C:\Users\jerry\Desktop\V8\v8_sync.tar.gz"

# --- Step 1: Create tarball locally (exclude target/, __pycache__, .pyc) ---
print("[1/4] Creating tarball...")
exclude_dirs = {"target", "__pycache__", ".git", "node_modules", ".venv"}
exclude_exts = {".pyc", ".pyo"}

with tarfile.open(TAR_PATH, "w:gz") as tar:
    for root, dirs, files in os.walk(V8_LOCAL):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            if any(f.endswith(ext) for ext in exclude_exts):
                continue
            # Skip the tarball itself
            full = os.path.join(root, f)
            if full == TAR_PATH:
                continue
            arcname = os.path.relpath(full, V8_LOCAL)
            tar.add(full, arcname=arcname)

size_mb = os.path.getsize(TAR_PATH) / 1024 / 1024
print(f"    Tarball: {TAR_PATH} ({size_mb:.1f} MB)")

# --- Step 2: Upload via SFTP ---
print("[2/4] Uploading to N150...")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.124.230.63', username='jerry', password='i982030i', timeout=15)

remote_tar = "/home/jerry/v8_sync.tar.gz"
sftp = client.open_sftp()
sftp.put(TAR_PATH, remote_tar)
print(f"    Uploaded to {remote_tar}")

# --- Step 3: Backup old V8 and extract ---
print("[3/4] Backing up old V8 and extracting...")
cmds = [
    # Backup critical files that might have been modified on N150
    f"cp -r {V8_REMOTE}/Cargo.toml {V8_REMOTE}/Cargo.toml.bak.20260531 2>/dev/null; true",
    # Extract new code over the old directory
    f"cd {V8_REMOTE} && tar xzf {remote_tar}",
    # Clean up
    f"rm -f {remote_tar}",
    # Verify
    f"find {V8_REMOTE} -name '*.py' -not -path '*__pycache__*' | wc -l",
    f"cat {V8_REMOTE}/Cargo.toml | head -5",
]

for cmd in cmds:
    print(f"    > {cmd[:80]}...")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    if out: print(f"    {out}")
    if err: print(f"    STDERR: {err}")

sftp.close()
client.close()

# Clean up local tarball
os.remove(TAR_PATH)
print(f"\n[4/4] Local tarball cleaned. Sync complete!")

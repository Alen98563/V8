"""Check cargo/rust availability on N150 and install if needed."""
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.124.230.63', username='jerry', password='i982030i', timeout=15)

cmds = [
    # Check common cargo locations
    "ls -la ~/.cargo/bin/cargo 2>/dev/null; ls -la /usr/local/cargo/bin/cargo 2>/dev/null; ls -la /usr/bin/cargo 2>/dev/null; echo '=== END CARGO ==='",
    # Check rustup
    "ls -la ~/.rustup/ 2>/dev/null || echo 'NO_RUSTUP'",
    # Check pip/pipx
    "pip3 --version 2>/dev/null; pipx --version 2>/dev/null; echo '=== END PIP ==='",
    # Check if venv exists
    "ls -la ~/V8/.venv/ 2>/dev/null || echo 'NO_VENV'",
    # Check Python versions available
    "ls /usr/bin/python* 2>/dev/null; python3.12 --version 2>/dev/null || echo 'NO_PY312'",
    # Check apt packages
    "dpkg -l | grep -i rust 2>/dev/null | head -5; echo '=== END APT ==='",
    # Check if rsync is available
    "which rsync 2>/dev/null; which scp 2>/dev/null; echo '=== END TRANSFER ==='",
]

for i, cmd in enumerate(cmds):
    print(f"\n[{i+1}] {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=15)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    if out: print(out)
    if err: print(f"STDERR: {err}")

client.close()

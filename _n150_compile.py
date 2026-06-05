"""Fix PATH and install deps, then compile on N150."""
import os
import os
import paramiko
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(os.getenv('N150_HOST', '100.124.230.63'), username=os.getenv('N150_USER', 'jerry'), password=os.getenv('N150_PASS', ''), timeout=15)

def run(cmd, timeout=120, label=""):
    print(f"\n{'='*60}")
    print(f"[{label}] {cmd[:120]}")
    print('='*60)
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    raw = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    exit_code = stdout.channel.recv_exit_status()
    safe = raw.encode('ascii', errors='replace').decode('ascii') if raw else ""
    safe_err = err.encode('ascii', errors='replace').decode('ascii') if err else ""
    if safe: print(safe[-3000:])
    if safe_err: print(f"STDERR: {safe_err[-2000:]}")
    print(f"EXIT: {exit_code}")
    return exit_code, raw, err

# Step 1: Install Python deps in venv
run('''source ~/V8/.venv/bin/activate
pip install numpy pandas polars pyyaml python-dotenv msgpack-python redis requests 2>&1 | tail -5
''', timeout=180, label="1: Python deps")

# Step 2: Compile with explicit PATH
print("\n[2: COMPILE] with explicit PATH for cargo...")
compile_cmd = '''export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
source ~/V8/.venv/bin/activate
cd ~/V8
echo "cargo: $(which cargo)"
echo "rustc: $(which rustc)"
echo "maturin: $(which maturin)"
echo "python: $(which python)"
maturin develop 2>&1
echo "MATURIN_EXIT=$?"
'''
stdin, stdout, stderr = client.exec_command(compile_cmd, timeout=600)
raw = stdout.read().decode('utf-8', errors='replace')
exit_code = stdout.channel.recv_exit_status()
safe = raw.encode('ascii', errors='replace').decode('ascii')
print(safe[-6000:] if len(safe) > 6000 else safe)
print(f"EXIT: {exit_code}")

# Step 3: Find .so
run("find ~/V8 -name 'v8_core_engine*.so' -not -path '*/target/debug/deps/*' -not -path '*/target/release/deps/*' -not -path '*/target/debug/build/*' -not -path '*/target/release/build/*'", label="3: Find .so")

# Step 4: Import test
run('''export PATH="$HOME/.cargo/bin:$PATH"
source ~/V8/.venv/bin/activate
cd ~/V8
python -c "import v8_core_engine as vce; print('IMPORT OK'); print(dir(vce))" 2>&1
''', label="4: Import test")

client.close()
print("\nDone!")

"""Check V8 project status on N150."""
import paramiko
import sys

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('100.124.230.63', username='jerry', password='i982030i', timeout=15)

cmds = [
    # 1. Check if V8 project directory exists
    "ls -la ~/V8/ 2>/dev/null || echo 'V8_DIR_NOT_FOUND'",
    # 2. Check if Rust .so file exists
    "find ~/V8/ -name '*.so' -o -name '*.pyd' 2>/dev/null | head -20 || echo 'NO_SO_FOUND'",
    # 3. Check cargo/maturin availability
    "which cargo 2>/dev/null; which maturin 2>/dev/null; cargo --version 2>/dev/null; echo '---'; python3 --version 2>/dev/null",
    # 4. Check if v8_core_engine is importable
    "cd ~/V8 && python3 -c 'import v8_core_engine as vce; print(dir(vce))' 2>&1 || echo 'IMPORT_FAILED'",
    # 5. Check docker (Redis/TimescaleDB)
    "docker ps 2>/dev/null || echo 'DOCKER_NOT_RUNNING'",
    # 6. Check project file count
    "find ~/V8/ -name '*.py' -not -path '*__pycache__*' 2>/dev/null | wc -l",
    # 7. Check Cargo.toml features section
    "grep -A3 '\\[features\\]' ~/V8/Cargo.toml 2>/dev/null || echo 'NO_FEATURES_SECTION'",
    # 8. Check if orchestrator exists
    "ls -la ~/V8/orchestrator/ 2>/dev/null || echo 'NO_ORCHESTRATOR'",
    # 9. Check memory/disk
    "free -h | head -3; echo '---'; df -h / | tail -1",
]

for i, cmd in enumerate(cmds):
    print(f"\n{'='*60}")
    print(f"[{i+1}] {cmd}")
    print('='*60)
    stdin, stdout, stderr = client.exec_command(cmd, timeout=15)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    if out:
        print(out)
    if err:
        print(f"STDERR: {err}")

client.close()
print("\n\nDone.")

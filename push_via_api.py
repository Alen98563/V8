#!/usr/bin/env python3
"""Push V8 directory to GitHub via Git Data API."""
import os, json, base64, sys, urllib.request, urllib.error

TOKEN = "REDACTED_PAT_PLACEHOLDER"
OWNER, REPO = "Alen98563", "V8"
BASE = f"https://api.github.com/repos/{OWNER}/{REPO}"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28",
    "Accept": "application/vnd.github+json",
}
V8_ROOT = r"C:\Users\jerry\Desktop\V8"
BINARY_EXTS = {".svg", ".png", ".jpg", ".jpeg", ".ico", ".pdf"}

def api(method, path, data=None):
    url = f"{BASE}{path}" if path.startswith("/") else f"{BASE}/{path}"
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=HEADERS, method=method)
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"API ERROR {method} {path}: {e.code} {err}")
        sys.exit(1)

# 1. Get HEAD
ref = api("GET", "/git/ref/heads/main")
parent_sha = ref["object"]["sha"]
print(f"HEAD: {parent_sha}")

commit = api("GET", f"/git/commits/{parent_sha}")
base_tree_sha = commit["tree"]["sha"]
print(f"Base tree: {base_tree_sha}")

tree = api("GET", f"/git/trees/{base_tree_sha}")
existing = {e["path"] for e in tree["tree"]}
print(f"Existing files: {existing}")
print()

# 2. Collect files
all_files = []
for dirpath, dirnames, filenames in os.walk(V8_ROOT):
    if ".git" in dirpath.split(os.sep):
        continue
    for fn in filenames:
        if fn.endswith(".ps1"):
            continue
        full = os.path.join(dirpath, fn)
        rel = os.path.relpath(full, V8_ROOT).replace("\\", "/")
        all_files.append((rel, full))

print(f"Total files to process: {len(all_files)}")

# 3. Create blobs for new files
new_entries = []
for i, (rel, full) in enumerate(all_files, 1):
    if rel in existing:
        print(f"[{i}/{len(all_files)}] SKIP: {rel}")
        continue

    ext = os.path.splitext(full)[1].lower()
    is_binary = ext in BINARY_EXTS

    if is_binary:
        with open(full, "rb") as f:
            content = base64.b64encode(f.read()).decode("ascii")
        blob_data = {"content": content, "encoding": "base64"}
    else:
        with open(full, "r", encoding="utf-8") as f:
            content = f.read()
        blob_data = {"content": content, "encoding": "utf-8"}

    blob = api("POST", "/git/blobs", blob_data)
    print(f"[{i}/{len(all_files)}] BLOB {blob['sha'][:7]} : {rel}")
    new_entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})

print(f"\nNew entries: {len(new_entries)}")

if not new_entries:
    print("Nothing new to push.")
    sys.exit(0)

# 4. Create tree
tree_data = {"base_tree": base_tree_sha, "tree": new_entries}
new_tree = api("POST", "/git/trees", tree_data)
new_tree_sha = new_tree["sha"]
print(f"New tree: {new_tree_sha}")

# 5. Create commit
commit_data = {
    "message": "feat: upload V8 full codebase",
    "tree": new_tree_sha,
    "parents": [parent_sha],
}
new_commit = api("POST", "/git/commits", commit_data)
new_commit_sha = new_commit["sha"]
print(f"New commit: {new_commit_sha}")

# 6. Update ref
api("PATCH", "/git/refs/heads/main", {"sha": new_commit_sha, "force": False})

print(f"\nSUCCESS: https://github.com/{OWNER}/{REPO}")
print(f"Commit: https://github.com/{OWNER}/{REPO}/commit/{new_commit_sha}")

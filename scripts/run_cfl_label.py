import re, py_compile, tempfile, os

with open("/home/jerry/V8/labeling/counterfactual_labeler.py", "rb") as f:
    raw = f.read()

text = raw.decode("utf-8", errors="replace")
lines = text.split("\n")
fixed = 0

# Keywords that must NOT be commented out
KEYWORDS = [
    "try:", "except", "if ", "for ", "while ", "def ", "class ",
    "import ", "from ", "with ", "return ", "raise ", "assert ",
    "yield ", "break", "continue", "pass", "elif ", "else", "finally",
    "print(", "not ", "and ", "or ", "in ", "is ",
]

for i, line in enumerate(lines):
    stripped = line.strip()
    if not stripped.startswith("?") or stripped.startswith("? "):
        continue
    rest = stripped[1:]
    indent = line[:len(line) - len(stripped)]

    # Check if this starts a Python keyword
    is_kw = any(rest.startswith(kw) for kw in KEYWORDS)
    if is_kw:
        lines[i] = indent + rest
        fixed += 1
        continue

    # Check if it's a corrupted comment (starts with digit after ?)
    if rest and rest[0].isdigit():
        lines[i] = indent + "#" + rest
        fixed += 1
        continue

    # Starts with Chinese or alpha -> comment
    if rest and (ord(rest[0]) > 127 or rest[0].isalpha()):
        lines[i] = indent + "# " + rest
        fixed += 1
        continue

    # Unknown: try removing ? and see if line is valid Python
    lines[i] = indent + rest
    fixed += 1

text = "\n".join(lines)

# Fix the broken print string at line 375-376
text = text.replace(
    'print("\n?CounterfactualLabeler self-test passed")',
    'print("\\nCounterfactualLabeler self-test passed")'
)

tmp = "/tmp/_cfl_test.py"
with open(tmp, "w") as f:
    f.write(text)
try:
    py_compile.compile(tmp, doraise=True)
    print(f"compile OK (fixed {fixed} lines), saving")
    with open("/home/jerry/V8/labeling/counterfactual_labeler.py", "w") as f:
        f.write(text)
except py_compile.PyCompileError as e:
    print(f"STILL ERR (fixed {fixed} lines):", str(e))
os.unlink(tmp)
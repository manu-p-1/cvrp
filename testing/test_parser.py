"""Quick edge-case tests for OCVRPParser refactor."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ocvrp.util import OCVRPParser

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        failed += 1

def write_tmp(content):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".ocvrp", delete=False)
    f.write(content)
    f.close()
    return f.name

# --- Test 1: Colon in COMMENTS value ---
tmp = write_tmp("NAME: TestColon\nCOMMENTS: Augerat 1995: Set A: Extended\nDIM: 3\nCAPACITY: 100\nOPTIMAL: 500\nNODES:\n1 0 0 0\n2 10 20 5\n3 30 40 10\n")
try:
    ps = OCVRPParser(tmp).parse()
    check("colon in comments", ps.get_ps_comments() == "Augerat 1995: Set A: Extended")
finally:
    os.unlink(tmp)

# --- Test 2: Missing required header ---
tmp = write_tmp("NAME: TestMissing\nDIM: 3\nOPTIMAL: 500\nNODES:\n1 0 0 0\n2 10 20 5\n3 30 40 10\n")
try:
    OCVRPParser(tmp).parse()
    check("missing CAPACITY", False)
except SyntaxError as e:
    check("missing CAPACITY", "CAPACITY" in str(e))
finally:
    os.unlink(tmp)

# --- Test 3: Wrong DIM count ---
tmp = write_tmp("NAME: TestDim\nDIM: 5\nCAPACITY: 100\nOPTIMAL: 500\nNODES:\n1 0 0 0\n2 10 20 5\n3 30 40 10\n")
try:
    OCVRPParser(tmp).parse()
    check("wrong DIM", False)
except SyntaxError as e:
    check("wrong DIM", "DIM" in str(e))
finally:
    os.unlink(tmp)

# --- Test 4: Blank line in node section ---
tmp = write_tmp("NAME: TestBlank\nDIM: 4\nCAPACITY: 100\nOPTIMAL: 500\nNODES:\n1 0 0 0\n\n2 10 20 5\n3 30 40 10\n4 50 60 15\n")
try:
    ps = OCVRPParser(tmp).parse()
    check("blank in nodes", len(ps.get_ps_buildings()) == 3 and ps.get_ps_dim() == 3)
finally:
    os.unlink(tmp)

# --- Test 5: Bad node row (wrong column count) ---
tmp = write_tmp("NAME: TestBadRow\nDIM: 3\nCAPACITY: 100\nOPTIMAL: 500\nNODES:\n1 0 0 0\n2 10 20\n3 30 40 10\n")
try:
    OCVRPParser(tmp).parse()
    check("bad node row", False)
except SyntaxError as e:
    check("bad node row", "4 values" in str(e))
finally:
    os.unlink(tmp)

# --- Test 6: Non-.ocvrp extension ---
try:
    OCVRPParser("data/test.txt")
    check("bad extension", False)
except SyntaxError:
    check("bad extension", True)

# --- Test 7: Missing NODES section entirely ---
tmp = write_tmp("NAME: NoNodes\nDIM: 3\nCAPACITY: 100\nOPTIMAL: 500\n")
try:
    OCVRPParser(tmp).parse()
    check("missing NODES", False)
except SyntaxError as e:
    check("missing NODES", "NODES" in str(e))
finally:
    os.unlink(tmp)

# --- Test 8: All real datasets parse ---
import glob
real_files = sorted(glob.glob("data/*.ocvrp"))
for f in real_files:
    ps = OCVRPParser(f).parse()
    name = ps.get_ps_name()
    n = len(ps.get_ps_buildings())
    check(f"{name} ({n} nodes)", n == ps.get_ps_dim())

print(f"\n{passed} passed, {failed} failed")
sys.exit(failed)

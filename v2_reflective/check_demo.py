"""Headless demo check: record the cache (first run), then time a normal run (<= 180 s), fail on error cells, and verify that every
'PAPER:' number printed by the notebook occurs in the frozen sources. Run after each freeze and the day before the presentation."""
import json, os, re, subprocess, sys, time
NB = "presentation/Final_Demo.ipynb"; OUT = "presentation/Final_Demo_executed.ipynb"
def run(env):
    t = time.time(); subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", NB, "--output", os.path.basename(OUT),
                                     "--ExecutePreprocessor.timeout=600"], check=True, env={**os.environ, **env}); return time.time() - t
if not os.path.exists("presentation/demo_cache.json"): print("record run", f"{run({'DEMO_RECORD': '1'}):.0f}s")
secs = run({}); nb = json.load(open(OUT)); text = ""
for c in nb["cells"]:
    for o in c.get("outputs", []):
        assert o.get("output_type") != "error", f"error cell: {o.get('ename')}: {o.get('evalue')}"
        text += "".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", ""))
src = open("v2_reflective/e23/results_frozen.md").read() + open("v2_reflective/e23/b2_report.json").read() if os.path.exists("v2_reflective/e23/b2_report.json") else open("v2_reflective/e23/results_frozen.md").read()
src += open("paper/e23_results.tex").read()
src += open("v2_reflective/EXPERIMENTS_SUMMARY.md").read()                # leak audit (Kaggle close-out) is recorded here
src = src.replace("−", "-").replace("$-$", "-")                             # unicode / LaTeX minus
bad = [m for m in re.findall(r"PAPER: .*? (-?[\d.]+)\n", text) if m not in src]
print(f"runtime {secs:.0f}s, PAPER numbers checked {len(re.findall('PAPER:', text))}, missing {bad}")
assert secs <= 180, "demo slower than 3 minutes"; assert not bad, "printed numbers not found in frozen sources"
print("DEMO CHECK PASS")

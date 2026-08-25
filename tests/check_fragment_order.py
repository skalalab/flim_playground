"""Guard the fragment-escalation ordering.

st.rerun() discards the state of every widget in the fragment that had not yet been
rendered during the interrupted run — including untouched ones, which fall back to their
module defaults. So the single escalation must fire only after every widget in the
fragment has re-registered, and nothing called from inside the fragment may st.rerun()
on its own. AppTest cannot drive this page (no file_uploader accessor), so the invariant
is checked on the source.
"""
import ast
import sys

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
FAILS = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"   {detail}"))
    if not cond:
        FAILS.append(name)

page = open(f"{ROOT}/pages/data_analysis.py").read()
tree = ast.parse(page)

fragment = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_render_plot_and_controls":
        fragment = node
check("the plot fragment exists", fragment is not None)

if fragment:
    reruns, widgets = [], []
    for node in ast.walk(fragment):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name == "rerun":
                reruns.append(node.lineno)
            if name in ("plot_config_widget", "reorder_x_axis_widget"):
                widgets.append((name, node.lineno))
    check("the fragment renders the styling widget", any(n == "plot_config_widget" for n, _ in widgets), widgets)
    check("the fragment escalates exactly once", len(reruns) == 1, reruns)
    last_widget = max((line for _n, line in widgets), default=0)
    check("escalation comes after every widget in the fragment",
          bool(reruns) and min(reruns) > last_widget,
          f"rerun at {reruns}, last widget at {last_widget}")

widgets_src = open(f"{ROOT}/src/widgets/visualization_widgets.py").read()
wtree = ast.parse(widgets_src)
for node in ast.walk(wtree):
    if isinstance(node, ast.FunctionDef) and node.name == "reorder_x_axis_widget":
        inner = [n.lineno for n in ast.walk(node)
                 if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "rerun"]
        check("the reorder widget never reruns on its own (it runs inside the fragment)",
              not inner, f"st.rerun() at {inner}")

check("a rebuild flag carries the reorder request to the single escalation",
      "_plot_needs_rebuild" in widgets_src and "_plot_needs_rebuild" in page)

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILING: {FAILS}"))
sys.exit(1 if FAILS else 0)

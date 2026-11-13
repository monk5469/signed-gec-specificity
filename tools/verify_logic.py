"""Verify that the released scripts are logically identical to the versions frozen and executed for the paper.

The logic fingerprint of a Python module is the SHA-256 of ast.dump() of its syntax tree after (a) dropping `from paths import
DATA_ROOT` (or `DATA_ROOT, MATLAB_EXE`), (b) writing every path under the data root as the string "{ROOT}..." and the MATLAB
executable as "{MATLAB}", (c) removing all docstrings and (d) blanking the values of the metadata keywords `note`, `spec` and `stage`. Comments are not
part of the syntax tree; every other token of the code, including all other string literals and constants, enters the fingerprint.
For a MATLAB file, the fingerprint is the SHA-256 of its text after removing full-line comments, blank lines and leading and
trailing white space, and writing the data root as "{ROOT}"; every other character enters the fingerprint. The fingerprints in
provenance/fingerprints.json were computed from the frozen originals with the same normalisation.

Usage:
    python tools/verify_logic.py
"""
import ast, hashlib, json, os, re, sys

META = ("note", "spec", "stage")
ROOT_NAME, PLACEHOLDER = "DATA_ROOT", "{ROOT}"
MATLAB_NAME, MATLAB_PLACEHOLDER = "MATLAB_EXE", "{MATLAB}"
NEW_MODULES = ("paths.py",)          # added for the release; defines DATA_ROOT and MATLAB_EXE


def _is_root(node):
    return isinstance(node, ast.Name) and node.id == ROOT_NAME


class _Canonical(ast.NodeTransformer):
    """Write every DATA_ROOT-relative path as the constant "{ROOT}<relative path>"."""

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Add) and isinstance(node.left, ast.Constant) and node.left.value == PLACEHOLDER \
                and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
            return ast.Constant(value=PLACEHOLDER + node.right.value)
        return node

    def visit_JoinedStr(self, node):
        values = []
        for v in node.values:
            if isinstance(v, ast.FormattedValue) and _is_root(v.value) and v.conversion == -1 and v.format_spec is None:
                v = ast.Constant(value=PLACEHOLDER)
            else:
                v = self.visit(v)
            if values and isinstance(v, ast.Constant) and isinstance(values[-1], ast.Constant):
                values[-1] = ast.Constant(value=values[-1].value + v.value)
            else:
                values.append(v)
        node.values = values
        return node

    def visit_Name(self, node):
        if _is_root(node):
            return ast.Constant(value=PLACEHOLDER)
        if node.id == MATLAB_NAME:
            return ast.Constant(value=MATLAB_PLACEHOLDER)
        return node


def fingerprint_tree(tree):
    tree.body = [s for s in tree.body if not (isinstance(s, ast.ImportFrom) and s.module == "paths"
                                              and ROOT_NAME in [a.name for a in s.names]
                                              and {a.name for a in s.names} <= {ROOT_NAME, MATLAB_NAME})]
    tree = _Canonical().visit(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in META:
            node.value = ast.Constant(value="")
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def logic_fingerprint(source):
    return fingerprint_tree(ast.parse(source))


def matlab_fingerprint(text):
    """Released MATLAB files: [getenv('SGEC_DATA_ROOT') '<rel>'] -> '{ROOT}<rel>'; full-line comments and blank lines dropped."""
    text = re.sub(r"\[getenv\('SGEC_DATA_ROOT'\) '([^']*)'\]", lambda m: "'" + PLACEHOLDER + m.group(1) + "'", text)
    lines = [ln.strip() for ln in text.splitlines()]
    return hashlib.sha256("\n".join(ln for ln in lines if ln and not ln.startswith("%")).encode()).hexdigest()


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rec = json.load(open(os.path.join(root, "provenance", "fingerprints.json"), encoding="utf-8"))
    code = os.path.join(root, "code")
    now = {f: logic_fingerprint(open(os.path.join(code, f), encoding="utf-8").read())
           for f in sorted(os.listdir(code)) if f.endswith(".py") and f not in NEW_MODULES}
    now.update({f: matlab_fingerprint(open(os.path.join(code, f), encoding="utf-8").read())
                for f in sorted(os.listdir(code)) if f.endswith(".m")})
    bad = [f for f, r in sorted(rec.items()) if f in now and now[f] != r["logic_fingerprint"]]
    missing = sorted(set(rec) - set(now)); extra = sorted(set(now) - set(rec))
    print(f"checked {len(rec)} scripts | logic identical: {len(rec) - len(bad) - len(missing)} | differing: {bad} | "
          f"missing: {missing} | unlisted: {extra}")
    sys.exit(1 if (bad or missing or extra) else 0)


if __name__ == "__main__":
    main()

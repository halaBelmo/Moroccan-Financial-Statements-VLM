"""Read the exact prompts and group definitions from Local_extraction.ipynb."""

from __future__ import annotations

import ast
import json
from functools import lru_cache
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent / "Local" / "Local_extraction.ipynb"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def _cell_source(cells: list[dict], index: int) -> str:
    return "".join(cells[index]["source"])


def _function(source: str, name: str):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace[name]


def _literal(source: str, name: str):
    tree = ast.parse(source)
    node = next(
        n.value for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    )
    return ast.literal_eval(node)


@lru_cache(maxsize=1)
def notebook_settings():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    assert _literal(_cell_source(cells, 25), "MODEL_ID") == MODEL_ID
    actif = _literal(_cell_source(cells, 11), "GROUPES")
    passif = _literal(_cell_source(cells, 25), "GROUPES_PASSIF")
    prompt_actif = _function(_cell_source(cells, 13), "create_prompt_v7")
    prompt_passif = _function(_cell_source(cells, 25), "create_prompt_passif_v13")
    # The ACTIF notebook groups cover source rows 3–53. Rows 1–2 are
    # introductory table rows and are omitted from its extraction output.
    assert sum(count for _, count in actif.values()) == 51
    assert sum(count for _, count in passif.values()) == 44
    return {
        "ACTIF": {"groups": actif, "prompt": prompt_actif, "version": "V7"},
        "PASSIF": {"groups": passif, "prompt": prompt_passif, "version": "V13"},
    }

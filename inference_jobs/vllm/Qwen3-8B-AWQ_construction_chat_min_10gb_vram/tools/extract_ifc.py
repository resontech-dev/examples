#!/usr/bin/env python3
"""Extract a compact BIM digest (JSON) from an IFC file — the Level A step.

LOCAL tool — runs on your machine, not shipped to the cluster (only
`scripts/` is uploaded). Produces the digest shape that predict.py consumes
for both Level A (digest-in-context) and Level B (SQLite + tools).

    pip install ifcopenshell
    python tools/extract_ifc.py Duplex_A_20110907.ifc -o sample_data/bim_digest.json

Good starter files:
  * Duplex Apartment — Duplex_A_20110907.ifc (classic buildingSMART sample)
  * Schependomlaan  — github.com/openBIMstandards/DataSetSchependomlaan
  * buildingSMART Sample-Test-Files — github.com/buildingSMART/Sample-Test-Files

Notes:
  * Quantities come from BaseQuantities ONLY if the designer exported them.
    If absent, compute geometrically via ifcopenshell.geom — offline, here in
    the extract step, never in the agent's runtime.
  * Clashes: run `ifcclash` separately and merge its output into the
    "clashes" list (predict.py's Level B reads them from there).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import ifcopenshell
    import ifcopenshell.util.element as ue
except ImportError:
    sys.exit("ifcopenshell is required: pip install ifcopenshell")


def _scalar(v):
    """Psets can contain nested/odd values — keep JSON-serializable scalars."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def extract(ifc_path: Path) -> dict:
    f = ifcopenshell.open(str(ifc_path))

    # IFC2x3 calls them IfcBuildingElement; IFC4.3 renamed to IfcBuiltElement.
    try:
        els = f.by_type("IfcBuildingElement")
    except Exception:
        els = f.by_type("IfcBuiltElement")

    storeys = [s.Name for s in f.by_type("IfcBuildingStorey")]
    project = next(iter(f.by_type("IfcProject")), None)

    elements = []
    for el in els:
        container = ue.get_container(el)
        psets = {
            pset: {k: _scalar(v) for k, v in props.items() if k != "id"}
            for pset, props in (ue.get_psets(el) or {}).items()
            if isinstance(props, dict)
        }
        elements.append({
            "guid": el.GlobalId,
            "class": el.is_a(),
            "name": el.Name,
            "storey": container.Name if container is not None else None,
            "material": [m.Name for m in (ue.get_materials(el) or []) if getattr(m, "Name", None)],
            "psets": psets,
        })

    return {
        "project": getattr(project, "Name", None) or ifc_path.stem,
        "schema": f.schema,
        "storeys": storeys,
        "elements": elements,
        "clashes": [],   # merge ifcclash output here for Level B clash questions
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ifc", type=Path, help="path to the .ifc file")
    ap.add_argument("-o", "--out", type=Path,
                    default=Path("sample_data/bim_digest.json"))
    args = ap.parse_args()

    digest = extract(args.ifc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(digest, indent=1, default=str))

    kb = args.out.stat().st_size // 1024
    print(f"{args.out}: {len(digest['elements'])} elements, "
          f"{len(digest['storeys'])} storeys, {kb} KB")
    if kb > 400:
        print("⚠ digest is large for Level A (32k context ≈ ~90 KB of text) — "
              "filter element classes or go straight to Level B (SQLite).")


if __name__ == "__main__":
    main()

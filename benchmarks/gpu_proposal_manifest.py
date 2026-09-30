"""Verify step-8 source hashes and inventory measured outputs and inputs."""

import argparse
import json
from pathlib import Path

from gpu_resident_manifest import ROOT, digest


def main(artifacts):
    artifacts = artifacts.resolve()
    for stage in ("short", "long"):
        for path in (artifacts / stage).glob("*/results.json"):
            result = json.loads(path.read_text())
            for name, expected in result["source_sha256"].items():
                if digest(ROOT / name) != expected:
                    raise ValueError(f"Measured source differs: {path}, {name}")
    docs = ROOT / "docs/performance"
    files = list(artifacts.rglob("*")) + list((docs / "results").glob("step8*"))
    files += [
        docs / "STEP8.md",
        docs / "README.md",
        docs / "CHANGELOG.md",
        docs / "AUTHORSHIP_REWRITE.json",
    ]
    for folder, pattern in (
        ("benchmarks", "*.py"),
        ("benchmarks/gpu", "*"),
        ("src", "*.py"),
        ("tests/data", "*"),
        ("tests/reference", "*.py"),
        ("data/Cu", "*"),
    ):
        files += list((ROOT / folder).rglob(pattern))
    files += [
        ROOT / name
        for name in (
            "examples/multi_calib.py",
            "examples/cu_config.yaml",
            "pyproject.toml",
            "uv.lock",
            "tests/test_pooled_gibbs_batch.py",
            "tests/test_pooled_proposal_batch.py",
            "tests/test_gpu_resident_predictions.py",
            "tests/test_gpu_material_history.py",
            "tests/test_batched_gpu_requests.py",
        )
    ]
    files += list(
        (ROOT.parent / "benchmark/results/reference").glob("pooled_*/draws.npz")
    )
    manifest = {
        str(p.relative_to(ROOT.parent)): {
            "sha256": digest(p),
            "bytes": p.stat().st_size,
        }
        for p in sorted(set(files))
        if p.is_file()
        and p.suffix != ".pyc"
        and p.name != "step8-manifest.json"
    }
    (docs / "results/step8-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Verified measured source hashes and hashed {len(manifest)} files")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, required=True)
    main(p.parse_args().artifacts)

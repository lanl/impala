"""Hash step-5 artifacts and inputs; verify measured short/long source hashes."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main(artifacts):
    artifacts = artifacts.resolve()
    for stage in ("short", "long"):
        for path in (artifacts / stage).glob("*/results.json"):
            result = json.loads(path.read_text())
            for name, expected in result["source_sha256"].items():
                if digest(ROOT / name) != expected:
                    raise ValueError(f"Measured source differs: {path}, {name}")
    docs = ROOT / "docs/performance"
    files = list(artifacts.rglob("*"))
    files += list((docs / "results").glob("step5*"))
    files += [docs / "STEP5.md", docs / "README.md", docs / "CHANGELOG.md"]
    for directory, pattern in (
        ("benchmarks", "*.py"),
        ("benchmarks/gpu", "*"),
        ("src", "*.py"),
        ("tests/data", "*"),
        ("tests/reference", "*.py"),
        ("data/Cu", "*"),
    ):
        files += list((ROOT / directory).rglob(pattern))
    files += [
        ROOT / name
        for name in (
            "examples/multi_calib.py",
            "examples/cu_config.yaml",
            "pyproject.toml",
            "uv.lock",
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
        and p.name != "step5-manifest.json"
    }
    (docs / "results/step5-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"Verified measured sources and hashed {len(manifest)} artifacts/inputs"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, required=True)
    main(p.parse_args().artifacts)

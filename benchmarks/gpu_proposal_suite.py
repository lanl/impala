"""Step-8 paired proposal-likelihood controls, retaining step-7 Gibbs batching."""

import argparse
from pathlib import Path

import gpu_worker_suite as suite

suite.OPTIONS = {
    name: [
        *options,
        *(
            []
            if "--worker-batch-gibbs" in options
            else ["--worker-batch-gibbs"]
        ),
        *(["--worker-batch-proposals"] if name.endswith("candidate") else []),
    ]
    for name, options in suite.OPTIONS.items()
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["short", "long", "profile"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    suite.main(parser.parse_args())

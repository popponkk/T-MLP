"""Check the dataset names required by the CLS-readout ablation script."""

from data.env import available_datasets


REQUIRED = [
    "hpcg2",
    "hpgmg3",
    "ramspeed",
    "mix_with_five_datasets161",
    "raiderstream",
    "stream",
    "cachesweep",
]


def main():
    available = set(available_datasets())
    missing = [dataset for dataset in REQUIRED if dataset not in available]
    if missing:
        raise SystemExit(f"Unregistered requested datasets: {', '.join(missing)}")
    print("All seven CLS-readout ablation datasets are registered.")


if __name__ == "__main__":
    main()

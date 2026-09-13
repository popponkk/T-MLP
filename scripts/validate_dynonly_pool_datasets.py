"""Fail early when a requested pool-ablation dataset is not registered."""

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
    missing = [name for name in REQUIRED if name not in available]
    if missing:
        raise SystemExit(f"Unregistered requested datasets: {', '.join(missing)}")
    print("All seven dynamic-only pool-ablation datasets are registered.")


if __name__ == "__main__":
    main()

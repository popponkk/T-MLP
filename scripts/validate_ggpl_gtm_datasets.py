"""Validate the seven requested formal GGPL-GTM datasets before batch runs."""

from data.env import available_datasets


REQUIRED = [
    "hpcg2", "hpgmg3", "ramspeed", "mix_with_five_datasets161",
    "raiderstream", "stream", "cachesweep",
]


def main() -> None:
    missing = [name for name in REQUIRED if name not in set(available_datasets())]
    if missing:
        raise SystemExit(f"Unregistered requested datasets: {', '.join(missing)}")
    print("All seven formal GGPL-GTM datasets are registered.")


if __name__ == "__main__":
    main()

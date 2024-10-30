"""Module with functions to run make commands."""

import subprocess


def run_makefile(target: str) -> None:
    """Executes a specified target in the Makefile using a subprocess.

    Args:
        target (str): The target command from the Makefile to run.

    Raises:
        subprocess.CalledProcessError: If the Makefile command fails.
    """
    try:
        subprocess.run(["make", target], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Makefile target '{target}':")
        print(e)
        raise e

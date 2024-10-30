import subprocess


def run_makefile(target):

    try:
        subprocess.run(["make", target], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Makefile target '{target}':")
        print(e)
        raise e

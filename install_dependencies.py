import subprocess, sys, pathlib

def main():
    reqs = ["numpy", "pandas", "lightgbm"]
    optional = ["torch", "holidayskr"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", *reqs])
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *optional])
    except subprocess.CalledProcessError:
        print("Optional packages failed to install; continuing.")

if __name__ == "__main__":
    main()

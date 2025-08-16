import torch


def select_device() -> str:
    """Interactively select computing device.

    Returns 'cpu', 'cuda', or 'mps'.
    """
    while True:
        choice = input("Select compute environment (macOS/gpu/cpu): ").strip().lower()
        if choice == "macos":
            if torch.backends.mps.is_available():
                return "mps"
            else:
                print("MPS not available. Please choose another option.")
        elif choice in ("gpu", "cuda"):
            if torch.cuda.is_available():
                return "cuda"
            else:
                print("CUDA GPU not available. Please choose another option.")
        elif choice == "cpu":
            return "cpu"
        else:
            print("Invalid option. Choose from macOS/gpu/cpu.")

#!/usr/bin/env python3
"""Converts the RWKV .pth file to SafeTensors for use in Rust."""

if __name__ == "__main__":
    import torch
    from safetensors.torch import save_file

    filename = "models/rwkv-430m/RWKV-4-Pile-430M-20220808-8066.pth"
    out = filename.rstrip(".pth") + ".safetensors"
    model = torch.load(filename, map_location="cpu")
    save_file(model, out)
    print("Completed!")

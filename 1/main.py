"""Проверка устройств PyTorch перед работой над лабораторной."""

from statistics import median
from time import perf_counter

import torch
from torch import nn


def benchmark(device: torch.device, image_size: int) -> float:
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    ).to(device)
    images = torch.randn(16, 3, image_size, image_size, device=device)
    labels = torch.randint(0, 10, (16,), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    timings = []

    for step in range(8):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if step >= 3:
            timings.append(perf_counter() - start)

    return median(timings) * 1_000


def main() -> None:
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda or 'no'}")
    print(f"CUDA available: {cuda_available}")
    print(f"CPU threads: {torch.get_num_threads()}")

    devices = [torch.device("cpu")]
    if cuda_available:
        devices.append(torch.device("cuda:0"))
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    for device in devices:
        for image_size in (32, 224):
            elapsed_ms = benchmark(device, image_size)
            print(f"{device} | batch=16 | {image_size}x{image_size} | {elapsed_ms:.1f} ms/step")


if __name__ == "__main__":
    main()

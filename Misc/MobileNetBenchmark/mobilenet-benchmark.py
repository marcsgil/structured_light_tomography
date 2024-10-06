import torch
import torchvision.models as models
import time
import numpy as np


def create_dummy_input():
    # Create a single dummy image with shape (1, 2, 224, 224)
    return torch.randn(1, 2, 224, 224)


def benchmark_inference(model, input_tensor, num_iterations=100):
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            end_time = time.perf_counter()
            # Convert to milliseconds
            times.append((end_time - start_time) * 1000)

    return times


def main():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create MobileNetV3 Large model
    model = models.mobilenet_v3_small()

    # Modify the first conv layer to accept 2 channels instead of 3
    original_conv = model.features[0][0]
    model.features[0][0] = torch.nn.Conv2d(
        2, original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False if original_conv.bias is None else True
    )

    model = model.to(device)
    model.eval()

    # Create input tensor
    input_tensor = create_dummy_input().to(device)

    # Run benchmark
    print("Running benchmark...")
    times = benchmark_inference(model, input_tensor)

    # Print results
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"\nResults over {len(times)} iterations:")
    print(f"Mean inference time: {mean_time:.2f} ms")
    print(f"Std deviation: {std_time:.2f} ms")
    print(f"Min time: {np.min(times):.2f} ms")
    print(f"Max time: {np.max(times):.2f} ms")


if __name__ == "__main__":
    main()

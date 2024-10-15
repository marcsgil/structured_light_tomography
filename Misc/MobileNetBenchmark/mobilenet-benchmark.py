import torch
import torchvision.models as models
import torch.utils.benchmark as benchmark
import numpy as np


def build_model(device):
    # Create MobileNetV3 Large model
    model = models.mobilenet_v3_small()

    # Modify the first conv layer to accept 1 channel instead of 3
    original_conv = model.features[0][0]
    model.features[0][0] = torch.nn.Conv2d(
        1, original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False if original_conv.bias is None else True
    )

    return model.to(device)


def run_inference(model, input_tensor):
    model.eval()
    with torch.no_grad():
        return model(input_tensor)


def main():
    devices = [torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"), torch.device("cpu")]

    for device in devices:
        print(f"Using device: {device}")

        model = build_model(device)

        # Create input tensor
        input_tensor = torch.randn(1, 1, 224, 224, device=device)

        num_threads = torch.get_num_threads()

        timer = benchmark.Timer(
            stmt='run_inference(model, input_tensor)',
            setup='from __main__ import run_inference',
            globals={'model': model, 'input_tensor': input_tensor},
            num_threads=num_threads)

        m = timer.blocked_autorange(min_run_time=1)
        times = m.times
        print(
            f"{(round(np.mean(times) * 1000, 2))} +- {round(1000*np.std(times), 2)} ms")


if __name__ == "__main__":
    main()

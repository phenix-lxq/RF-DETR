import argparse
import json
from pathlib import Path

import torch
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.strategies import SingleDeviceStrategy
from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer


class XPUAccelerator(Accelerator):
    @staticmethod
    def name():
        return "xpu"

    def setup_device(self, device):
        if device != torch.device("xpu:0") or not self.is_available():
            raise RuntimeError(f"XPU unavailable or unsupported device: {device}")
        torch.xpu.set_device(device)

    def teardown(self):
        torch.xpu.empty_cache()

    def get_device_stats(self, device):
        return {"allocated_bytes": torch.xpu.memory_allocated(device)}

    @staticmethod
    def parse_devices(devices):
        if devices != 1:
            raise ValueError("This adapter supports exactly one XPU")
        return [0]

    @staticmethod
    def get_parallel_devices(devices):
        return [torch.device("xpu:0")]

    @staticmethod
    def auto_device_count():
        return 1

    @staticmethod
    def is_available():
        return torch.xpu.is_available()


class DeviceProof(Callback):
    def __init__(self):
        self.report = {"batches": 0, "backward_steps": 0}

    def check(self, tensor):
        if tensor.device != torch.device("xpu:0"):
            raise RuntimeError(f"Unexpected tensor device: {tensor.device}")

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        for parameter in module.parameters():
            self.check(parameter)
        samples, targets = batch
        self.check(samples.tensors)
        if samples.mask is not None:
            self.check(samples.mask)
        for target in targets:
            for value in target.values():
                if isinstance(value, torch.Tensor):
                    self.check(value)
        self.report["batches"] += 1
        self.report["model_device"] = str(next(module.parameters()).device)
        self.report["input_device"] = str(samples.tensors.device)

    def on_after_backward(self, trainer, module):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        if not gradients:
            raise RuntimeError("Backward produced no gradients")
        for gradient in gradients:
            self.check(gradient)
            if not torch.isfinite(gradient).all().item():
                raise RuntimeError("Nonfinite gradients")
        if not any(gradient.count_nonzero().item() for gradient in gradients):
            raise RuntimeError("All gradients are zero")
        torch.xpu.synchronize()
        self.report.update(gradient_device=str(gradients[0].device), gradient_tensors=len(gradients))
        self.report["backward_steps"] += 1


def main():
    seed_everything(42, workers=True)
    if not torch.xpu.is_available():
        raise RuntimeError("XPU required; CPU fallback is disabled")
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=root / "rfdetr-smoke")
    parser.add_argument("--output", type=Path, default=root / "rfdetr-xpu-verified")
    parser.add_argument("--full-epoch", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()
    output = args.output.resolve()
    annotation = json.loads((args.dataset / "annotations/instances_train2017.json").read_text())
    num_classes = max(category["id"] for category in annotation["categories"])
    model_config = RFDETRNanoConfig(pretrain_weights="rf-detr-nano.pth" if args.pretrained else None,
                                   device="xpu:0", amp=False, num_classes=num_classes)
    train_config = TrainConfig(
        dataset_dir=str(args.dataset.resolve()), dataset_file="coco",
        output_dir=str(output), epochs=args.epochs, batch_size=1, grad_accum_steps=1,
        lr=args.lr if args.lr is not None else (1e-5 if args.pretrained else 1e-4),
        lr_encoder=args.lr if args.lr is not None else (1e-5 if args.pretrained else 1.5e-4),
        num_workers=0, multi_scale=False, expanded_scales=False,
        use_ema=False, tensorboard=False, wandb=False, run_test=False, compute_val_loss=True,
    )
    module = RFDETRModelModule(model_config, train_config, load_encoder_weights=False)
    data = RFDETRDataModule(model_config, train_config)
    proof = DeviceProof()
    trainer = build_trainer(
        train_config, model_config, accelerator=XPUAccelerator(),
        strategy=SingleDeviceStrategy(device="xpu:0"), devices=1,
        precision="32-true", extra_callbacks=[proof], limit_train_batches=1.0 if args.full_epoch else 2,
        limit_val_batches=1.0 if args.full_epoch else 1, num_sanity_val_steps=0, log_every_n_steps=1,
    )
    output.mkdir(parents=True, exist_ok=True)
    baseline = trainer.validate(module, datamodule=data)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2))
    trainer.fit(module, datamodule=data)
    if proof.report["backward_steps"] < 2 or trainer.global_step < 2:
        raise RuntimeError("Training did not complete two optimizer steps")
    trainer.save_checkpoint(output / "verified.ckpt")
    proof.report.update(optimizer_steps=trainer.global_step, device_name=torch.xpu.get_device_name(0))
    (output / "device-proof.json").write_text(json.dumps(proof.report, indent=2))
    print(json.dumps(proof.report, indent=2))


if __name__ == "__main__":
    main()

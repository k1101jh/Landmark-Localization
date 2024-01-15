import os
import hydra
import logging
from datetime import datetime
from tqdm import tqdm
from hydra.utils import instantiate
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from loss.ac_loss import ACloss

OmegaConf.register_new_resolver("merge", lambda x, y: x + y)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    config_yaml = OmegaConf.to_yaml(cfg)
    print(config_yaml)

    ## device 설정
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print("\tGPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    ## tensorboard 설정
    # current_time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    work_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    writer = SummaryWriter(log_dir=work_dir)

    ## dataset
    datasets = {
        "train": instantiate(cfg.dataset, cfg.train_dataset_type, cfg),
        "test": instantiate(cfg.dataset, cfg.test_dataset_type, cfg),
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
        ),
    }

    ## 파라미터 설정
    best_loss = 10
    epoch_loss = 0.0
    epoch_w_loss = 0.0
    start_epoch = 0

    loss_function = ACloss(width=cfg.dataset.width, height=cfg.dataset.height, num_landmarks=cfg.dataset.num_landmarks)

    ## 모델 설정
    model = instantiate(cfg.model.constructor)
    model = model.to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    if cfg.train.load_model_path:
        loaded_model = torch.load(cfg.load_model_path, map_location=device)
        start_epoch = loaded_model["epoch"]
        model.load_state_dict(loaded_model["model"])
        optimizer.load_state_dict(loaded_model["optim"])
        scheduler.load_state_dict(loaded_model["scheduler"])

    if cfg.train.profiling:
        # 프로파일링 결과 폴더 생성
        profiling_result_dir = f"{work_dir}/profiling"
        os.makedirs(profiling_result_dir, exist_ok=True)

        # profiler 설정
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiling_result_dir),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        # profiling 시작
        prof.start()

    ## 훈련 시작
    logging.info("Start training...")

    train_pbar = tqdm(range(start_epoch + 1, cfg.train.epochs + 1), desc="epochs", position=0, leave=True)
    for epoch in train_pbar:
        if epoch % cfg.train.test_interval == 0:
            phases = ["train", "test"]
        else:
            phases = ["train"]

        for phase in phases:
            if phase == "train":
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels, _ in tqdm(dataloaders[phase], desc="iterations", position=1, leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    loss, l2_loss, w_loss, angle_loss, dist_loss = loss_function.forward(outputs, labels)
                    metrics["loss"] += loss.item() * labels.size(0)
                    metrics["w_loss"] += w_loss.item() * labels.size(0)
                    metrics["angle_loss"] += angle_loss.item() * labels.size(0)
                    metrics["dist_loss"] += dist_loss.item() * labels.size(0)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            if phase == "train":
                scheduler.step()

            # print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics["loss"] / epoch_samples
            epoch_w_loss = metrics["w_loss"] / epoch_samples
            epoch_angle_loss = metrics["angle_loss"] / epoch_samples
            epoch_dist_loss = metrics["dist_loss"] / epoch_samples

            if epoch == 1:
                img_grid = torchvision.utils.make_grid(inputs)
                writer.add_image("sample augmentations", img_grid, epoch_samples)

            writer.add_scalar(phase + " loss", epoch_loss, epoch)
            writer.add_scalar(phase + " w loss", epoch_w_loss, epoch)
            writer.add_scalar(phase + " angle loss", epoch_angle_loss, epoch)
            writer.add_scalar(phase + " dist loss", epoch_dist_loss, epoch)

            # 모델 저장
            if phase == "test":
                save_filenames = []
                if epoch % cfg.test_interval:
                    save_filenames.append(f"epoch_{epoch}.pth")
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    save_filenames.append("best_model.pth")

                state = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }

                for save_filename in save_filenames:
                    save_file_path = os.path.join(work_dir, save_filename)
                    torch.save(state, save_file_path)

        train_pbar.set_description(desc="loss: {:0.8f}".format(epoch_loss) + " w_loss: {:0.8f}".format(epoch_w_loss))

    writer.close()

    # 프로파일링 결과 파일 저장
    if cfg.train.profiling:
        prof.stop()
        # prof.export_chrome_trace(f"{profiling_result_dir}/trace.json")
        prof.export_stacks(f"{profiling_result_dir}/profiler_stacks.txt", "self_cuda_time_total")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


if __name__ == "__main__":
    train()

import argparse
import os

import paddle
import paddle.distributed as dist
import paddle.optimizer as optim
from paddle import nn
from paddle.io import DataLoader
from paddle.optimizer.lr import OneCycleLR
from tqdm import tqdm

from dataloader import Doc3dDataset
from GeoTr import GeoTrP
from loss import Unwarploss


def train(args):
    dist.init_parallel_env()

    # experiment name
    experiment_name = os.path.join(args.project, args.name)
    os.makedirs(experiment_name, exist_ok=True)

    dataset = Doc3dDataset
    data_path = args.data_path
    train_dataset = dataset(
        data_path, is_transform=True, img_size=(args.img_rows, args.img_cols)
    )
    train_data_size = len(train_dataset)
    print("The number of training samples = %d" % train_data_size)
    val_dataset = dataset(
        data_path,
        is_transform=True,
        split="val",
        img_size=(args.img_rows, args.img_cols),
    )
    val_data_size = len(val_dataset)
    print("The number of validation samples = %d" % val_data_size)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    model = GeoTrP()
    model = paddle.DataParallel(model)

    toal_steps = int(train_data_size / args.batch_size * args.epochs)
    scheduler = OneCycleLR(
        max_learning_rate=args.lr, total_steps=toal_steps, phase_pct=0.1
    )
    optimizer = optim.AdamW(learning_rate=scheduler, parameters=model.parameters())

    MSE = nn.MSELoss()
    loss_fn = nn.L1Loss()
    reconst_loss = Unwarploss()

    epoch_start = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"Loading model and optimizer from checkpoint '{args.resume}'")
            checkpoint = paddle.load(path=args.resume)
            model.set_state_dict(state_dict=checkpoint["model_state"])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            epoch_start = checkpoint["epoch"]
        else:
            print(f"No checkpoint found at '{args.resume}'")

    best_val_mse = 99999.0
    global_step = 0
    for epoch in range(epoch_start, args.epochs):
        avg_loss = 0.0
        avgl1loss = 0.0
        avgrloss = 0.0
        avgssimloss = 0.0
        train_mse = 0.0
        model.train()
        for i, (images, labels) in tqdm(enumerate(train_dataloader)):
            target = model(images[:, 3:, :, :])
            x = target
            perm_0 = list(range(x.ndim))
            perm_0[1] = 2
            perm_0[2] = 1
            x = x.transpose(perm=perm_0)
            perm_1 = list(range(x.ndim))
            perm_1[2] = 3
            perm_1[3] = 2
            target_nhwc = x.transpose(perm=perm_1)
            l1loss = loss_fn(target_nhwc, labels)
            rloss, ssim, uworg, uwpred = reconst_loss(
                images[:, :-1, :, :], target_nhwc, labels
            )
            loss = 10.0 * l1loss + 0.5 * rloss
            avgl1loss += float(l1loss)
            avg_loss += float(loss)
            avgrloss += float(rloss)
            avgssimloss += float(ssim)
            train_mse += MSE(target_nhwc, labels).item()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            scheduler.step()
            global_step += 1
            if (i + 1) % 50 == 0:
                avg_loss = avg_loss / 50
                print(
                    "Epoch[%d/%d] Batch [%d/%d] Loss: %.4f"
                    % (epoch + 1, args.n_epoch, i + 1, len(train_dataloader), avg_loss)
                )
                avg_loss = 0.0
        avgssimloss = avgssimloss / len(train_dataloader)
        avgrloss = avgrloss / len(train_dataloader)
        avgl1loss = avgl1loss / len(train_dataloader)
        train_mse = train_mse / len(train_dataloader)
        print("Training L1:%4f" % avgl1loss)
        print(f"Training MSE:'{train_mse}'")
        optimizer.get_lr()
        model.eval()
        val_l1loss = 0.0
        val_mse = 0.0
        val_rloss = 0.0
        val_ssimloss = 0.0
        for i_val, (images_val, labels_val) in tqdm(enumerate(val_dataloader)):
            with paddle.no_grad():
                target = model(images_val[:, 3:, :, :])
                x = target
                perm_2 = list(range(x.ndim))
                perm_2[1] = 2
                perm_2[2] = 1
                x = x.transpose(perm=perm_2)
                perm_3 = list(range(x.ndim))
                perm_3[2] = 3
                perm_3[3] = 2
                target_nhwc = x.transpose(perm=perm_3)
                pred = target_nhwc.data.cpu()
                gt = labels_val.cpu()
                l1loss = loss_fn(target_nhwc, labels_val)
                rloss, ssim, uworg, uwpred = reconst_loss(
                    images_val[:, :-1, :, :], target_nhwc, labels_val
                )
                val_l1loss += float(l1loss.cpu())
                val_rloss += float(rloss.cpu())
                val_ssimloss += float(ssim.cpu())
                val_mse += float(MSE(pred, gt))
        val_l1loss = val_l1loss / len(val_dataloader)
        val_mse = val_mse / len(val_dataloader)
        val_ssimloss = val_ssimloss / len(val_dataloader)
        val_rloss = val_rloss / len(val_dataloader)
        print(f"val loss at epoch {epoch + 1}:: {val_l1loss}")
        print(f"val mse: {val_mse}")
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            base_name = (
                f"{epoch + 1}_{val_mse}_{train_mse}_{experiment_name}_best_model.pkl"
            )
            full_name = os.path.join(experiment_name, base_name)
            paddle.save(obj=state, path=full_name, protocol=4)
        if (epoch + 1) % 10 == 0:
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            base_name = f"{epoch + 1}_{val_mse}_{train_mse}_{experiment_name}_model.pkl"
            full_name = os.path.join(experiment_name, base_name)
            paddle.save(obj=state, path=full_name, protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-params")
    parser.add_argument(
        "--data_path", nargs="?", type=str, default="", help="Data path to load data"
    )
    parser.add_argument(
        "--img_rows", nargs="?", type=int, default=128, help="Height of the input image"
    )
    parser.add_argument(
        "--img_cols", nargs="?", type=int, default=128, help="Width of the input image"
    )
    parser.add_argument(
        "--epochs", nargs="?", type=int, default=100, help="# of the epochs"
    )
    parser.add_argument(
        "--batch_size", nargs="?", type=int, default=1, help="Batch Size"
    )
    parser.add_argument(
        "--lr", nargs="?", type=float, default=1e-04, help="Learning Rate"
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        default=None,
        help="Path to previous saved model to restart from",
    )
    parser.add_argument(
        "--name", type=str, default="DocTrPlus", help="Name of the experiment"
    )
    parser.add_argument(
        "--project", type=str, default="runs", help="Name of the project"
    )
    args = parser.parse_args()
    train(args)

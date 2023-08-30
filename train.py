import argparse
import os

import paddle
import paddle.distributed as dist
import paddle.optimizer as optim
from paddle import nn
from paddle.io import DataLoader
from paddle.optimizer.lr import OneCycleLR

from dataloader import Doc3dDataset
from GeoTr import GeoTr


def train(args):
    dist.init_parallel_env()

    # experiment name
    experiment_name = os.path.join(args.project, args.name)
    os.makedirs(experiment_name, exist_ok=True)

    data_path = args.data_path
    train_dataset = Doc3dDataset(data_path, img_size=(args.img_rows, args.img_cols))
    train_data_size = len(train_dataset)
    print("The number of training samples = %d" % train_data_size)
    val_dataset = Doc3dDataset(
        data_path,
        split="val",
        img_size=(args.img_rows, args.img_cols),
    )
    val_data_size = len(val_dataset)
    print("The number of validation samples = %d" % val_data_size)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    model = GeoTr()
    model = paddle.DataParallel(model)

    toal_steps = len(train_dataloader) * args.epochs
    scheduler = OneCycleLR(
        max_learning_rate=args.lr, total_steps=toal_steps, phase_pct=0.1
    )
    optimizer = optim.AdamW(learning_rate=scheduler, parameters=model.parameters())

    loss_fn = nn.L1Loss()

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

    best_val_l1loss = 99999.0
    global_step = 0
    for epoch in range(epoch_start, args.epochs):
        avg_loss = 0.0
        model.train()
        # images [N, C, H, W] 24 x 3 x 288 x 288
        # labels [N, H, W, C] 24 x 288 x 288 x 2
        for i, (images, labels) in enumerate(train_dataloader):
            target = model(images)
            x = target
            perm_0 = list(range(x.ndim))
            perm_0[1] = 2
            perm_0[2] = 1
            x = x.transpose(perm=perm_0)
            perm_1 = list(range(x.ndim))
            perm_1[2] = 3
            perm_1[3] = 2
            target_nhwc = x.transpose(perm=perm_1)
            loss = loss_fn(target_nhwc, labels)
            avg_loss += float(loss.cpu())
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            scheduler.step()
            global_step += 1
            if (i + 1) % 50 == 0:
                avg_loss = avg_loss / 50
                print(
                    "Epoch[%d/%d] Batch [%d/%d] Loss: %.4f"
                    % (epoch + 1, args.epochs, i + 1, len(train_dataloader), avg_loss)
                )
                avg_loss = 0.0
        avg_l1_loss = avg_loss / len(train_dataloader)
        lr = optimizer.get_lr()
        print(f"Train loss at epoch {epoch + 1}: {avg_l1_loss:.4f}, lr: {lr:.6f}")
        model.eval()
        val_l1loss = 0.0
        for _, (images_val, labels_val) in enumerate(val_dataloader):
            with paddle.no_grad():
                target = model(images_val)
                x = target
                perm_0 = list(range(x.ndim))
                perm_0[1] = 2
                perm_0[2] = 1
                x = x.transpose(perm=perm_0)
                perm_1 = list(range(x.ndim))
                perm_1[2] = 3
                perm_1[3] = 2
                target_nhwc = x.transpose(perm=perm_1)
                l1loss = loss_fn(target_nhwc, labels_val)
                val_l1loss += float(l1loss.cpu())
        val_l1loss = val_l1loss / len(val_dataloader)
        print(f"Val loss at epoch {epoch + 1}: {val_l1loss:.4f}")
        if val_l1loss < best_val_l1loss:
            best_val_l1loss = val_l1loss
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            base_name = "best_model.pkl"
            full_name = os.path.join(experiment_name, base_name)
            paddle.save(obj=state, path=full_name, protocol=4)
        if (epoch + 1) % 10 == 0:
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            base_name = f"epoch_{epoch}_model.pkl"
            full_name = os.path.join(experiment_name, base_name)
            paddle.save(obj=state, path=full_name, protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-params")
    parser.add_argument(
        "--data_path", nargs="?", type=str, default="", help="Data path to load data"
    )
    parser.add_argument(
        "--img_rows", nargs="?", type=int, default=288, help="Height of the input image"
    )
    parser.add_argument(
        "--img_cols", nargs="?", type=int, default=288, help="Width of the input image"
    )
    parser.add_argument(
        "--epochs", nargs="?", type=int, default=56, help="# of the epochs"
    )
    parser.add_argument(
        "--batch_size", nargs="?", type=int, default=24, help="Batch Size"
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

    # dist.spawn(train, args=(args,))

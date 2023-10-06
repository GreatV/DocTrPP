import copy
import os

import matplotlib.pyplot as plt
import paddle.optimizer as optim

from GeoTr import GeoTr


def plot_lr_scheduler(optimizer, scheduler, epochs=65, save_dir=""):
    """
    Plot the learning rate scheduler
    """

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    lr = []
    for _ in range(epochs):
        for _ in range(30):
            lr.append(scheduler.get_lr())
            optimizer.step()
            scheduler.step()

    epoch = [float(i) / 30.0 for i in range(len(lr))]

    plt.figure()
    plt.plot(epoch, lr, ".-", label="Learning Rate")
    plt.xlabel("epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Scheduler")
    plt.savefig(os.path.join(save_dir, "lr_scheduler.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    model = GeoTr()

    schaduler = optim.lr.OneCycleLR(
        max_learning_rate=1e-4,
        total_steps=1950,
        phase_pct=0.1,
        end_learning_rate=1e-4 / 2.5e5,
    )
    optimizer = optim.AdamW(learning_rate=schaduler, parameters=model.parameters())
    plot_lr_scheduler(
        scheduler=schaduler, optimizer=optimizer, epochs=65, save_dir="./"
    )

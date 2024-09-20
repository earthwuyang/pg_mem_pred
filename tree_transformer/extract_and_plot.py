import re
import matplotlib.pyplot as plt
import os
def main(dataset):
    # 初始化存储数据的字典
    epochs = []
    train_losses = []
    validation_losses = []
    qerror_50 = []

    # 定义正则表达式以提取数据
    epoch_pattern = re.compile(r"\[.*\]:Epoch (\d+)/")
    train_val_loss_pattern = re.compile(r"Train loss: ([\d\.]+) - Val loss: ([\d\.]+)")
    qerror_pattern = re.compile(r"'qerror_50 \(Median\)': ([\d\.]+)")

    # 读取日志文件
    # log_file = f'train_{dataset}.nohup'
    if dataset == 'tpch':
        log_file = f'logs/train_tpch_2024_09_17_20_56_29_109341.log'
    elif dataset == 'tpcds':
        log_file = f'logs/train_tpcds_2024_09_17_20_57_14_312796.log'

    with open(log_file, 'r') as file:
        for line in file:
            # 检查是否为新的epoch
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                epochs.append(current_epoch)
                # continue  # 继续读取下一行

            # 提取Train Loss和Validation Loss
            loss_match = train_val_loss_pattern.search(line)
            # print(f"loss_match: {loss_match}")
            if loss_match:
                train_loss = float(loss_match.group(1))
                val_loss = float(loss_match.group(2))
                train_losses.append(train_loss)
                validation_losses.append(val_loss)
                continue  # 继续读取下一行

            # 提取qerror_50
            qerror_match = qerror_pattern.search(line)
            if qerror_match:
                q_error = float(qerror_match.group(1))
                qerror_50.append(q_error)
                continue  # 继续读取下一行

    # print(f"len(epochs): {len(epochs)}, len(train_losses): {len(train_losses)}, len(validation_losses): {len(validation_losses)}, len(qerror_50): {len(qerror_50)}")
    # exit()
    # 检查数据是否完整
    if not (len(epochs) == len(train_losses) == len(validation_losses) == len(qerror_50)):
        print("警告：提取的数据长度不一致，可能有遗漏。")

    # 打印提取的数据
    for epoch, train, val, q50 in zip(epochs, train_losses, validation_losses, qerror_50):
        print(f"Epoch {epoch}: Train Loss = {train}, Validation Loss = {val}, qerror_50 = {q50}")

    # 绘制双 y 轴图表
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss / Validation Loss', color=color)
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o')
    ax1.plot(epochs, validation_losses, label='Validation Loss', color='green', marker='o')
    ax1.tick_params(axis='y', labelcolor=color)  # This sets the color of the tick labels on the left Y-axis (for Train Loss and Validation Loss) to match the label color.

    ax2 = ax1.twinx()  # 创建第二个y轴
    color = 'tab:red'
    ax2.set_ylabel('qerror_50 (Median)', color=color)
    ax2.plot(epochs, qerror_50[:len(epochs)], label='qerror_50 (Median)', color='red', marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加图例
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    plt.title('Training Metrics Over Epochs')
    fig.tight_layout()
    dir = 'figures'
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig_path = os.path.join(dir, f'{dataset}_training_curve.png')
    plt.savefig(fig_path)  # 保存图表
    # plt.show()


if __name__ == '__main__':
    main('tpch')
    main('tpcds')
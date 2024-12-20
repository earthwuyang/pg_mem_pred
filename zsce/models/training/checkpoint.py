import csv
import os
import shutil
import time

import joblib
import torch


def torch_save(logger, checkpoint, target_path, model_name, postfix='', filetype='.m'):
    save_start_t = time.perf_counter()
    if not (target_path is None or model_name is None):
        os.makedirs(target_path, exist_ok=True)
        # make saving atomic by first writing a temp file and then renaming it
        target_temp_m_path = os.path.join(target_path, f'temp_{model_name}{postfix}{filetype}')
        torch.save(checkpoint, target_temp_m_path)

        target_m_path = os.path.join(target_path, f'{model_name}{postfix}{filetype}')
        shutil.move(target_temp_m_path, target_m_path)

        logger.info(f"Saved checkpoint to {target_m_path} in {time.perf_counter() - save_start_t:.3f} secs")
    else:
        logger.info("Skipping saving")


def save_csv(csv_rows, target_csv_path):
    os.makedirs(os.path.dirname(target_csv_path), exist_ok=True)

    # make sure the first row contains all possible keys. Otherwise dictwriter raises an error.
    for csv_row in csv_rows:
        for key in csv_row.keys():
            if key not in csv_rows[0].keys():
                csv_rows[0][key] = None

    with open(target_csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, csv_rows[0].keys())
        for i, row in enumerate(csv_rows):
            if i == 0:
                w.writeheader()
            w.writerow(row)


def save_checkpoint(logger, epochs_wo_improvement, epoch, model, optimizer, target_path, model_name, metrics,
                    csv_stats, finished=False):
    checkpoint = {
        'epochs_wo_improvement': epochs_wo_improvement,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'finished': finished
    }
    if metrics is not None:
        for m in metrics:
            checkpoint[m.metric_name + '_best_model'] = m.best_model
            checkpoint[m.metric_name + '_best_seen_value'] = m.best_seen_value

    torch_save(logger, checkpoint, target_path, model_name, filetype='.pt')

    if not (target_path is None or model_name is None):
        target_csv_path = os.path.join(target_path, f'{model_name}.csv')
        save_csv(csv_stats, target_csv_path)
        # if model.label_norm is not None:
        #     label_norm_path = os.path.join(target_path, f'{model_name}_label_norm.pkl')
        #     joblib.dump(model.label_norm, label_norm_path, compress=1)
        if model.mem_norm is not None:
            mem_norm_path = os.path.join(target_path, f'{model_name}_mem_norm.pkl')
            joblib.dump(model.mem_norm, mem_norm_path, compress=1)
        if model.time_norm is not None:
            time_norm_path = os.path.join(target_path, f'{model_name}_time_norm.pkl')            
            joblib.dump(model.time_norm, time_norm_path, compress=1)
    else:
        logger.info("Skipping saving the epoch stats")


def load_stats_csv(target_csv_path):
    with open(target_csv_path) as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = next(reader)
        csv_rows = [dict(zip(header, row)) for row in reader]

    return csv_rows


def load_checkpoint(logger, model, target_path, model_name, filetype='.pt', optimizer=None, metrics=None):
    """
    Load all training state from a checkpoint. Does not work across devices (e.g., GPU->CPU).
    """
    load_start_t = time.perf_counter()
    epochs_wo_improvement = 0
    epoch = 0
    finished = False
    csv_stats = []

    if target_path is not None and model_name is not None:
        try:
            checkpoint = torch.load(os.path.join(target_path, f'{model_name}{filetype}'))
            epochs_wo_improvement = checkpoint['epochs_wo_improvement']
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])

            # read label normalizer
            label_norm_path = os.path.join(target_path, f'{model_name}_label_norm.pkl')  # model_name = args.filename_model
            if os.path.exists(label_norm_path):
                model.label_norm = joblib.load(label_norm_path)

            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            finished = checkpoint['finished']

            if metrics is not None:
                for m in metrics:
                    m.best_model = checkpoint[m.metric_name + '_best_model']
                    m.best_seen_value = checkpoint[m.metric_name + '_best_seen_value']

            target_csv_path = os.path.join(target_path, f'{model_name}.csv')
            csv_stats = load_stats_csv(target_csv_path)
            epoch += 1

            logger.info(f"Successfully loaded checkpoint from epoch {epoch} ({len(csv_stats)} csv rows) "
                  f"in {time.perf_counter() - load_start_t:.3f} secs") 
            # 用于返回当前系统的高精度性能计数器的值。它返回一个表示自程序运行以来经过的时间的浮点数。time.perf_counter()函数通常用于性能计时和时间差的测量。它返回的值的单位取决于具体的操作系统，但通常是秒或毫秒的精确度。这个函数返回的值是在整个系统中是单调递增的，因此可以用来衡量不同操作的执行时间。

        # reset to defaults if something goes wrong
        except Exception as e:
            epochs_wo_improvement = 0
            epoch = 0
            finished = False
            csv_stats = []
            logger.info(f"No valid checkpoint found {e}")
    else:
        logger.info("Filetype or model name are None")

    return csv_stats, epochs_wo_improvement, epoch, model, optimizer, metrics, finished

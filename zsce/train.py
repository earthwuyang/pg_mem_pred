import os
import time
from copy import copy
import logging
import numpy as np
import optuna
import torch
import torch.optim as opt
from tqdm import tqdm
from datetime import datetime
import argparse
import json
import random
import pandas as pd
import sys

from cross_db_benchmark.benchmark_tools.utils import load_json

from models.dataset.dataset_creation import create_dataloader
from models.training.checkpoint import save_checkpoint, load_checkpoint, save_csv
from models.training.metrics import MRE, RMSE, QError, MeanQError
from models.training.utils import batch_to, flatten_dict, find_early_stopping_metric
from models.zero_shot_models.specific_models.model import zero_shot_models
from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from combine_stats import combine_stats

from get_raw_plans import get_raw_plans
from parse_plans import parse_raw
from split_parsed_plans import split_dataset
from gather_feature_statistics import gather_feature_statistics

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.preprocessing.extract_mem_time_info import extract_mem_info



def get_logger(logfile):

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    fmt = f"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]:%(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    fh=logging.FileHandler(logfile)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    log.addHandler(fh)
    return log

def train_epoch(logger, epoch_stats, train_loader, model, optimizer, max_epoch_tuples, mem_pred, time_pred, custom_batch_to=batch_to):
    model.train()

    # run remaining batches
    train_start_t = time.perf_counter()
    losses = []
    errs = []
    errs_mem = []
    errs_time = []
    # for batch_idx, batch in enumerate(tqdm(train_loader)):
    for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        if max_epoch_tuples is not None and batch_idx * train_loader.batch_size > max_epoch_tuples:
            break

        input_model, mem_label, time_label, sample_idxs = custom_batch_to(batch, model.device, model.mem_norm, model.time_norm) 
        # graph, dictionaries = input_model
        # print(graph, dictionaries.keys()) # dict_keys(['column', 'table', 'output_column', 'filter_column', 'plan0', 'plan1', 'plan2', 'plan3', 'plan4', 'plan5', 'plan6', 'plan7', 'plan8', 'plan9', 'plan10', 'plan11', 'plan12', 'plan13', 'logical_pred_0', 'logical_pred_1'])
        # while 1:pass
        optimizer.zero_grad()
        output = model(input_model)
        # logger.info(f"output shape: {output.shape}")
        output_mem = output[:, 0].reshape(-1, 1)
        output_time = output[:, 1].reshape(-1, 1)
        # print(f"output_mem {output_mem}, output_time {output_time}")
        # while 1:pass
        # print(f"output {output}, label {label}")
        # print(f"output shape: {output.shape}, label shape: {label.shape}")  # output shape: torch.Size([2048, 1]), label shape: torch.Size([2048])
        mem_loss = model.loss_fxn(output_mem, mem_label)
        time_loss = model.loss_fxn(output_time, time_label)
        # logger.info(f"mem_loss: {mem_loss}, time_loss: {time_loss}")
        # while 1:pass

        if mem_pred:
            loss = mem_loss
        if time_pred:
            loss = time_loss
        if mem_pred and time_pred:
            alpha = 1
            loss = mem_loss + alpha * time_loss
        if torch.isnan(loss):
            raise ValueError('Loss was NaN')
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        # output = output.detach().cpu().numpy().reshape(-1)
        # label = label.detach().cpu().numpy().reshape(-1)
        # errs = np.concatenate((errs, output - label))
        output_mem = output_mem.detach().cpu().numpy().reshape(-1)
        output_time = output_time.detach().cpu().numpy().reshape(-1)
        mem_label = mem_label.detach().cpu().numpy().reshape(-1)
        time_label = time_label.detach().cpu().numpy().reshape(-1)
        # print(f"output_mem {output_mem}, output_time {output_time}")
        errs_mem = np.concatenate((errs_mem, output_mem - mem_label))
        errs_time = np.concatenate((errs_time, output_time - time_label))
        # print(f"errs_mem {errs_mem}, errs_time {errs_time}")
        losses.append(loss)

    mean_loss = np.mean(losses)
    mean_rmse_mem = np.sqrt(np.mean(np.square(errs_mem)))
    mean_rmse_time = np.sqrt(np.mean(np.square(errs_time)))
    # print(f"Train Loss: {mean_loss:.2f}")
    # print(f"Train RMSE: {mean_rmse:.2f}")
    epoch_stats.update(train_time=time.perf_counter() - train_start_t, mean_loss=mean_loss, mean_rmse_mem=mean_rmse_mem, mean_rmse_time=mean_rmse_time)


def validate_model(logger, val_loader, model, epoch=0, epoch_stats=None, metrics=None, max_epoch_tuples=None,
                   mem_pred = True,  time_pred = False, 
                   custom_batch_to=batch_to, verbose=False, log_all_queries=False):
    model.eval()

    with torch.autograd.no_grad():
        val_loss = torch.Tensor([0])
        val_loss_mem = torch.Tensor([0])
        val_loss_time = torch.Tensor([0])
        mem_preds = []
        time_preds = []
        mem_labels = []
        time_labels = []
        probs = []
        sample_idxs = []

        # evaluate test set using model
        test_start_t = time.perf_counter()
        val_num_tuples = 0
        # for batch_idx, batch in enumerate(tqdm(val_loader)):
        for batch_idx, batch in enumerate(val_loader):
            if max_epoch_tuples is not None and batch_idx * val_loader.batch_size > max_epoch_tuples:
                break

            val_num_tuples += val_loader.batch_size

            input_model, mem_label, time_label, sample_idxs_batch = custom_batch_to(batch, model.device, model.mem_norm, model.time_norm)  
            sample_idxs += sample_idxs_batch
            output = model(input_model)
            output_mem = output[:, 0].reshape(-1, 1)
            output_time = output[:, 1].reshape(-1, 1)
            # logging.info(f"test one batch time: {time.perf_counter() - test_start_t} seconds")
            # while 1:pass

            # sum up mean batch losses
            # val_loss += model.loss_fxn(output, label).cpu()
            val_loss_mem += model.loss_fxn(output_mem, mem_label).cpu()
            val_loss_time += model.loss_fxn(output_time, time_label).cpu()

            # inverse transform the predictions and labels
            # curr_pred = output.cpu().numpy()
            curr_pred_mem = output_mem.cpu().numpy()
            curr_pred_time = output_time.cpu().numpy()
            # curr_label = label.cpu().numpy()
            curr_mem_label = mem_label.cpu().numpy()
            curr_time_label = time_label.cpu().numpy()
            # print(f"curr_pred {curr_pred}, curr_label {curr_label}")
            if model.mem_norm is not None:
                curr_pred_mem = model.mem_norm.inverse_transform(curr_pred_mem)
                curr_mem_label = model.mem_norm.inverse_transform(curr_mem_label.reshape(-1, 1)).reshape(-1)
                
            if model.time_norm is not None:
                    curr_pred_time = model.time_norm.inverse_transform(curr_pred_time)
                    curr_time_label = model.time_norm.inverse_transform(curr_time_label.reshape(-1, 1)).reshape(-1)
               
            mem_preds.append(curr_pred_mem.reshape(-1))
            time_preds.append(curr_pred_time.reshape(-1))
            mem_labels.append(curr_mem_label.reshape(-1))
            time_labels.append(curr_time_label.reshape(-1))

        if epoch_stats is not None:
            epoch_stats.update(val_time=time.perf_counter() - test_start_t)
            epoch_stats.update(val_num_tuples=val_num_tuples)
            # val_loss = (val_loss.cpu() / len(val_loader)).item()
            # logger.info(f'val_loss epoch {epoch}: {val_loss}')
            # epoch_stats.update(val_loss=val_loss)
            val_loss_mem = (val_loss_mem.cpu() / len(val_loader)).item()
            val_loss_time = (val_loss_time.cpu() / len(val_loader)).item()
            logger.info(f'val_loss_mem epoch {epoch}: {val_loss_mem}')
            logger.info(f'val_loss_time epoch {epoch}: {val_loss_time}')
            epoch_stats.update(val_loss_mem=val_loss_mem, val_loss_time=val_loss_time)

        # labels = np.concatenate(labels, axis=0)
        # preds = np.concatenate(preds, axis=0)
        mem_preds = np.concatenate(mem_preds, axis=0)
        time_preds = np.concatenate(time_preds, axis=0)
        mem_labels = np.concatenate(mem_labels, axis=0)
        time_labels = np.concatenate(time_labels, axis=0)
        if verbose:
            logger.debug(f'mem labels: {mem_labels}')
            logger.debug(f'time labels: {time_labels}')
            logger.debug(f'mem preds: {mem_preds}')
            logger.debug(f'time preds: {time_preds}')
        epoch_stats.update(val_std_mem=np.std(mem_labels), val_std_time=np.std(time_labels))
        if log_all_queries:
            epoch_stats.update(val_mem_labels=[float(f) for f in mem_labels], val_time_labels=[float(f) for f in time_labels], val_mem_preds=[float(f) for f in mem_preds], val_time_preds=[float(f) for f in time_preds], val_sample_idxs=sample_idxs)
            epoch_stats.update(val_mem_preds=[float(f) for f in mem_preds], val_time_preds = [float(f) for f in time_preds])
            epoch_stats.update(val_sample_idxs=sample_idxs)

        # save best model for every metric
        any_best_metric = False
        if metrics is not None:
            logger.info(f"evaluating memory prediction")
            for metric in metrics:
                best_seen = metric.evaluate(metrics_dict=epoch_stats, model=model, labels=mem_labels, preds=mem_preds,
                                            probs=probs)
                if best_seen and metric.early_stopping_metric:
                    any_best_metric = True
                    logger.info(f"New best model for {metric.metric_name}")
            logging.info(f"\nevaluating time prediction")
            for metric in metrics:
                metric.evaluate(metrics_dict=epoch_stats, model=model, labels=time_labels, preds=time_preds,
                                            probs=probs)

    return any_best_metric


def optuna_intermediate_value(metrics):
    for m in metrics:
        if m.early_stopping_metric:
            assert isinstance(m, QError)
            return m.best_seen_value
    raise ValueError('Metric invalid')


def train_model(logger, data_dir, train_workload_runs, val_workload_runs, test_workload_runs, combined_stats,
                target_dir,
                filename_model,
                optimizer_class_name='Adam',
                optimizer_kwargs=None,
                final_mlp_kwargs=None,
                node_type_kwargs=None,
                model_kwargs=None,  
                tree_layer_name='GATConv',
                tree_layer_kwargs=None,
                hidden_dim=32,
                batch_size=4096,
                output_dim=1,
                epochs=0,
                device='cuda:0',
                plan_featurization_name=None,
                max_epoch_tuples=100000,
                param_dict=None,  # param_dict=param_dict
                num_workers=8,
                early_stopping_patience=20,
                trial=None,
                database=None,
                mem_pred = True,
                time_pred = False,
                limit_queries=None,
                limit_queries_affected_wl=None,
                skip_train=False):
    if model_kwargs is None: # model_kwargs is not passed from the call of train_model(), thus None
        model_kwargs = dict()

    

    target_test_csv_paths = []
    p=test_workload_runs

    test_workload = os.path.basename(p).replace('.json', '')
    target_test_csv_paths.append(os.path.join(target_dir, f'test_{filename_model}_{test_workload}.csv'))

    if len(target_test_csv_paths) > 0 and all([os.path.exists(p) for p in target_test_csv_paths]):
        logger.info(f"Model was already trained and tested ({target_test_csv_paths} exists)")
        return

    # create a dataset
    loss_class_name = final_mlp_kwargs['loss_class_name']
    mem_norm, time_norm, feature_statistics, train_loader, val_loader, test_loaders = \
        create_dataloader(logger, data_dir, train_workload_runs, val_workload_runs, test_workload_runs, combined_stats, plan_featurization_name, database,
                          val_ratio=0, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=False, limit_queries=limit_queries,
                          limit_queries_affected_wl=limit_queries_affected_wl, loss_class_name=loss_class_name)
    
    if loss_class_name == 'QLoss':
        metrics = [RMSE(), MRE(), QError(percentile=50, early_stopping_metric=True), QError(percentile=95),
                   QError(percentile=100), MeanQError()]
    elif loss_class_name == 'MSELoss':
        metrics = [RMSE(early_stopping_metric=True), MRE(), QError(percentile=50), QError(percentile=95),
                   QError(percentile=100), MeanQError()]

    # create zero shot model dependent on database
    model = zero_shot_models[database](device=device, hidden_dim=hidden_dim, final_mlp_kwargs=final_mlp_kwargs,
                                       node_type_kwargs=node_type_kwargs, output_dim=output_dim,
                                       feature_statistics=feature_statistics, tree_layer_name=tree_layer_name,
                                       tree_layer_kwargs=tree_layer_kwargs,
                                       plan_featurization_name=plan_featurization_name,
                                       mem_norm=mem_norm, time_norm=time_norm,
                                       **model_kwargs)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params}")
    # 计算模型的总大小
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()  # numel() 是元素总数, element_size() 是每个元素的字节数

    # 转换为 KB, MB
    logger.info(f"Model total size: {total_size / 1024:.2f} KB")
    logger.info(f"Model total size: {total_size / (1024 ** 2):.2f} MB")
    
    # move to gpu
    model = model.to(model.device)
    # print(model)
    optimizer = opt.__dict__[optimizer_class_name](model.parameters(), **optimizer_kwargs)  # import torch.optim as opt

    csv_stats, epochs_wo_improvement, epoch, model, optimizer, metrics, finished = \
        load_checkpoint(logger, model, target_dir, filename_model, optimizer=optimizer, metrics=metrics, filetype='.pt')

    # train an actual model (q-error? or which other loss?)
    while epoch < epochs and not finished and not skip_train:
        logger.info(f"Epoch {epoch}")

        epoch_stats = copy(param_dict)
        epoch_stats.update(epoch=epoch)
        epoch_start_time = time.perf_counter()
        # try:
        train_epoch(logger, epoch_stats, train_loader, model, optimizer, max_epoch_tuples, mem_pred, time_pred)

        any_best_metric = validate_model(logger, val_loader, model, epoch=epoch, epoch_stats=epoch_stats, metrics=metrics,
                                         max_epoch_tuples=max_epoch_tuples, mem_pred=mem_pred, time_pred=time_pred)
        epoch_stats.update(epoch=epoch, epoch_time=time.perf_counter() - epoch_start_time)

        # report to optuna
        if trial is not None:
            intermediate_value = optuna_intermediate_value(metrics)
            epoch_stats['optuna_intermediate_value'] = intermediate_value

            logger.info(f"Reporting epoch_no={epoch}, intermediate_value={intermediate_value} to optuna "
                  f"(Trial {trial.number})")
            trial.report(intermediate_value, epoch)

        # see if we can already stop the training
        stop_early = False
        if not any_best_metric:
            epochs_wo_improvement += 1
            if early_stopping_patience is not None and epochs_wo_improvement > early_stopping_patience:
                stop_early = True
        else:
            epochs_wo_improvement = 0
        if trial is not None and trial.should_prune():
            stop_early = True
        # also set finished to true if this is the last epoch
        if epoch == epochs - 1:
            stop_early = True

        epoch_stats.update(stop_early=stop_early)
        logger.info(f"epochs_wo_improvement: {epochs_wo_improvement}")

        # save stats to file
        csv_stats.append(epoch_stats)

        # save current state of training allowing us to resume if this is stopped
        save_checkpoint(logger, epochs_wo_improvement, epoch, model, optimizer, target_dir,
                        filename_model, metrics=metrics, csv_stats=csv_stats, finished=stop_early)

        epoch += 1

        # Handle pruning based on the intermediate value.
        if trial is not None and trial.should_prune():
            raise optuna.TrialPruned()

        if stop_early:
            logger.info(f"Early stopping kicked in due to no improvement in {early_stopping_patience} epochs")
            break
        # except:
        #     print("Error during epoch. Trying again.")

    # if we are not doing hyperparameter search, evaluate test set
    if trial is None and test_loaders is not None:
        if not (target_dir is None or filename_model is None):
            assert len(target_test_csv_paths) == len(test_loaders), f"target_test_csv_paths length {len(target_test_csv_paths)} != test_loaders length {len(test_loaders)}"
            for test_path, test_loader in zip(target_test_csv_paths, test_loaders):
                logger.info(f"Starting validation for {test_path}")
                test_stats = copy(param_dict)

                early_stop_m = find_early_stopping_metric(metrics)
                logger.info("Reloading best model")
                # model.load_state_dict(early_stop_m.best_model)
                validate_model(logger, test_loader, model, epoch=epoch, epoch_stats=test_stats, metrics=metrics,
                               log_all_queries=True)

                save_csv([test_stats], test_path)

        else:
            logger.info("Skipping saving the test stats")

    if trial is not None:
        return optuna_intermediate_value(metrics)
    

if __name__ == '__main__':

    seed = 1
    # seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Directory where the data is stored", default='/home/wuy/DB/pg_mem_data/')
    parser.add_argument("--dataset", type=str, nargs='+', help="Datasets to use for training", default=['tpch_sf1'])
    parser.add_argument("--val_dataset", type=str, help="Dataset to use for validation", default=None)
    parser.add_argument("--test_dataset", type=str, help="Dataset to use for test", default=None)
    parser.add_argument("--skip_train", action='store_true', help="Skip training and only evaluate test set")
    parser.add_argument("--force", action='store_true', help="Force overwrite of existing files")
    parser.add_argument('--mem_pred', action='store_true', default=True, help='predict memory')
    parser.add_argument('--no_mem_pred', action='store_false', dest='mem_pred', help='do not predict memory')
    parser.add_argument('--time_pred', action='store_true', default=False, help='predict time')
    parser.add_argument('--debug', action='store_true', help="Debug mode")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for data loading")
    args = parser.parse_args()

    if args.test_dataset is None:
        assert len(args.dataset) == 1, "If test_dataset is not specified, only one dataset can be used for training and validation"
        args.val_dataset = args.dataset[0]
        args.test_dataset = args.dataset[0]
    args.train_dataset = args.dataset

    hyperparameter_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'setup/tuned_hyperparameters/tune_est_best_config.json')
    # hyperparameter_path = 'setup/tuned_hyperparameters/tune_est_best_config.json'
    hyperparams = load_json(hyperparameter_path, namespace=False)

    # loss_class_name='QLoss'
    loss_class_name='MSELoss'
    max_epoch_tuples=100000
    seed = 0
    device = 'cuda:0'
    num_workers = args.num_workers
    limit_queries=None
    limit_queries_affected_wl=None
    skip_train=False
    max_no_epochs = None

    p_dropout = hyperparams.pop('p_dropout')
    # general fc out
    fc_out_kwargs = dict(p_dropout=p_dropout,
                            activation_class_name='LeakyReLU',
                            activation_class_kwargs={},
                            norm_class_name='Identity',
                            norm_class_kwargs={},
                            residual=hyperparams.pop('residual'),
                            dropout=hyperparams.pop('dropout'),
                            activation=True,
                            inplace=True)
    final_mlp_kwargs = dict(width_factor=hyperparams.pop('final_width_factor'),
                            n_layers=hyperparams.pop('final_layers'),
                            loss_class_name=loss_class_name,
                            loss_class_kwargs=dict())
    tree_layer_kwargs = dict(width_factor=hyperparams.pop('tree_layer_width_factor'),
                                n_layers=hyperparams.pop('message_passing_layers'))
    node_type_kwargs = dict(width_factor=hyperparams.pop('node_type_width_factor'),
                            n_layers=hyperparams.pop('node_layers'),
                            one_hot_embeddings=True,
                            max_emb_dim=hyperparams.pop('max_emb_dim'),
                            drop_whole_embeddings=False)
    final_mlp_kwargs.update(**fc_out_kwargs)
    tree_layer_kwargs.update(**fc_out_kwargs)
    node_type_kwargs.update(**fc_out_kwargs)


    train_kwargs = dict(optimizer_class_name='AdamW',
                        optimizer_kwargs=dict(
                            lr=hyperparams.pop('lr'),
                        ),
                        final_mlp_kwargs=final_mlp_kwargs,
                        node_type_kwargs=node_type_kwargs,
                        tree_layer_kwargs=tree_layer_kwargs,
                        tree_layer_name=hyperparams.pop('tree_layer_name'),
                        plan_featurization_name=hyperparams.pop('plan_featurization_name'),  # 'PostgresEstSystemCardDetail' in tune_est_best_config.json, while 'PostgresTrueCardDetail' is the default as defined in train_default(), the third one is 'PostgresDeepDBEstSystemCardDetail'
                        hidden_dim=hyperparams.pop('hidden_dim'),
                        output_dim=2,
                        epochs=200 if max_no_epochs is None else max_no_epochs,
                        early_stopping_patience=20,
                        max_epoch_tuples=max_epoch_tuples,
                        batch_size=hyperparams.pop('batch_size'),
                        device=device,
                        num_workers=num_workers,
                        limit_queries=limit_queries,
                        limit_queries_affected_wl=limit_queries_affected_wl,
                        skip_train=args.skip_train
                        )

    # assert len(hyperparams) == 0, f"Not all hyperparams were used (not used: {hyperparams.keys()}). Hence generation " \
                                    # f"and reading does not seem to fit"

    data_dir = args.data_dir
    
    # statistics_file = os.path.join(data_dir, args.train_dataset, 'zsce', 'statistics_workload_combined.json')
    # statistics_file = '/home/wuy/DB/pg_mem_pred/tpch_data/statistics_workload_combined.json' # CAUTION
    target_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'evaluation_train_{"".join(args.train_dataset)}_test_{args.test_dataset}_{"mem" if args.mem_pred else ""}_{"time" if args.time_pred else ""}')
    filename_model = f'{"".join(args.train_dataset)}'
    database = DatabaseSystem.POSTGRES

    param_dict = flatten_dict(train_kwargs)  # https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    # train_workload_runs = os.path.join(data_dir, args.train_dataset, 'zsce', 'train_plans.json')
    # val_workload_runs = os.path.join(data_dir, args.train_dataset, 'zsce', 'val_plans.json')
    # test_workload_runs = os.path.join(data_dir, args.test_dataset, 'zsce', 'test_plans.json')
    train_workload_runs = args.train_dataset
    val_workload_runs = args.train_dataset
    test_workload_runs = args.test_dataset

    if args.debug:
        train_workload_runs = os.path.join(data_dir, args.train_dataset, 'zsce', 'tiny_plans.json')
        val_workload_runs = os.path.join(data_dir, args.train_dataset, 'zsce', 'tiny_plans.json')
        test_workload_runs = os.path.join(data_dir, args.test_dataset, 'zsce', 'tiny_plans.json')
    # train_workload_runs = '/home/wuy/DB/pg_mem_pred/tpch_data/val_plans.json'
    # val_workload_runs = '/home/wuy/DB/pg_mem_pred/tpch_data/val_plans.json'
    # test_workload_runs = '/home/wuy/DB/pg_mem_pred/tpch_data/val_plans.json'

    logfilepath = os.path.join('logs', f'train_{"".join(args.train_dataset)}_test_{args.test_dataset}_{"mem" if args.mem_pred else ""}_{"time" if args.time_pred else ""}.log')
    if not os.path.exists(logfilepath):
        os.system(f"mkdir -p {logfilepath}")
    logfile = os.path.join(logfilepath, f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}.log")
    logger = get_logger(logfile)

    # logger.info(f"extracting mem time info...")
    # if not os.path.exists(os.path.join(args.data_dir, args.dataset, 'raw_data', 'mem_info.csv')):
    #     extract_mem_info(args.data_dir, args.dataset)
    # else:
    #     logger.info(f"mem_info.csv already exists, skipping extraction")

    for dataset in args.dataset + [args.val_dataset, args.test_dataset]:
        logger.info(f"get raw plans for {dataset}...")
        if args.force or not os.path.exists(os.path.join(args.data_dir, dataset, 'zsce', 'raw_plan.json')):
            get_raw_plans(args.data_dir, dataset)
        else:
            logger.info(f"raw_plan.json already exists, skipping extraction")

        logger.info(f"parsing plans...")
        if args.force or not os.path.exists(os.path.join(args.data_dir, dataset, 'zsce', 'parsed_plan.json')):
            parse_raw(args.data_dir, dataset)
        else:
            logger.info(f"parsed_plan.json already exists, skipping parsing")

        logger.info(f"spliting parsed plans...")
        if args.force or not (os.path.exists(os.path.join(args.data_dir, dataset, 'zsce', 'train_plans.json')) 
            and os.path.exists(os.path.join(args.data_dir, dataset, 'zsce', 'val_plans.json')) 
            and os.path.exists(os.path.join(args.data_dir, dataset, 'zsce', 'test_plans.json'))):
            split_dataset(args.data_dir, dataset)
        else:
            logger.info(f"train_plans.json, val_plans.json, test_plans.json already exists, skipping splitting")

    combined_stats = combine_stats(logger, args)
    # with open(os.path.join(args.data_dir, 'zsce_combined_statistics_workload.json'),'r') as f:
    #     combined_stats = json.load(f)

    # logger.info(f"gathering faeture statistics from train_plans.json...")
    # if args.force or not os.path.exists(os.path.join(args.data_dir, args.dataset, 'zsce','statistics_workload_combined.json')):
    #     gather_feature_statistics(args.data_dir, args.dataset)
    # else:
    #     logger.info(f"statistics_workload_combined.json already exists, skipping gathering feature statistics")
    
    train_model(logger, data_dir, train_workload_runs, val_workload_runs, test_workload_runs, combined_stats, target_dir, filename_model,
                param_dict=param_dict, database=database, mem_pred=args.mem_pred, time_pred=args.time_pred, **train_kwargs)
        

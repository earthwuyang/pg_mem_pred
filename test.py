from models.training.utils import batch_to as custom_batch_to
from models.training.utils import flatten_dict
import torch
import numpy as np
import json
import os
import pandas as pd
import logging
import models.dataset.dataset_creation as dataset_creation
from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
import importlib
from models.zero_shot_models.specific_models.model import zero_shot_models

from cross_db_benchmark.benchmark_tools.utils import load_json
from tqdm import tqdm
hyperparameter_path = 'setup/tuned_hyperparameters/tune_est_best_config.json'
hyperparams = load_json(hyperparameter_path, namespace=False)

loss_class_name='QLoss'
# loss_class_name='MSELoss'
max_epoch_tuples=100000
seed = 0
device = 'cpu'
num_workers = 1
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
                output_dim=1,
                epochs=200 if max_no_epochs is None else max_no_epochs,
                early_stopping_patience=20,
                max_epoch_tuples=max_epoch_tuples,
                batch_size=hyperparams.pop('batch_size'),
                device=device,
                num_workers=num_workers,
                seed=seed,
                limit_queries=limit_queries,
                limit_queries_affected_wl=limit_queries_affected_wl,
                skip_train=skip_train
                )

assert len(hyperparams) == 0, f"Not all hyperparams were used (not used: {hyperparams.keys()}). Hence generation " \
                                    f"and reading does not seem to fit"

# Set up the parameters for the experiment
train_dataset = 'tpcds'
test_dataset = 'tpcds'
target_dir = f'evaluation_train_{train_dataset}_test_{test_dataset}'
statistics_file = f'{train_dataset}_data/statistics_workload_combined.json'

workload_runs = f'{train_dataset}_data/train_plans.json'
test_workload_runs = f'{test_dataset}_data/val_plans.json'

param_dict = flatten_dict(train_kwargs)


def get_logger():

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    fmt = f"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]:%(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log

logger = get_logger()

seed = 0
# seed for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

plan_featurization_name = train_kwargs['plan_featurization_name']
database = DatabaseSystem.POSTGRES
batch_size = train_kwargs['batch_size']
num_workers = train_kwargs['num_workers']
limit_queries = train_kwargs['limit_queries']
limit_queries_affected_wl = train_kwargs['limit_queries_affected_wl']
final_mlp_kwargs = train_kwargs['final_mlp_kwargs']
loss_class_name = final_mlp_kwargs['loss_class_name']

importlib.reload(dataset_creation)
label_norm, feature_statistics, train_loader, val_loader, test_loader = \
dataset_creation.create_dataloader(logger, workload_runs, test_workload_runs, statistics_file, plan_featurization_name, database,
                    val_ratio=0.15, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    pin_memory=False, limit_queries=limit_queries,
                    limit_queries_affected_wl=limit_queries_affected_wl, loss_class_name=loss_class_name)
print(f"train_loader length: {len(train_loader.dataset)}")
print(f"val_loader length: {len(val_loader.dataset)}")
print(f"test_loader length: {len(test_loader.dataset)}")




model_kwargs = dict()

hidden_dim = train_kwargs['hidden_dim']
output_dim = train_kwargs['output_dim']
tree_layer_name = train_kwargs['tree_layer_name']
model = zero_shot_models[database](device=device, hidden_dim=hidden_dim, final_mlp_kwargs=final_mlp_kwargs,
                                       node_type_kwargs=node_type_kwargs, output_dim=output_dim,
                                       feature_statistics=feature_statistics, tree_layer_name=tree_layer_name,
                                       tree_layer_kwargs=tree_layer_kwargs,
                                       plan_featurization_name=plan_featurization_name,
                                       label_norm=label_norm,
                                       **model_kwargs)
    # move to gpu
model = model.to(model.device)

checkpoint = torch.load(f'evaluation/{train_dataset}.pt')
model.load_state_dict(checkpoint['model'])

model.eval()
labels = []
preds =[]
with torch.autograd.no_grad():
    for batch in tqdm(test_loader):
        input_model, label, sample_idxs_batch = custom_batch_to(batch, model.device, model.label_norm)
        output = model(input_model)
        curr_pred = output.cpu().numpy()
        curr_label = label.cpu().numpy()
        if model.label_norm is not None:
            curr_pred = model.label_norm.inverse_transform(curr_pred)
            curr_label = model.label_norm.inverse_transform(curr_label.reshape(-1, 1))
            curr_label = curr_label.reshape(-1)
        preds.append(curr_pred.reshape(-1))
        labels.append(curr_label.reshape(-1))

labels = np.concatenate(labels)
preds = np.concatenate(preds)
from metrics import compute_metrics
metrics = compute_metrics(labels, preds)
print(metrics)
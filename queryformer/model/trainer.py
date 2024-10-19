# model/trainer.py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .dataset import PlanTreeDataset
import os
import time
import torch
from scipy.stats import pearsonr
import logging

from model.dataset import collator

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def print_qerror(preds_unnorm, labels_unnorm, prints=False):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(float(preds_unnorm[i]) / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    e_50, e_90 = np.median(qerror), np.percentile(qerror,90)    
    e_mean = np.mean(qerror)

    if prints:
        print("Median: {}".format(e_50))
        print("Mean: {}".format(e_mean))

    res = {
        'q_median' : e_50,
        'q_90' : e_90,
        'q_mean' : e_mean,
    }

    return res

def get_corr(ps, ls): # unnormalized
    ps = np.array(ps)
    ls = np.array(ls)
    corr, _ = pearsonr(np.log(ps), np.log(ls))
    
    return corr


def eval_workload(workload, methods):
    """
    Evaluates the model on a given workload.

    Args:
        workload (str): The name of the workload ('job-light', 'synthetic', etc.).
        methods (dict): Dictionary containing necessary methods and parameters.

    Returns:
        tuple: Evaluation scores and correlation.
    """
    get_table_sample = methods['get_sample']

    workload_file_name = './data/imdb/workloads/' + workload + '.csv'
    table_sample = get_table_sample(workload_file_name)
    plan_df = pd.read_csv('./data/imdb/{}_plan.csv'.format(workload))
    workload_csv = pd.read_csv(workload_file_name, sep='#', header=None, names=['id', 'tables_joins', 'predicate', 'cardinality'])
    ds = PlanTreeDataset(plan_df, workload_csv, \
        methods['encoding'], methods['hist_file'], methods['cost_norm'], \
        methods['cost_norm'], 'cost', table_sample)
    
    eval_score, corr = evaluate(methods['model'], ds, methods['bs'], methods['cost_norm'], methods['device'], True)
    return eval_score, corr


def evaluate(model, ds, bs, norm, device, prints=False):
    model.eval()
    label_predss = np.empty(0)

    with torch.no_grad():
        for i in range(0, len(ds), bs):
            batch, batch_labels = collator([ds[j] for j in range(i, min(i+bs, len(ds)) )])

            batch = batch.to(device)

            label_preds, _ = model(batch)
            label_preds = label_preds.squeeze()

            label_predss = np.append(label_predss, label_preds.cpu().detach().numpy())
    scores = print_qerror(norm.inverse_transform(label_predss.reshape(-1,1)), ds.labels, prints)
    corr = get_corr(norm.inverse_transform(label_predss.reshape(-1,1)), ds.labels)
    if prints:
        print('Corr: ',corr)
    return scores, corr


def train(model, train_ds, val_ds, crit, \
    label_norm, args, optimizer=None, scheduler=None):
    
    bs, device, epochs, clip_size = \
        args.bs, args.device, args.epochs, args.clip_size
    lr = args.lr

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, args.sch_decay)

    t0 = time.time()

    rng = np.random.default_rng()

    best_prev = 999999

    for epoch in range(epochs):
        losses = 0
        label_predss = np.empty(0)

        model.train()

        train_idxs = rng.permutation(len(train_ds))

        labels = np.array(train_ds.labels)[train_idxs]

        for idxs in chunks(train_idxs, bs):
            optimizer.zero_grad()

            batch, batch_labels = collator([train_ds[j] for j in idxs])


            batch_cost_label = torch.FloatTensor(batch_labels).to(device)
            batch = batch.to(device)

            label_preds, _ = model(batch)
            label_preds = label_preds.squeeze()

            loss = crit(label_preds, batch_cost_label)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_size)

            optimizer.step()
            # Fix the out of memory issue
            del batch
            del batch_labels
            torch.cuda.empty_cache()

            losses += loss.item()
            label_predss = np.append(label_predss, label_preds.detach().cpu().numpy())

        if epoch > 40:
            test_scores, corrs = evaluate(model, val_ds, bs, label_norm, device, False)

            if test_scores['q_mean'] < best_prev: ## mean mse
                best_model_path = logging_fn(args, epoch, test_scores, filename = 'log.txt', save_model = True, model = model)
                best_prev = test_scores['q_mean']

        if epoch % 20 == 0:
            print('Epoch: {}  Avg Loss: {:.4f}, Time: {:.2f}s'.format(epoch, losses/len(train_ds), time.time()-t0))
            train_scores = print_qerror(label_norm.inverse_transform(label_predss.reshape(-1,1)), labels, True)

        scheduler.step()   

    return model, best_model_path


def logging_fn(args, epoch, qscores, filename = None, save_model = False, model = None):
    arg_keys = [attr for attr in dir(args) if not attr.startswith('__')]
    arg_vals = [getattr(args, attr) for attr in arg_keys]
    
    res = dict(zip(arg_keys, arg_vals))
    model_checkpoint = str(hash(tuple(arg_vals))) + '.pt'

    res['epoch'] = epoch
    res['model'] = model_checkpoint 


    res = {**res, **qscores}

    filename = args.newpath + filename
    model_checkpoint = args.newpath + model_checkpoint
    
    if filename is not None:
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            res_df = pd.DataFrame([res])
            df = pd.concat([df, res_df], ignore_index=True)
            df.to_csv(filename, index=False)
        else:
            df = pd.DataFrame(res, index=[0])
            df.to_csv(filename, index=False)
    if save_model and model is not None:
        torch.save({
            'model': model.state_dict(),
            'args' : args
        }, model_checkpoint)
    
    return res['model']

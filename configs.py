from cross_db_benchmark.benchmark_tools.utils import load_json


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



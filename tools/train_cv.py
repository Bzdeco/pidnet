from tools.train import powerlines_config, run_training

if __name__ == '__main__':
    config = powerlines_config()
    config.cv_name = config.name
    folds = config.data.cv.get("folds_select", list(range(config.data.cv.num_folds)))
    print(f"Running folds {folds}")

    for fold in folds:
        print(f"Fold {fold}")
        fold_config = config.copy()
        fold_config.name = f"{config.name}-fold-{fold}"
        fold_config.data.cv.fold = fold
        run_training(fold_config)

# fixtime

Pipeline per stimare il tempo di fix dei bug (`fixing_time`) a partire da feature tabulari.
Per ogni scope (singolo progetto oppure tutti i progetti aggregati) e per ogni dataset
(D0 = solo feature statiche, ..., D6 = tutte le feature) esegue:

1. **Regressione** di `fixing_time` con diversi regressori.
2. **Classificazione fast/slow** per tre soglie θ (50°, 75° e 90° percentile di `fixing_time`),
   ognuna con classificatori non bilanciati e bilanciati.

Entrambe le fasi girano in 5-fold cross validation sull'80% dei dati. Poi i modelli vengono
riaddestrati sull'intero 80% e valutati sul 20% di holdout. I modelli dell'holdout vengono
salvati per l'analisi SHAP, che è a valle di questa pipeline.

## Setup

Requisiti: [uv](https://docs.astral.sh/uv/). Le versioni esatte sono fissate in `uv.lock`.

```bash
uv sync
```

I CSV non sono nel repository: vanno messi in `datasets/` con i path indicati in
`configs/default.yaml`.

## Uso

```bash
uv run fixtime run                                   # tutto: ogni scope, dataset, stage e task
uv run fixtime run --scope W3C --dataset D6          # sottoinsieme
uv run fixtime run --stage holdout --model lightgbm  # solo una fase / un modello
uv run fixtime run --limit 500                       # smoke test su un sottocampione (run separato)
uv run fixtime report                                # tabelle CSV in outputs/<run>/results/
uv run pytest                                        # test
```

Ogni fit (scope, dataset, fold, task, modello, θ, bilanciamento) è un'unità indipendente.
Un'unità già calcolata viene saltata, a meno di `--force`. Viene invece ricalcolata da sola
se cambia qualcosa che la influenza: iperparametri, file di dati, split, seed oppure
`PIPELINE_VERSION` in `src/fixtime/runner.py`. Per rifare un singolo pezzo basta cancellarne
il file `.json`.

## Metodo

- **Split**: 80/20 stratificato su 10 bin di quantile di `fixing_time` (nell'aggregato anche
  per progetto), poi 5-fold StratifiedKFold sull'80%. Lo split viene calcolato una volta per
  scope e riusato da tutti i dataset D0..D6, così i dataset si confrontano sugli stessi bug.
  Seed 42.
- **Soglia θ**: percentile di `fixing_time` calcolato solo sui dati di training (di ogni fold,
  o dell'80% per l'holdout). L'etichetta è `slow = 1` se `fixing_time >= θ`, altrimenti `fast = 0`.
- **Regressione**: i modelli imparano `log1p(fixing_time)`. Le predizioni vengono riportate in
  giorni con `expm1` e tagliate all'intervallo `[0, max(y_train)]`.
- **Bilanciamento**: `class_weight="balanced"` (XGBoost: `scale_pos_weight = n_fast / n_slow`).
- **Modelli**: Dummy, Ridge / LogisticRegression (con standardizzazione), Random Forest,
  XGBoost, LightGBM. Tutti con imputazione a mediana. Iperparametri fissi in config.
- **Metriche di regressione**: MAE, MdAE, RMSE e R² in giorni; MAE, RMSE e R² in log1p.
- **Metriche di classificazione**: MCC, PR-AUC, ROC-AUC, F1 (slow e macro), balanced accuracy,
  accuracy, precision/recall (slow), matrice di confusione.

## Output

```
outputs/<run>/
  manifests/<timestamp>.json         config, commit git, versioni dei pacchetti di ogni lancio
  splits/<scope>.json                id di holdout e id di test di ogni fold
  cv/<scope>/<dataset>/regression/<model>/fold<k>.{json,parquet}
  cv/<scope>/<dataset>/classification/p<θ>/<balanced|unbalanced>/<model>/fold<k>.{json,parquet}
  holdout/<...stesso schema...>/{metrics.json,predictions.parquet,model.joblib}
  results/{cv_*_folds,cv_*_summary,holdout_*}.csv
```

I `.json` contengono metriche, θ, dimensioni e iperparametri. I `.parquet` contengono le
predizioni per id. Ogni `model.joblib` è una sklearn Pipeline completa (imputer, scaler,
modello): per SHAP si caricano il modello e le righe di holdout da `splits/<scope>.json`,
con le colonne in `feature_names` del `metrics.json`.

## Aggiungere un dataset

Basta aggiungere una voce sotto `datasets:` in `configs/default.yaml`:

```yaml
datasets:
  D0:
    project_path: "datasets/.../D0_{project}.csv"
    aggregated_path: "datasets/.../D0_aggregato.csv"
```

Tutti i dataset di uno scope devono contenere gli stessi id. Se non è così, la pipeline si
ferma con un errore.

## Dati

V1.0 (`datasets/V1.0_no time`): 7 progetti (Eclipse, Novell, Mozilla, OpenOffice, KDE,
OpenXchange, W3C) più l'aggregato, 45 feature. Per ora viene usato come D6. Il dataset V2.0,
con le versioni temporali D0..D5, è in preparazione.

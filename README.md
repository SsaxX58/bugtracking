# fixtime

Pipeline per stimare il tempo di fix dei bug (`fixing_time`) a partire da feature tabulari.
Per ogni scope (singolo progetto oppure tutti i progetti aggregati) e per ogni dataset
(uno snapshot del bug nel tempo: `T0` = solo feature statiche, `T1C`, `T24`, `T72`, `T7D`,
`FULL` = tutta la storia) esegue:

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
uv run fixtime run --scope W3C --dataset T24         # sottoinsieme
uv run fixtime run --config configs/v1.yaml          # vecchio dataset V1 (run v1)
uv run fixtime run --stage holdout --model lightgbm  # solo una fase / un modello
uv run fixtime run --limit 500                       # smoke test su un sottocampione (run separato)
uv run fixtime report                                # tabelle CSV in outputs/<run>/results/
uv run pytest                                        # test
uv run mypy src                                      # controllo statico dei tipi (strict)
```

Ogni fit (scope, dataset, fold, task, modello, θ, bilanciamento) è un'unità indipendente.
Un'unità già calcolata viene saltata, a meno di `--force`. Viene invece ricalcolata da sola
se cambia qualcosa che la influenza: iperparametri, file di dati, split, seed oppure
`PIPELINE_VERSION` in `src/fixtime/runner.py`. Per rifare un singolo pezzo basta cancellarne
il file `.json`.

## Tipi e validazione

Tutte le strutture di dati sono modelli [pydantic](https://docs.pydantic.dev/):

- `config.py`: `Config` e le sue sezioni, con gli iperparametri tipizzati per ogni modello.
  Il YAML viene validato al caricamento: chiavi sconosciute, iperparametri scritti male o
  valori fuori scala sono errori e il run non parte.
- `data.py`: `Dataset` (feature, target e id in memoria).
- `splits.py`: `Split` (`splits/<scope>.json`).
- `metrics.py`: `RegressionMetrics`, `ClassificationMetrics`.
- `records.py`: `Unit`, `RegressionResult`, `ClassificationResult` (i `.json` di ogni unità)
  e `Manifest`. `UNIT_RESULT.validate_json(...)` rilegge un risultato con il tipo giusto.

Il pacchetto passa `mypy --strict` (configurazione in `pyproject.toml`).

## Metodo

- **Split**: 80/20 stratificato su 10 bin di quantile di `fixing_time` (nell'aggregato anche
  per progetto), poi 5-fold StratifiedKFold sull'80%. Lo split viene calcolato una volta per
  scope sul dataset `split_reference` (`T0`, che contiene tutti i bug) e riusato dagli altri
  snapshot, che contengono solo i bug ancora aperti in quel momento: ogni bug sta nello stesso
  fold in tutti i dataset. Seed 42.
- **Soglia θ**: percentile di `fixing_time` calcolato solo sui dati di training (di ogni fold,
  o dell'80% per l'holdout) del dataset in uso. L'etichetta è `slow = 1` se `fixing_time >= θ`, altrimenti `fast = 0`.
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
  T14D:
    project_path: "datasets/.../{project}/{project}_T14D.csv"
    aggregated_path: "datasets/.../T14D_aggregato.csv"   # facoltativo
```

Senza `aggregated_path` l'aggregato viene costruito unendo i file dei progetti, con id
`Progetto:bug_id`. Ogni bug di un dataset deve esistere nel dataset `split_reference`; se non
è così, la pipeline si ferma con un errore. Le colonne che non devono diventare feature
(id, varianti del target, testo) vanno elencate in `drop_columns`.

## Dati

- **V2** (`datasets/V2.0_time_Gabriele/Progetti separati/`, config `configs/default.yaml`,
  run `v2`): 7 progetti × 6 snapshot. Feature: T0 28, T1C 32, T24/T72/T7D 38, FULL 45.
  L'aggregato non è fornito e viene costruito dal codice.
- **V1** (`datasets/V1.0_no time`, config `configs/v1.yaml`, run `v1`): un solo dataset con
  tutte le feature (equivalente a FULL) più l'aggregato.

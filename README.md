# fixtime

Pipeline per stimare il tempo di fix dei bug (`fixing_time`, in giorni) a partire da feature
tabulari estratte dai bug tracker di 7 progetti open source.

Per ogni **scope** (un singolo progetto oppure tutti i progetti aggregati) e per ogni
**dataset** (uno snapshot del bug in un certo momento della sua vita) la pipeline esegue:

1. **Regressione**: stima del tempo di fix in giorni, con diversi regressori.
2. **Classificazione fast/slow**: per tre soglie θ (50°, 75° e 90° percentile del tempo di
   fix), classificatori non bilanciati e bilanciati.

Entrambe le fasi girano in 5-fold cross validation sull'80% dei dati. Poi i modelli vengono
riaddestrati sull'intero 80% e valutati sul 20% di holdout, mai visto prima. I modelli
dell'holdout vengono salvati per l'analisi SHAP, che è a valle di questa pipeline.

## Indice

1. [Setup](#1-setup)
2. [Uso rapido](#2-uso-rapido)
3. [I dati](#3-i-dati)
4. [Come funziona la pipeline](#4-come-funziona-la-pipeline)
5. [Analisi aggregata](#5-analisi-aggregata)
6. [Output](#6-output)
7. [Leggere i risultati](#7-leggere-i-risultati)
8. [Interfaccia per SHAP](#8-interfaccia-per-shap)
9. [Configurazione](#9-configurazione)
10. [Struttura del codice](#10-struttura-del-codice)
11. [Riproducibilità](#11-riproducibilità)
12. [Cosa manca e limiti noti](#12-cosa-manca-e-limiti-noti)

---

## 1. Setup

Requisiti: [uv](https://docs.astral.sh/uv/). Python 3.12 e le versioni esatte di tutti i
pacchetti sono fissate in `uv.lock`.

```bash
uv sync
```

I CSV non sono nel repository (`datasets/` e `outputs/` sono in `.gitignore`): vanno messi in
`datasets/` con i percorsi indicati nella config (vedi [I dati](#3-i-dati)).

## 2. Uso rapido

```bash
uv run fixtime run                                   # tutto: ogni scope, dataset, fase e task
uv run fixtime run --scope W3C --dataset T24         # un sottoinsieme
uv run fixtime run --scope aggregated --dataset T0   # solo l'aggregato
uv run fixtime run --stage holdout --model lightgbm  # solo una fase / un modello
uv run fixtime run --task classification             # solo la classificazione
uv run fixtime run --force --scope KDE               # rifà anche ciò che è in cache
uv run fixtime run --limit 500                       # prova veloce su un sottocampione (run separato)
uv run fixtime run --config configs/v1.yaml          # vecchio dataset V1 (run v1)
uv run fixtime report                                # tabelle CSV in outputs/<run>/results/
uv run pytest                                        # test
uv run mypy src                                      # controllo statico dei tipi (strict)
```

Altre opzioni di `run`: `--n-jobs` (sovrascrive `n_jobs` della config) e `--run-name`
(sovrascrive `run_name`, utile per un esperimento separato senza toccare i risultati esistenti).
`--scope`, `--dataset`, `--stage`, `--task` e `--model` accettano più valori.

Una tipica sessione: `fixtime run` (le unità già fatte vengono saltate), poi `fixtime report`,
poi lettura dei CSV in `outputs/<run>/results/`.

---

## 3. I dati

### 3.1 Versioni

| Versione             | Cartella                                                               | Config                 | Run          | Contenuto                                                                                          |
| -------------------- | ---------------------------------------------------------------------- | ---------------------- | ------------ | -------------------------------------------------------------------------------------------------- |
| **V2** (predefinita) | `datasets/V2.0_time_Gabriele/Progetti separati/<P>/<P>_<snapshot>.csv` | `configs/default.yaml` | `outputs/v2` | 7 progetti × 6 snapshot temporali; nessun aggregato fornito, lo costruisce il codice               |
| **V1**               | `datasets/V1.0_no time/`                                               | `configs/v1.yaml`      | `outputs/v1` | un solo dataset per progetto con tutte le feature (equivalente a FULL di V2) più un file aggregato |

Progetti: Eclipse, Novell, Mozilla, OpenOffice, KDE, OpenXchange, W3C.

### 3.2 Gli snapshot di V2

Ogni snapshot descrive i bug con le sole informazioni disponibili in un certo momento.

| Dataset | Momento                | Feature                                                                                           | Quali bug contiene                         |
| ------- | ---------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `T0`    | apertura del bug       | 28 (solo statiche: testo della descrizione, gravità e priorità iniziali, esperienza del reporter) | tutti                                      |
| `T1C`   | primo commento         | 32 (T0 + primo commento e il suo tempo)                                                           | solo i bug ancora aperti al primo commento |
| `T24`   | 24 ore dopo l'apertura | 38 (T0 + 10 feature dinamiche sui commenti)                                                       | solo i bug ancora aperti dopo 24 ore       |
| `T72`   | 72 ore                 | 38                                                                                                | solo quelli ancora aperti dopo 72 ore      |
| `T7D`   | 7 giorni               | 38                                                                                                | solo quelli ancora aperti dopo 7 giorni    |
| `FULL`  | tutta la vita del bug  | 45                                                                                                | tutti                                      |

Conseguenze importanti:

- T1C, T24, T72 e T7D sono **sottoinsiemi** di T0. Per questo lo split si calcola su T0 e
  viene riusato da tutti (vedi [Split](#42-split)).
- Il target `fixing_time` è il **tempo totale** dall'apertura alla risoluzione, identico in
  tutti gli snapshot dello stesso bug (non il tempo residuo dal momento dello snapshot).
- FULL usa informazioni che all'apertura non esistono ancora (numero di commenti, silenzi,
  riaperture...). Serve come riferimento, non come scenario realistico.

### 3.3 Colonne

- **Target**: `fixing_time` (giorni, con decimali).
- **Id**: `bug_id` nei progetti; nell'aggregato `row_uid` = `Progetto:bug_id` (vedi
  [Analisi aggregata](#5-analisi-aggregata)).
- **Feature**: tutte le colonne numeriche che non sono in `drop_columns`.
- **Escluse** (`drop_columns` in config):
  - id e metadati: `bug_id`, `project`, `row_uid`, `creation_ts`;
  - la data di risoluzione e ogni variante del target (`fixing_time_hours/days/log`,
    `residual_*`), che darebbero la risposta al modello;
  - stringhe e testo libero: `product`, `component`, `version`, `rep_platform`, `op_sys`,
    `first_severity`, `first_priority`, `*_status`, `reporter`, `assigned_to`, `short_desc`.
    Le versioni numeriche (`first_severity_score`, ...) restano. Attenzione se si pensa di
    riaggiungerle: `product`, `component`, `version`, `assigned_to` e `short_desc` contengono
    il valore **finale** del bug, non quello all'apertura (verificato sulla cronologia
    pubblica di Bugzilla per Mozilla), quindi negli snapshot porterebbero informazioni dal futuro;
  - la stessa grandezza in un'altra unità (`time_to_first_comment_hours/_log`,
    `max_silence_dyn_hours`, `avg_comment_gap_dyn_hours`): si tiene solo la versione in giorni.
- **Valori mancanti**: imputati con la mediana del train dentro ogni modello.

### 3.4 Aggiungere un dataset

Basta una voce sotto `datasets:` nella config:

```yaml
datasets:
  T14D:
    project_path: "datasets/.../{project}/{project}_T14D.csv"
    aggregated_path: "datasets/.../T14D_aggregato.csv" # facoltativo
```

Ogni bug del nuovo dataset deve esistere nel dataset `split_reference` (T0), altrimenti la
pipeline si ferma con un errore. Eventuali nuove colonne da non usare come feature vanno
aggiunte a `drop_columns`. Le unità già calcolate degli altri dataset restano in cache.

---

## 4. Come funziona la pipeline

### 4.1 Flusso

```
config YAML ──► per ogni scope:
                  carica il dataset di riferimento (T0) ──► split: holdout 20% + 5 fold sull'80%
                                                             (salvato in splits/<scope>.json)
                  per ogni dataset (T0, T1C, ..., FULL):
                    carica X, y, id ──► tiene di ogni fold solo i bug presenti nel dataset
                    per ogni unità di lavoro (fold × task × modello × θ × bilanciamento):
                      se già in cache con la stessa impronta ──► salta
                      altrimenti: addestra sul train, valuta sul test, salva JSON + parquet
                      (holdout: anche il modello .joblib)
fixtime report ──► raccoglie tutti i JSON in tabelle CSV
```

### 4.2 Split

- **Uno split per scope, non per dataset.** Holdout e fold si calcolano una volta sul dataset
  `split_reference` (T0, che contiene tutti i bug) e valgono per tutti gli snapshot: ogni bug
  sta nello stesso fold in T0, T24, FULL, ecc. Così i dataset si confrontano sugli stessi bug.
  Negli snapshot più piccoli ogni fold contiene solo i suoi bug ancora presenti.
- **Holdout**: 20% stratificato; **CV**: 5 fold `StratifiedKFold` sull'80% restante.
- **Stratificazione**: 10 fasce di quantile di `fixing_time`, così holdout e fold hanno la
  stessa distribuzione dei tempi (nell'aggregato: progetto × fascia).
- **Seed** 42, salvato in config. Lo split viene scritto in `outputs/<run>/splits/<scope>.json`
  e riletto alle esecuzioni successive.

### 4.3 Regressione

- I modelli imparano `log1p(fixing_time)` (il tempo è molto asimmetrico: molti bug chiusi in
  poche ore, alcuni dopo anni), tramite `TransformedTargetRegressor`.
- Le predizioni tornano in giorni con `expm1` e vengono tagliate all'intervallo
  `[0, max(y_train)]` (il massimo è salvato come `pred_clip_max`): senza taglio un modello
  lineare può estrapolare tempi assurdi.

### 4.4 Classificazione fast/slow

- **Soglia θ**: 50°, 75° o 90° percentile di `fixing_time`, calcolato **solo sui dati di
  training** (di ogni fold, o dell'80% per l'holdout) del dataset in uso. Il test non
  influisce mai sulla soglia. Siccome la popolazione cambia tra snapshot, θ cambia anche tra
  snapshot (in T7D restano solo bug più lunghi).
- **Etichetta**: `slow = 1` se `fixing_time >= θ`, altrimenti `fast = 0`, sul tempo reale.
  È una classificazione diretta, non derivata dalla previsione del regressore.
- **Non bilanciato**: il modello vede le classi con le loro frequenze (a p75, 25% slow).
- **Bilanciato**: pesi di classe, nessun dato aggiunto o tolto. `class_weight="balanced"`
  dà a ogni classe il peso `n_totale / (2 × n_classe)` (a p75 un bug slow pesa come tre fast);
  XGBoost usa l'equivalente `scale_pos_weight = n_fast / n_slow`. I conteggi vengono dal train.

### 4.5 Modelli

Ogni modello è una `sklearn.Pipeline`: imputazione a mediana, standardizzazione (solo modelli
lineari), modello.

| Nome in config     | Regressione       | Classificazione                      |
| ------------------ | ----------------- | ------------------------------------ |
| `dummy`            | mediana del train | classe a priori (riferimento: MCC 0) |
| `ridge` / `logreg` | Ridge             | Logistic Regression                  |
| `random_forest`    | Random Forest     | Random Forest                        |
| `xgboost`          | XGBoost           | XGBoost                              |
| `lightgbm`         | LightGBM          | LightGBM                             |

Iperparametri fissi in config (nessuna ricerca automatica). Ogni modello usa un solo thread:
il parallelismo è tra unità (vedi [4.8](#48-parallelismo)).

### 4.6 Metriche

- **Regressione**: MAE, MdAE (errore mediano), RMSE, R² in giorni; MAE, RMSE, R² sul
  logaritmo (`mae_log`, ...), che misurano l'errore "in proporzione" e non sono dominati dai
  pochi bug lunghissimi.
- **Classificazione**: MCC, PR-AUC, ROC-AUC, F1 (slow e macro), balanced accuracy, accuracy,
  precision e recall della classe slow, matrice di confusione (`tn`, `fp`, `fn`, `tp`), quota
  di slow nel test. PR-AUC e ROC-AUC sono `null` se nel test c'è una sola classe.

Il confronto con il modello `dummy` dello stesso scope e dataset dice quanto un modello impara
davvero.

### 4.7 Unità di lavoro e cache

Ogni fit è un'**unità** indipendente, identificata da:

```
(scope, dataset, fase {cv, holdout}, task {regression, classification}, modello, fold,
 θ e bilanciamento solo per la classificazione)
```

- Ogni unità scrive i suoi file in un percorso fisso (vedi [Output](#6-output)).
- Nel JSON c'è un'**impronta** (sha256) di tutto ciò che influenza il risultato: unità,
  iperparametri, seed, target, hash del file di dati, elenco delle feature, impronta dello
  split, impostazioni dello split e `PIPELINE_VERSION` (in `src/fixtime/runner.py`).
- Al lancio, un'unità con JSON presente e impronta uguale viene **saltata**. Se cambia
  qualcosa (un iperparametro, un CSV, l'elenco delle feature), vengono rifatte solo le unità
  coinvolte.
- Per rifare un pezzo: cancellarne il `.json`, oppure usare `--force` con i filtri.
- Quando una modifica al codice cambia i risultati, si incrementa `PIPELINE_VERSION`:
  tutte le unità vengono rifatte.
- Il JSON viene scritto per ultimo e in modo atomico: è il segnale di unità completata, quindi
  un'interruzione non lascia mai un'unità segnata come fatta a metà.
- Un'unità che fallisce non ferma il run: l'errore viene stampato e le altre proseguono.

### 4.8 Parallelismo

Le unità girano in parallelo con joblib (`n_jobs: -1` = tutti i core); ogni modello usa un
solo thread. Il run completo V2 (8 400 unità di CV + 1 680 di holdout) richiede circa 20
minuti su 16 thread. La GPU non è usata: il tempo è dominato da Random Forest, che gira solo
su CPU.

---

## 5. Analisi aggregata

Lo scope `aggregated` mette insieme i bug di tutti i progetti in un solo dataset, per
addestrare e valutare un modello unico.

### 5.1 Come viene costruito

- **V2**: nessun file aggregato è fornito. Per ogni snapshot il codice (`_read_table` in
  `src/fixtime/data.py`) legge i 7 CSV dei progetti e li concatena in memoria. Non viene
  salvato su disco: si ricostruisce identico a ogni esecuzione (l'impronta combina gli hash
  dei 7 file).
- **V1**: c'è un file aggregato vero, indicato da `aggregated_path` in `configs/v1.yaml`.
- Se in futuro arriverà un aggregato V2 ufficiale, basterà aggiungere `aggregated_path` alla
  voce del dataset (meglio con un nuovo `run_name`).

### 5.2 Id e progetto

- I `bug_id` si ripetono tra progetti, quindi l'id dell'aggregato è
  `row_uid = Progetto:bug_id` (per esempio `Eclipse:1000041`).
- Viene aggiunta la colonna `project`, che **non** è una feature (è in `drop_columns`):
  il modello non sa da quale progetto viene un bug.
- Lo split è stratificato per **progetto × fascia di tempo**: ogni progetto ha la stessa quota
  in holdout e in ogni fold.

### 5.3 Soglie e peso dei progetti

- θ è **unico** per tutto l'aggregato, calcolato sul pool di training. In ogni singolo
  progetto, quindi, la quota di slow può essere molto diversa dal 25% (a p75).
- I progetti **non sono bilanciati**: ognuno pesa per il numero dei suoi bug. Eclipse,
  Novell e Mozilla insieme sono circa il 95% dell'aggregato; KDE, OpenXchange e W3C meno del
  2% ciascuno. Modello, soglie e metriche dell'aggregato riflettono quindi soprattutto i tre
  progetti grandi.

### 5.4 Lanciarla e leggerla

```bash
uv run fixtime run --scope aggregated                # tutti i dataset
uv run fixtime run --scope aggregated --dataset T0   # un solo snapshot
uv run fixtime report
```

Nei CSV di `results/` l'aggregato compare con `scope = aggregated`. Per vedere come va il
modello aggregato su ciascun progetto, basta raggruppare le sue predizioni per il prefisso
dell'id:

```python
import pandas as pd
from sklearn.metrics import matthews_corrcoef

unit = "outputs/v2/holdout/aggregated/T0/classification/p75/balanced/lightgbm"
pred = pd.read_parquet(f"{unit}/predictions.parquet")
pred["project"] = pred["id"].str.split(":").str[0]
print(pred.groupby("project").apply(lambda g: matthews_corrcoef(g.label_true, g.label_pred)))
```

(Lo stesso vale per i `fold<k>.parquet` della CV.) Il confronto giusto per un progetto è tra
il modello aggregato filtrato su quel progetto e il modello addestrato solo su quel progetto,
ricordando che le soglie sono diverse.

---

## 6. Output

```
outputs/<run>/
  manifests/<timestamp>.json         config risolta, commit git, versioni dei pacchetti, argomenti
  splits/<scope>.json                id di holdout e id di test di ogni fold
  cv/<scope>/<dataset>/regression/<model>/fold<k>.{json,parquet}
  cv/<scope>/<dataset>/classification/p<θ>/<balanced|unbalanced>/<model>/fold<k>.{json,parquet}
  holdout/<scope>/<dataset>/regression/<model>/{metrics.json,predictions.parquet,model.joblib}
  holdout/<scope>/<dataset>/classification/p<θ>/<balanced|unbalanced>/<model>/{...stessi file}
  results/cv_{regression,classification}_folds.csv
  results/cv_{regression,classification}_summary.csv
  results/holdout_{regression,classification}.csv
```

- **`.json`** di un'unità: scope, dataset, fold, modello, iperparametri (`params`), metriche,
  `n_train`, `n_test`, `fit_seconds`, `feature_names` (nell'ordine usato dal modello),
  impronta (`hash`); per la regressione `pred_clip_max`, per la classificazione `theta` (in
  giorni) e `train_slow_rate`.
- **`.parquet`**: una riga per bug di test. Regressione: `id`, `y_true`, `y_pred` (giorni).
  Classificazione: `id`, `y_true`, `label_true`, `proba_slow`, `label_pred`.
- **`model.joblib`** (solo holdout): il modello addestrato sull'80%, compresso.
- **`splits/<scope>.json`**: `holdout` (lista di id), `folds` (lista di 5 liste di id di
  test), `ids_hash`.

Nella CV si salvano predizioni e metriche ma non i modelli (spazio): ogni metrica si può
ricalcolare dalle predizioni senza riaddestrare.

## 7. Leggere i risultati

`uv run fixtime report` legge tutti i JSON del run e scrive:

| File               | Contenuto                                                                  |
| ------------------ | -------------------------------------------------------------------------- |
| `cv_*_folds.csv`   | una riga per unità di CV (ogni fold) con tutte le metriche                 |
| `cv_*_summary.csv` | media e deviazione standard sui 5 fold (`<metrica>_mean`, `<metrica>_std`) |
| `holdout_*.csv`    | una riga per modello sull'holdout                                          |

Chiavi delle righe: `scope`, `dataset`, `model`, più `theta_percentile` e `balanced` per la
classificazione.

Indicazioni di lettura:

- Metriche principali: **MCC** e **PR-AUC** per la classificazione (il caso vale 0 e la
  quota di slow), **MAE_log** e il suo confronto con `dummy` per la regressione.
- Guardare sempre la deviazione standard tra i fold: nei progetti piccoli (OpenXchange, W3C,
  poche centinaia di bug) è grande e le differenze tra modelli spesso non sono significative.
- La CV serve a confrontare e scegliere; l'holdout a confermare su dati mai usati.
- Il punteggio `proba_slow` dei modelli bilanciati non è una probabilità calibrata (i pesi
  di classe gonfiano la classe slow): va usato per ordinare i bug o per la decisione
  sopra/sotto soglia.

---

## 8. Interfaccia per SHAP

SHAP è fuori dallo scopo di questa pipeline, che fornisce tutto il necessario per farlo.

### 8.1 Cosa serve

- `outputs/<run>/holdout/.../model.joblib`: i modelli addestrati sull'80%;
- `outputs/<run>/holdout/.../metrics.json`: `feature_names` (colonne e loro ordine);
- `outputs/<run>/splits/<scope>.json`: gli id dell'holdout;
- i CSV dei dati e la config, per ricostruire le righe.

Per ogni scope e dataset ci sono 5 regressori e 30 classificatori (5 modelli × 3 soglie ×
bilanciato/non). Per SHAP conviene scegliere un modello ad albero (`lightgbm`, `xgboost`,
`random_forest`) e spiegare quello migliore per lo scope e la soglia di interesse.

### 8.2 Ricostruire le righe di holdout e caricare il modello

```python
import json

import joblib
import pandas as pd

from fixtime.config import load_config
from fixtime.data import load_dataset
from fixtime.splits import Split

cfg = load_config("configs/default.yaml")
scope, dataset = "Eclipse", "T0"                 # oppure "aggregated", "T24", ...
unit = cfg.run_dir / "holdout" / scope / dataset / "classification" / "p75" / "balanced" / "lightgbm"

split = Split.model_validate_json((cfg.run_dir / "splits" / f"{scope}.json").read_text())
ds = load_dataset(cfg, scope, dataset)
ids = [i for i in split.holdout if i in ds.X.index]   # negli snapshot non tutti i bug ci sono

model = joblib.load(unit / "model.joblib")            # Pipeline: impute (+ scale) + model
features = json.loads((unit / "metrics.json").read_text())["feature_names"]
X = ds.X.loc[ids, features]

proba = model.predict_proba(X)[:, 1]                  # identico a proba_slow in predictions.parquet
```

### 8.3 Calcolare SHAP

`shap` non è una dipendenza del progetto: si può usare senza installarlo stabilmente con
`uv run --with shap python script.py`. SHAP va applicato all'ultimo passo della pipeline,
sulle feature già trasformate:

```python
import shap

X_t = pd.DataFrame(model[:-1].transform(X), columns=features, index=X.index)  # imputazione (+ scala)
explainer = shap.TreeExplainer(model[-1])
shap_values = explainer.shap_values(X_t)
```

Per un **regressore** il file contiene un `TransformedTargetRegressor`: la pipeline interna
addestrata è `reg.regressor_` e lavora sul logaritmo.

```python
reg = joblib.load(cfg.run_dir / "holdout" / scope / dataset / "regression" / "lightgbm" / "model.joblib")
inner = reg.regressor_                                # Pipeline: impute + model, output log1p(giorni)
X_t = pd.DataFrame(inner[:-1].transform(X), columns=features, index=X.index)
shap_values = shap.TreeExplainer(inner[-1]).shap_values(X_t)
```

### 8.4 Trappole

- **Regressori sul logaritmo**: i valori SHAP sono in unità di `log1p(giorni)`, non in giorni.
  `reg.predict(X)` restituisce giorni (`expm1`) ma **senza** il taglio a `[0, pred_clip_max]`
  applicato nelle predizioni salvate.
- **Classificatori bilanciati**: i pesi di classe spostano il valore base e i punteggi verso
  la classe slow; spiegano l'ordinamento dei bug, non probabilità reali. Con `TreeExplainer`
  l'output di LightGBM e XGBoost è in log-odds.
- **Modelli lineari** (`ridge`, `logreg`): usano `shap.LinearExplainer` e la pipeline
  include la standardizzazione, quindi i valori sono riferiti alle feature standardizzate.
- **Snapshot**: l'holdout di T24, T72, T7D, T1C contiene solo i bug ancora aperti in quel
  momento (sottoinsieme dell'holdout di T0).
- **Aggregato**: id `Progetto:bug_id`; il progetto non è una feature, quindi SHAP non lo
  vedrà mai come causa.
- **Random Forest** con 200 alberi è lento con `TreeExplainer` sull'holdout dell'aggregato
  (circa 20 000 bug): conviene un campione.

---

## 9. Configurazione

Le config sono in `configs/` (`default.yaml` = V2, `v1.yaml` = V1) e vengono validate al
caricamento: chiavi sconosciute, iperparametri scritti male o valori fuori scala sono errori
e il run non parte.

| Chiave                                                        | Significato                                                |
| ------------------------------------------------------------- | ---------------------------------------------------------- |
| `base_dir`, `output_dir`, `run_name`                          | percorsi; i risultati vanno in `<output_dir>/<run_name>`   |
| `seed`                                                        | seed di split e modelli (42)                               |
| `target`                                                      | colonna del target (`fixing_time`)                         |
| `projects`, `aggregated`                                      | progetti e se includere lo scope aggregato                 |
| `id_column.project`, `id_column.aggregated`, `project_column` | colonne di id e di progetto                                |
| `drop_columns`                                                | colonne da non usare come feature                          |
| `datasets.<nome>.project_path`                                | percorso per progetto, con `{project}` come segnaposto     |
| `datasets.<nome>.aggregated_path`                             | facoltativo: file aggregato già pronto                     |
| `split_reference`                                             | dataset su cui si calcola lo split (predefinito: il primo) |
| `split.holdout_size`, `split.n_folds`, `split.n_bins`         | 0.2, 5, 10                                                 |
| `thresholds`                                                  | percentili delle soglie θ (`[50, 75, 90]`)                 |
| `n_jobs`                                                      | unità in parallelo (`-1` = tutti i core)                   |
| `models.regression.<nome>`, `models.classification.<nome>`    | modelli attivi e loro iperparametri                        |

Un modello è attivo se la sua chiave è presente sotto `models`. Gli iperparametri non scritti
restano al valore predefinito della libreria. Cambiare un iperparametro invalida solo le unità
di quel modello.

## 10. Struttura del codice

```
src/fixtime/
  config.py      Config e iperparametri tipizzati (pydantic), caricamento e validazione YAML
  data.py        Dataset: lettura dei CSV, aggregato per concatenazione, X / y / id
  splits.py      Split: holdout + fold stratificati, salvataggio e rilettura
  thresholds.py  θ dal train ed etichette fast/slow
  models.py      costruzione di regressori e classificatori (pipeline, pesi di classe, log1p)
  metrics.py     RegressionMetrics, ClassificationMetrics
  records.py     Unit, risultati per unità (JSON), Manifest
  runner.py      enumerazione delle unità, impronta e cache, esecuzione parallela
  report.py      raccolta dei JSON in CSV, medie e deviazioni standard sui fold
  cli.py         comandi `fixtime run` e `fixtime report`
tests/           test pytest (dati sintetici + controlli sui dati reali se presenti)
configs/         default.yaml (V2), v1.yaml (V1)
```

Tutte le strutture di dati sono modelli [pydantic](https://docs.pydantic.dev/):
`UNIT_RESULT.validate_json(...)` (in `records.py`) rilegge il JSON di un'unità con il tipo
giusto (regressione o classificazione). Il pacchetto passa `mypy --strict` (configurazione in
`pyproject.toml`).

## 11. Riproducibilità

- Versioni dei pacchetti fissate in `uv.lock`; Python 3.12.
- Seed fisso per split e modelli; split salvato su file e riusato.
- Ogni lancio scrive un manifest con config risolta, commit git, versioni e argomenti.
- Ogni risultato porta l'impronta di tutto ciò che lo determina (vedi [4.7](#47-unità-di-lavoro-e-cache)).
- Per un esperimento che cambia lo split o la config senza toccare i risultati esistenti:
  usare un nuovo `run_name` (`--run-name`).

## 12. Cosa manca e limiti noti

**Da aggiungere**

- **SHAP**: analisi di spiegabilità sui modelli dell'holdout, a valle. L'interfaccia è
  descritta in [Interfaccia per SHAP](#8-interfaccia-per-shap).

**Problemi noti del codice**

1. Cambiare seed o impostazioni dello split mantenendo lo stesso `run_name` non rifà lo split:
   le unità vengono ricalcolate, ma sul vecchio `splits/<scope>.json`. Per ora: chi cambia lo
   split usa un nuovo `run_name`.
2. `fixtime report` legge tutti i JSON della cartella del run, anche quelli di modelli o
   soglie tolti dalla config o non ancora ricalcolati dopo una modifica.
3. Un id modificato o cancellato a mano in `splits/<scope>.json` verrebbe ignorato senza
   errore (quel bug sparirebbe dai fold).

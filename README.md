# Traffic Disruption Prediction

Predicts how long a traffic disruption (breakdown, accident, road work, water-logging, etc.) will take to resolve, using a gradient-boosted classifier trained on historical Bengaluru traffic-incident data.

Given an incident's start time, location, and optional context (zone, junction, cause, description, etc.), the model classifies the **expected resolution time** into one of four buckets:

| Class | Label | Duration |
|---|---|---|
| 0 | Quick | < 30 mins |
| 1 | Minor | 30–90 mins |
| 2 | Major | 90–240 mins |
| 3 | Severe | > 240 mins |

---

## Project Structure

| File | Purpose |
|---|---|
| `FinalModel.ipynb` | End-to-end training notebook: data loading, feature engineering, preprocessing pipeline, hyperparameter search, evaluation, and model export (`traffic_disruption_model.pkl`). |
| `extract_medians.py` | One-off helper script that recomputes the real zone/junction historical medians from your training CSV and prints ready-to-paste Python dicts for `app.py`. |
| `app.py` | FastAPI inference service. Accepts ~20 raw incident fields, derives the full 40+ feature model input internally, and returns a predicted disruption category with probabilities. |

---

## Installation

```bash
pip install pandas numpy scikit-learn fastapi uvicorn pydantic huggingface_hub
```

> Python 3.9+ recommended (uses `dict[str, float]` type hints and `from __future__ import annotations`).

---

## 1. Training

Training is done in `FinalModel.ipynb`, which expects a `data.csv` with at least the following columns: `start_datetime`, `closed_datetime`, `created_date`, `latitude`, `longitude`, `endlatitude`, `endlongitude`, `description`, `comment`, `zone`, `junction`, `event_type`, `event_cause`, `priority`, `veh_type`, `corridor`, `direction`, `police_station`, `reason_breakdown`, `cargo_material`, `status`, `requires_road_closure`.

### Pipeline summary

1. **Load & label** — parses timestamps, computes `resolution_time_mins = closed_datetime − start_datetime`, and buckets it into the 4 classes above (`pd.cut`, bins `[0, 30, 90, 240, ∞]`).
2. **Feature engineering** (~40 features):
   - **Temporal**: hour/day-of-week/month, cyclical sin/cos encodings, rush-hour/night/lunch-hour/weekend flags, minutes-from-midnight.
   - **Spatial**: distance from Bengaluru city centre, end-coordinate deltas, incident spread.
   - **Text**: description/comment length & word counts, keyword flags (`heavy`, `blocked`, `accident`, `fire`, `tree`, `infra`, `bmtc`) and a combined `keyword_severity_score`.
   - **Operational**: triage lag (creation → start), road-closure flag, **zone/junction historical median resolution time** (target-encoding proxy), and an ordinal `cause_severity` mapping.
3. **Preprocessing**: `RobustScaler` + median imputation for numerical columns (robust to traffic-data outliers), `OneHotEncoder` (rare categories grouped under 1% frequency) for categorical columns, combined via `ColumnTransformer`.
4. **Model**: `HistGradientBoostingClassifier` with early stopping, tuned via 5-fold `GridSearchCV` over learning rate, max iterations, max depth, min samples per leaf, L2 regularization, max leaf nodes, and max bins (864 candidate combinations / 4320 fits).
5. **Evaluation**: prints best CV/test accuracy, a full classification report, and a confusion matrix.
6. **Export**: saves the best pipeline (preprocessing + classifier) to `traffic_disruption_model.pkl` via `pickle`.

### Latest run results

```
Best Parameters: {'classifier__l2_regularization': 5.0, 'classifier__learning_rate': 0.1,
                   'classifier__max_bins': 255, ...}
Best CV Accuracy : 55.40%
Test Set Accuracy: 56.07%
```

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Quick (<30 mins) | 0.47 | 0.38 | 0.42 | 167 |
| Minor (30–90 mins) | 0.51 | 0.59 | 0.55 | 210 |
| Major (90–240 mins) | 0.43 | 0.07 | 0.12 | 84 |
| Severe (>240 mins) | 0.68 | 0.95 | 0.79 | 165 |

Overall accuracy: **56%** (626 test samples). The model performs well on Quick/Minor/Severe cases but struggles to recall the Major (90–240 min) class — it's frequently confused with Minor and Severe, which is the main area for future improvement (e.g. more granular operational features, class rebalancing, or merging the Major bucket).

### Updating the historical medians

After retraining on new data, run:

```bash
python extract_medians.py --data data.csv
```

This recomputes `GLOBAL_MEDIAN`, `ZONE_MEDIANS`, and `JUNCTION_MEDIANS` from the CSV and prints them as ready-to-paste Python dict literals — copy the output directly into the constants section of `app.py` so the API's feature engineering stays in sync with the latest training data.

---

## 2. Serving the model (API)

`app.py` downloads `traffic_disruption_model.pkl` from the Hugging Face Hub (`SupratimKukri/traffic-disruption-model`) at startup and serves predictions over HTTP.

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` , `/health` | Health check — confirms the model loaded successfully. |
| `POST` | `/predict` | Predict disruption category for a single incident. |
| `POST` | `/predict/batch` | Predict for up to 100 incidents in one call. |

### Example request

```bash
curl -X POST "http://localhost:7860/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "start_datetime": "2026-06-19 08:30:00+0530",
        "latitude": 12.9352,
        "longitude": 77.6146,
        "event_type": "Breakdown",
        "event_cause": "vehicle_breakdown",
        "priority": "High",
        "zone": "East Zone 1",
        "junction": "Silk Board Junction",
        "veh_type": "Truck",
        "description": "Heavy truck breakdown causing blockage",
        "requires_road_closure": true
      }'
```

### Example response

```json
{
  "prediction": {
    "class_id": 1,
    "label": "30–90 mins (Minor)",
    "confidence": "47.2%"
  },
  "class_probabilities": {
    "<30 mins (Quick)": 0.21,
    "30–90 mins (Minor)": 0.4720,
    "90–240 mins (Major)": 0.18,
    ">240 mins (Severe)": 0.138
  },
  "input_summary": {
    "start_datetime": "2026-06-19 08:30:00+0530",
    "location": {"lat": 12.9352, "lon": 77.6146},
    "event_cause": "vehicle_breakdown",
    "zone": "East Zone 1"
  }
}
```

### Required vs. optional fields

Only `start_datetime`, `latitude`, and `longitude` are required. Everything else (`event_type`, `event_cause`, `priority`, `zone`, `junction`, `veh_type`, `corridor`, `direction`, `police_station`, `reason_breakdown`, `cargo_material`, `status`, `description`, `comment`, `created_date`, `endlatitude`/`endlongitude`, `requires_road_closure`) is optional but improves prediction accuracy — missing categorical fields default to `"Unknown"`, and missing zone/junction medians fall back to the global median.

The API internally reconstructs the exact ~40-feature vector used at training time (temporal, spatial, text/keyword, and operational features) via `build_feature_row()`, so the request schema mirrors the notebook's feature engineering logic 1:1.

---

## Notes

- The `zone_median_resolution` and `junction_median_resolution` features are target-encoding proxies computed from historical data — they must be regenerated (`extract_medians.py`) whenever the model is retrained on new data, or predictions will drift out of sync with the trained model.
- The model and its weights are hosted on the Hugging Face Hub; `app.py` will fail to load (returning `503` on `/predict`) if the model can't be downloaded — check network access and the `repo_id`/`filename` constants if this happens.

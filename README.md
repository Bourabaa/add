# ADD PFE Project

This project explores an intelligent routing workflow for e-Participation requests on `data.gov.ma`.

## Dockerized demo

The project can be launched as a self-contained demo with Docker Compose:

```powershell
docker compose up -d --build
```

This starts one service:

- `dashboard`: the Flask review interface exposed on `http://localhost:5000/notifications`
- the dashboard container runs the app through `gunicorn`

Useful commands:

```powershell
docker compose logs -f dashboard
docker compose down
```

Integration note:

- feedback collection is expected to be handled by the portal expert's endpoint/microservice
- this project receives feedbacks and organizations as JSON and returns routing decisions
- duplicate notifications are prevented by the SQLite `UNIQUE` constraint on `feedback_url` and by the `upsert_notification()` logic in `add_pfe/notifications/repository.py`

## Modular architecture

The project now follows a modular monolith layout that keeps the current PoC simple while preparing a future split into dedicated services or endpoints.

### Core application package

```text
add_pfe/
  config.py                 # shared paths, runtime settings, status enums
  lexical/
    matcher.py              # sparse lexical matcher and text feature extraction
  semantic/
    embedding_matcher.py    # ONNX multilingual embeddings matcher
  routing/
    service.py              # reusable routing service and decision thresholds
  notifications/
    repository.py           # SQLite persistence
    service.py              # notification pipeline / upsert workflow
  integration/
    payloads.py             # JSON payload normalization for external endpoints
```

### Legacy scripts kept as wrappers

The `scripts/` entrypoints are still available for demo and evaluation, but they now act as thin wrappers around the shared `add_pfe/` modules. This keeps command-line usage stable while avoiding logic duplication between the CLI, the dashboard and the routing API.

## Project structure

```text
.
|-- add_pfe/
|   |-- config.py
|   |-- lexical/
|   |-- semantic/
|   |-- routing/
|   `-- notifications/
|-- scripts/
|   |-- extract_organisations.py
|   |-- build_enriched_organisations.py
|   |-- matcher.py
|   |-- embedding_matcher.py
|   |-- build_notifications.py
|   |-- evaluate_matcher.py
|   |-- evaluate_embedding_matcher.py
|   `-- notifications_store.py
|-- data/
|   |-- raw/
|   |   |-- organisations.csv
|   |   `-- feedbacks.csv
|   |-- processed/
|   |   |-- organisations_enriched.csv
|   |   `-- feedbacks_eval_ready.csv
|   `-- cache/              # runtime cache, not committed
|-- models/
|   `-- multilingual-e5-small-onnx/
|       |-- model_O4.onnx
|       `-- tokenizer.json
|-- templates/
|-- static/
|-- app.py
`-- requirements.txt
```

## Offline model artifacts

The semantic matcher runs locally with ONNX Runtime. It does not call an external AI API and it no longer downloads files from Hugging Face at runtime.

Required local files:

```text
models/multilingual-e5-small-onnx/model_O4.onnx
models/multilingual-e5-small-onnx/tokenizer.json
```

The ONNX model is large, so use Git LFS or an internal artifact registry if the repository is pushed to a remote Git server.

## Main workflow

1. Export organizations

```powershell
python scripts\extract_organisations.py
```

2. Build enriched organization profiles

```powershell
python scripts\build_enriched_organisations.py
```

3. Test the lexical matcher on a single request

```powershell
python scripts\matcher.py --request "Je souhaite acceder aux jugements et decisions judiciaires du portail Mahakim" --top-k 3
```

4. Test the hybrid matcher on a single request

```powershell
python scripts\embedding_matcher.py --request "Je souhaite acceder aux jugements et decisions judiciaires du portail Mahakim" --top-k 3
```

5. Build or refresh the notification queue from a prepared feedback JSON/CSV source

```powershell
python scripts\build_notifications.py --feedbacks data\processed\feedbacks_eval_ready.csv
```

6. Run the final hybrid evaluation

```powershell
python scripts\evaluate_embedding_matcher.py --top-k 3
```

## Integration-ready endpoint

The Flask app now exposes a lightweight internal routing endpoint that can be used before a full microservice split:

```text
POST /api/route
```

Example JSON payload:

```json
{
  "title": "Demande de donnees sur les accidents de la route",
  "description": "Je souhaite obtenir des statistiques recentes...",
  "top_k": 3
}
```

This returns the hybrid routing decision, confidence values and ranked candidates, which makes the current monolith easier to plug into a future portal integration.

For the portal expert's microservice, use the batch JSON endpoint:

```text
POST /api/route-batch
```

Example JSON payload:

```json
{
  "organizations": [
    {
      "organization_slug": "narsa",
      "organization": "NARSA",
      "full_name": "Agence Nationale de la Securite Routiere",
      "aliases": ["narsa"],
      "domains": ["transport", "securite routiere"],
      "keywords_fr": ["accident", "route", "vehicule"],
      "profile_text": "NARSA transport securite routiere accident route vehicule"
    }
  ],
  "feedbacks": [
    {
      "feedback_id": "REQ-001",
      "title": "Demande de donnees sur les accidents de la route",
      "description": "Je souhaite obtenir des statistiques recentes."
    }
  ],
  "top_k": 3
}
```

Each response includes:

- `embedding_inference_ms`
- `lexical_matching_ms`
- `hybrid_scoring_ms`
- `total_routing_ms`

## Notes

- `data/processed/` stores enriched data used by the matcher and evaluation.
- `reports/` stores generated evaluation outputs and is ignored by Git.
- `data/cache/` stores reusable cached HTML/API resolutions and embeddings and is ignored by Git.
- The current design is still a monolith, but the business logic is now organized so that the routing engine and notification pipeline can be extracted later into dedicated services.

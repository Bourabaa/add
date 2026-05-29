# Expert Review Notes

## Goal

This repository contains a deployable prototype for intelligent routing of `data.gov.ma` e-Participation requests. The system is intentionally organized as a modular monolith so it remains easy to run today and can later be split into a routing microservice and admin interface.

## What changed for integration readiness

- The routing logic is centralized in `add_pfe/routing/service.py`.
- The lexical matcher is isolated in `add_pfe/lexical/`.
- The ONNX multilingual embedding matcher is isolated in `add_pfe/semantic/`.
- SQLite persistence is isolated in `add_pfe/notifications/repository.py`.
- Notification generation is isolated in `add_pfe/notifications/service.py`.
- CLI scripts in `scripts/` remain available as wrappers for demo and evaluation.
- The Flask app exposes the admin dashboard and integration-ready endpoints: `POST /api/route` and `POST /api/route-batch`.
- Feedbacks are expected as JSON from the portal team's endpoint/microservice. This repository no longer contains feedback scraping logic.

## Offline model policy

The routing engine does not send requests to an external AI service. Embeddings are computed locally with ONNX Runtime.

The model artifacts are expected locally under:

```text
models/multilingual-e5-small-onnx/
  model_O4.onnx
  tokenizer.json
```

The code no longer downloads model files at runtime. If those files are missing, startup fails with an explicit error. For Git hosting, the ONNX file should be managed with Git LFS or an internal artifact registry.

## Runtime components

- `dashboard`: Flask dashboard for notification review.
- `routing endpoint`: `POST /api/route`, reusable for a single request.
- `batch routing endpoint`: `POST /api/route-batch`, reusable by the portal expert's feedback microservice.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python scripts\evaluate_matcher.py --top-k 3
.\.venv\Scripts\python scripts\embedding_matcher.py --request "Demande de donnees sur les accidents de la route" --top-k 3
.\.venv\Scripts\python app.py
```

Dashboard:

```text
http://localhost:5000/notifications
```

## Run with Docker

```powershell
docker compose up -d --build
docker compose logs -f dashboard
```

## Integration endpoint example

```powershell
curl -X POST http://localhost:5000/api/route `
  -H "Content-Type: application/json" `
  -d "{\"title\":\"Demande de donnees sur les accidents de la route\",\"description\":\"Je souhaite obtenir des statistiques recentes\",\"top_k\":3}"
```

Batch endpoint expected from the portal integration:

```text
POST /api/route-batch
```

The request body contains `organizations`, `feedbacks`, and optional `top_k`. Each routed feedback returns the decision, confidence, margin, candidates and execution timings:

```text
embedding_inference_ms
lexical_matching_ms
hybrid_scoring_ms
total_routing_ms
```

## Current evaluation reference

Use:

```powershell
python scripts\evaluate_embedding_matcher.py --top-k 3
```

Reference metrics on `data/processed/feedbacks_eval_ready.csv`:

- Strong labels: `22`
- Top-1 accuracy: `72.73%`
- Top-3 accuracy: `95.45%`
- Auto-route coverage: `36.36%`
- Auto-route precision: `100%`

## Integration roadmap

1. Keep the current modular monolith for the expert demo.
2. Connect the portal-provided feedback microservice to `POST /api/route-batch`.
3. Extract `RoutingService` into a dedicated microservice if the portal team wants independent deployment.
4. Add review feedback fields such as `final_org_slug`, `reviewed_by`, and `reviewed_at` before any supervised training.
5. Move from SQLite to PostgreSQL for concurrent production usage.

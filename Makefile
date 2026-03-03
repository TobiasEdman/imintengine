# ── IMINT Engine — Development targets ────────────────────────────────────
#
# Common commands:
#   make build              Build Docker image (ARM64 on M1)
#   make test               Run unit tests
#   make test-seasonal      Test seasonal fetch locally (CDSE)
#   make test-seasonal-des  Test seasonal fetch locally (DES)
#   make test-compare       Fetch from both, compare reflectance
#   make docker-seasonal    Run seasonal fetch in Docker container
#   make submit-dry         Dry-run ColonyOS job submission
#

.PHONY: build test test-seasonal test-seasonal-des test-compare \
        bench-batch bench-batch-cdse bench-batch-quick \
        docker-seasonal docker-seasonal-des docker-analysis \
        submit-dry submit-live status clean help

PYTHON := .venv/bin/python
IMAGE  := imint-engine:latest

# ── Docker ────────────────────────────────────────────────────────────────

build:  ## Build Docker image (ARM64 native on M1 Max)
	docker build --platform linux/arm64 -t $(IMAGE) .

build-no-cache:  ## Full rebuild without cache
	docker build --platform linux/arm64 --no-cache -t $(IMAGE) .

# ── Unit tests ────────────────────────────────────────────────────────────

test:  ## Run all unit tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-utils:  ## Run utils tests only
	$(PYTHON) -m pytest tests/test_utils.py -v

test-cdse:  ## Run CDSE/copernicus-specific tests
	$(PYTHON) -m pytest tests/test_fetch.py tests/test_utils.py -v \
		-k "copernicus or cdse or CDSE or Copernicus or Sentinel2"

# ── Local seasonal fetch tests (no Docker) ────────────────────────────────

test-seasonal:  ## Test seasonal fetch locally via CDSE (fastest)
	$(PYTHON) scripts/test_seasonal_local.py --source copernicus

test-seasonal-des:  ## Test seasonal fetch locally via DES
	$(PYTHON) scripts/test_seasonal_local.py --source des

test-compare:  ## Fetch from both backends, compare reflectance
	$(PYTHON) scripts/test_seasonal_local.py --compare

# ── Batch fetch benchmark ────────────────────────────────────────────────

bench-batch:  ## Benchmark sequential vs batch-job fetch (DES)
	$(PYTHON) scripts/test_batch_fetch.py --source des

bench-batch-cdse:  ## Benchmark sequential vs batch-job fetch (CDSE)
	$(PYTHON) scripts/test_batch_fetch.py --source copernicus

bench-batch-quick:  ## Quick batch benchmark (skip STAC discovery)
	$(PYTHON) scripts/test_batch_fetch.py --skip-discovery --strategies merged batch

# ── Docker seasonal fetch (simulates ColonyOS) ───────────────────────────

docker-seasonal:  ## Run seasonal fetch in Docker (CDSE)
	docker compose run --rm seasonal-fetch-cdse

docker-seasonal-des:  ## Run seasonal fetch in Docker (DES)
	docker compose run --rm seasonal-fetch-des

docker-analysis:  ## Run IMINT analysis in Docker
	docker compose run --rm analysis

# ── ColonyOS job submission ───────────────────────────────────────────────

submit-dry:  ## Dry-run: show what jobs would be submitted
	$(PYTHON) scripts/submit_seasonal_jobs.py \
		--sources copernicus,des --dry-run --max-jobs 10

submit-live:  ## Submit seasonal fetch jobs to ColonyOS (first 100)
	$(PYTHON) scripts/submit_seasonal_jobs.py \
		--sources copernicus,des --max-jobs 100

submit-all:  ## Submit ALL seasonal fetch jobs to ColonyOS
	$(PYTHON) scripts/submit_seasonal_jobs.py --sources copernicus,des

status:  ## Check ColonyOS progress
	$(PYTHON) scripts/submit_seasonal_jobs.py --status

# ── Cleanup ───────────────────────────────────────────────────────────────

clean:  ## Remove test tiles and cache
	rm -rf data/test_tiles/
	rm -rf __pycache__ .pytest_cache

clean-docker:  ## Remove Docker image
	docker rmi $(IMAGE) 2>/dev/null || true

# ── Help ──────────────────────────────────────────────────────────────────

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

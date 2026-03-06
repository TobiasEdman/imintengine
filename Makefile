# ── IMINT Engine — Development targets ────────────────────────────────────
#
# Common commands:
#   make build              Build Docker image (ARM64 on M1)
#   make build-cuda         Build CUDA Docker image (x86_64 for H100)
#   make test               Run unit tests
#   make test-seasonal      Test seasonal fetch locally (CDSE)
#   make test-seasonal-des  Test seasonal fetch locally (DES)
#   make test-compare       Fetch from both, compare reflectance
#   make docker-seasonal    Run seasonal fetch in Docker container
#   make submit-dry         Dry-run ColonyOS job submission
#   make vm-setup VM_HOST=user@host   Bootstrap H100 VM
#   make vm-train VM_HOST=user@host   Launch training on VM
#

.PHONY: build build-cuda test test-seasonal test-seasonal-des test-compare \
        bench-batch bench-batch-cdse bench-batch-quick \
        docker-seasonal docker-seasonal-des docker-analysis \
        colony-up colony-down colony-logs colony-status colony-submit \
        colony-monitor submit-dry submit-live status \
        train-dev train-test train-prod \
        vpp-prefetch vpp-submit-dry vpp-submit \
        s2-submit-dry s2-submit s2-submit-all s2-status s2-local \
        vm-setup vm-transfer vm-train vm-status vm-attach vm-logs \
        clean help

PYTHON     := .venv/bin/python
IMAGE      := imint-engine:latest
CUDA_IMAGE := imint-engine:cuda
DATA_DIR   ?= ~/training_data
ENV        ?= dev
ARGS       ?=
VM_HOST    ?= user@ice-connect-vm

# ── Docker ────────────────────────────────────────────────────────────────

build:  ## Build Docker image (ARM64 native on M1 Max)
	docker build --platform linux/arm64 -t $(IMAGE) .

build-no-cache:  ## Full rebuild without cache
	docker build --platform linux/arm64 --no-cache -t $(IMAGE) .

# ── CUDA Docker (x86_64 / H100) ────────────────────────────────────────

build-cuda:  ## Build CUDA Docker image (x86_64 for H100)
	docker build --platform linux/amd64 -f Dockerfile.cuda -t $(CUDA_IMAGE) .

build-cuda-no-cache:  ## Full CUDA rebuild without cache
	docker build --platform linux/amd64 -f Dockerfile.cuda --no-cache -t $(CUDA_IMAGE) .

# ── H100 VM management (ICE Connect) ───────────────────────────────────

vm-setup:  ## Bootstrap H100 VM (install deps, clone repo, setup venv)
	./scripts/setup_h100_vm.sh $(VM_HOST)

vm-transfer:  ## Transfer dataset to H100 VM via rsync
	./scripts/setup_h100_vm.sh $(VM_HOST) --transfer-data

vm-train:  ## Launch training on H100 VM in tmux
	./scripts/ssh_train.sh $(VM_HOST)

vm-status:  ## Check GPU status on H100 VM
	@ssh $(VM_HOST) nvidia-smi

vm-attach:  ## Attach to training tmux session on H100 VM
	ssh $(VM_HOST) -t 'tmux attach -t training'

vm-logs:  ## Tail training log on H100 VM
	@ssh $(VM_HOST) 'tail -50 ~/ImintEngine/data/lulc_full/train.log'

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

# ── ColonyOS local platform ──────────────────────────────────────────────

COLONY_COMPOSE := docker compose -f docker-compose.colonyos.yml --env-file .env.colonyos

colony-up:  ## Start ColonyOS platform (TimescaleDB + MinIO + server + executor)
	$(COLONY_COMPOSE) up -d
	@echo ""
	@echo "  ColonyOS starting up..."
	@echo "  Server:       http://localhost:50080"
	@echo "  MinIO console: http://localhost:9001 (admin / admin12345)"
	@echo ""
	@echo "  Wait ~10s, then: make colony-status"

colony-down:  ## Stop ColonyOS platform
	$(COLONY_COMPOSE) down

colony-reset:  ## Stop and destroy all ColonyOS data (volumes)
	$(COLONY_COMPOSE) down -v

colony-logs:  ## Follow ColonyOS logs
	$(COLONY_COMPOSE) logs -f colonies-server docker-executor

colony-status:  ## Check ColonyOS is running (colony + executor)
	@echo "=== Colonies server ==="
	@docker compose -f docker-compose.colonyos.yml ps colonies-server 2>/dev/null || echo "not running"
	@echo ""
	@echo "=== Docker executor ==="
	@docker compose -f docker-compose.colonyos.yml ps docker-executor 2>/dev/null || echo "not running"

colony-submit:  ## Submit a single test job to ColonyOS
	colonies function submit --spec config/seasonal_fetch_job.json

colony-monitor:  ## Monitor seasonal fetch progress (writes dashboard JSON)
	$(PYTHON) scripts/monitor_seasonal_jobs.py --data-dir $(DATA_DIR) --interval 15

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

# ── Environment-aware training ─────────────────────────────────────────────

train-dev:  ## Train with dev environment (M1 Max / MPS)
	IMINT_ENV=dev $(PYTHON) scripts/train_lulc.py $(ARGS)

train-test:  ## Quick test run (2 epochs, CPU, small batch)
	IMINT_ENV=test $(PYTHON) scripts/train_lulc.py --epochs 2 --batch-size 2 $(ARGS)

train-prod:  ## Train with production settings (H100 / CUDA)
	IMINT_ENV=prod $(PYTHON) scripts/train_lulc.py $(ARGS)

# ── VPP phenology enrichment ─────────────────────────────────────────────

vpp-prefetch:  ## Prefetch VPP locally for all tiles (ThreadPoolExecutor)
	$(PYTHON) scripts/submit_vpp_jobs.py --local --workers 4

vpp-submit-dry:  ## Dry-run: show VPP jobs that would be submitted to ColonyOS
	$(PYTHON) scripts/submit_vpp_jobs.py --dry-run --max-jobs 10

vpp-submit:  ## Submit VPP enrichment jobs to ColonyOS
	$(PYTHON) scripts/submit_vpp_jobs.py

# ── S2 Process API fetch (Sentinel Hub) ──────────────────────────────────

s2-submit-dry:  ## Dry-run: show S2 Process API jobs that would be submitted
	$(PYTHON) scripts/submit_s2_jobs.py --dry-run --max-jobs 10

s2-submit:  ## Submit first 100 S2 Process API fetch jobs to ColonyOS
	$(PYTHON) scripts/submit_s2_jobs.py --max-jobs 100

s2-submit-all:  ## Submit ALL S2 Process API fetch jobs to ColonyOS
	$(PYTHON) scripts/submit_s2_jobs.py

s2-status:  ## Check S2 Process API fetch progress
	$(PYTHON) scripts/submit_s2_jobs.py --status

s2-local:  ## Run S2 Process API fetch locally (ThreadPoolExecutor)
	$(PYTHON) scripts/submit_s2_jobs.py --local --workers 4

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

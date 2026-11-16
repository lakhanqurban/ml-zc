PYTHON ?= python
PORT ?= 9696
IMAGE ?= hotel-cancellation:midterm

.PHONY: help setup data train train-gpu train-quick serve predict docker-build docker-run docker-test compose-up compose-down compose-logs fly-deploy fly-logs render-info clean

help:
	@echo "setup        create the virtualenv and install dependencies"
	@echo "data         download the raw dataset into data/raw/"
	@echo "train        train the model and write artifacts/"
	@echo "train-gpu    same, but force XGBoost onto the CUDA GPU"
	@echo "train-quick  reduced grid, for a fast smoke test"
	@echo "serve        start the FastAPI service on port $(PORT)"
	@echo "predict      score a sample booking from the command line"
	@echo "docker-build build the container image"
	@echo "docker-run   run the container on port $(PORT)"
	@echo "test         smoke-test the health and predict endpoints"
	@echo "clean        remove generated artifacts"

setup:
	$(PYTHON) -m venv .venv
	.venv/Scripts/activate 2>/dev/null || .venv/bin/activate
	pip install --upgrade pip
	pip install -r requirements.txt -r requirements-dev.txt

data:
	$(PYTHON) data/download_data.py

train:
	$(PYTHON) train.py

train-gpu:
	$(PYTHON) train.py --device cuda

train-quick:
	$(PYTHON) train.py --quick

serve:
	$(PYTHON) -m uvicorn serve:app --host 0.0.0.0 --port $(PORT)

predict:
	$(PYTHON) predict.py '{"hotel": "City Hotel", "lead_time": 120, "adr": 110.5, "arrival_date_month": "August", "arrival_date_day_of_week": "Tuesday", "stays_in_weekend_nights": 1, "stays_in_week_nights": 3, "adults": 2, "children": 1, "babies": 0, "country": "PRT", "market_segment": "Online TA", "deposit_type": "No Deposit", "customer_type": "Transient", "previous_cancellations": 0, "previous_bookings_not_canceled": 1, "booking_changes": 0, "required_car_parking_spaces": 0}'

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):$(PORT) $(IMAGE)

docker-test: docker-build
	@echo "--- running the container and checking the endpoints ---"
	@docker rm -f hotel-cancel-test >/dev/null 2>&1 || true
	@docker run -d --name hotel-cancel-test -p $(PORT):$(PORT) $(IMAGE) >/dev/null
	@sleep 8
	@echo "health : $$(curl -s http://localhost:$(PORT)/health)"
	@echo "predict: $$(curl -s http://localhost:$(PORT)/predict -H 'Content-Type: application/json' -d '{\"hotel\": \"City Hotel\", \"lead_time\": 120, \"adr\": 110.5}')"
	@docker rm -f hotel-cancel-test >/dev/null
	@echo "container removed"

compose-up:
	docker compose up -d --build
	@echo "waiting for the service to become healthy..."
	@sleep 10
	@echo "health : $$(curl -s http://localhost:$(PORT)/health)"

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f

fly-deploy:
	@command -v fly >/dev/null 2>&1 || { echo "install the flyctl CLI: https://fly.io/docs/hands-on/install-flyctl/"; exit 1; }
	fly launch --no-deploy
	fly deploy
	@echo ""
	@echo "your service is live - check it with:"
	@echo "  fly status"
	@echo "  fly open"

fly-logs:
	fly logs

render-info:
	@echo "Render deploys from render.yaml:"
	@echo "  1. push this repository to GitHub"
	@echo "  2. https://render.com/ -> New -> Blueprint -> select the repo"
	@echo "  3. deploy; Render detects render.yaml automatically"

check-readme:
	$(PYTHON) tools/check_readme.py

check-readme-tone:
	$(PYTHON) tools/check_readme_tone.py

check-subgroups:
	$(PYTHON) tools/check_subgroups.py

check: check-readme check-readme-tone check-subgroups

test:
	$(PYTHON) -c "import urllib.request,json,sys; \
h=json.loads(urllib.request.urlopen('http://localhost:$(PORT)/health').read()); \
print('health:', h); \
req=urllib.request.Request('http://localhost:$(PORT)/predict', \
data=json.dumps({'hotel':'City Hotel','lead_time':120,'adr':110.5}).encode(), \
headers={'Content-Type':'application/json'}); \
print('predict:', urllib.request.urlopen(req).read().decode())"

clean:
	rm -rf artifacts/*.pkl artifacts/*.csv artifacts/*.json
	find . -name __pycache__ -type d -exec rm -rf {} +

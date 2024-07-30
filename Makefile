install-dev:
	pip install -r requirements-dev.txt

install-req:
	pip install -r requirements.txt
	
install: install-req install-dev

format:
	ruff check --fix .
	ruff format .

mypy:
	mypy .

lint: format mypy

test:
	pytest .

run:
	python3 train_gpt2.py

install-dev:
	pip install -r requirements-dev.txt

install-req:
	pip install -r requirements.txt
	
install: install-req install-dev

format:
	ruff format .

mypy:
	mypy .

check: black mypy

test:
	pytest .
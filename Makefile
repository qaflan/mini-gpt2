install-dev:
	pip install -r requirements-dev.txt

install-req:
	pip install -r requirements.txt
	
install: install-dev install-req
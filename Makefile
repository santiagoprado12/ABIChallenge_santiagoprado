VENV=venv
virtual_env:
	python3 -m pip install virtualenv
	python3 -m virtualenv $(VENV)

install: virtual_env
	python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements.txt
	python3 -m pip install -r api/requirements.txt

test: 
	python3 -m ptest

test-coverage:
	python3 -m coverage run
	python3 -m coverage report

coverage-html:
	python3 -m coverage html

coverage-report:
	python3 -m coverage report

train:
	python3 -m src.CLI train --model=knn --model=random_forest --model=gradient_boosting -th=0.7 -rm

create-docker-image:
	aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 019994626350.dkr.ecr.us-east-2.amazonaws.com
	docker build -t titanic-api .
	docker tag titanic-api:latest 019994626350.dkr.ecr.us-east-2.amazonaws.com/titanic-api:latest
	docker push 019994626350.dkr.ecr.us-east-2.amazonaws.com/titanic-api:latest

dummy:
	@echo "Doing nothing"


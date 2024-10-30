FROM python:3.10-slim-buster

USER nonroot

WORKDIR /api

COPY api/requirements.txt .

COPY src/ ./src

RUN apt --no-install-recommends install -y build-essential

RUN apt-get install -y make

COPY api/ ./api

RUN pip install -U pip && pip install -r api/requirements.txt

COPY Makefile .

COPY models/best_model.pkl ./models/best_model.pkl

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
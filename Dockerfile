FROM python:3.9.12-slim-bullseye

RUN python -m pip install --upgrade pip
RUN pip install poetry==1.1.13
COPY poetry.lock pyproject.toml /
RUN poetry config virtualenvs.create false \
  && poetry install --no-dev --no-interaction --no-ansi

RUN apt-get update \
  && apt-get -y install tesseract-ocr \
  && apt-get -y install ffmpeg libsm6 libxext6

COPY api /code/
WORKDIR /code

RUN groupadd -g 1000 basicuser && \
  useradd -r -u 1000 -g basicuser basicuser
USER basicuser

EXPOSE 5000

CMD [ "python", "app.py" ]

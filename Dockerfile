FROM fedora:43 as base

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1

FROM base as runtime

RUN dnf install -y python3 python3-devel python3-pip poetry cmake gcc-c++ boost-devel git graphviz-devel pybind11-devel && dnf clean all
COPY pyproject.toml poetry.lock ./
ENV POETRY_VIRTUALENVS_IN_PROJECT 1
ENV POETRY_NO_INTERACTION 1
RUN poetry install --no-root --without=dev
ENV PATH="/.venv/bin:${PATH}"

FROM runtime as app
COPY . /app
WORKDIR /app

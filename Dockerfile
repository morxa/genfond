FROM fedora:latest as base

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1

FROM base as runtime

RUN dnf install -y python3 python3-devel python3-pip cmake gcc-c++ boost-devel git graphviz-devel && dnf clean all
ENV PATH="/root/.local/bin:${PATH}"
RUN python3 -m pip install pip
RUN pip install pipenv
COPY Pipfile Pipfile.lock ./
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy
ENV PATH="/.venv/bin:${PATH}"

FROM runtime as app
COPY . /app
WORKDIR /app

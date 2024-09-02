genfond_env.sif: genfond_env.def genfond_env.tar
	apptainer build -F $@ $<
genfond_env.tar: Dockerfile pyproject.toml poetry.lock
	podman build --pull --target=runtime -t genfond_env .
	rm -f $@
	podman save -o $@ genfond_env

.PHONY: genfond_env.tar

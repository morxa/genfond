genfond_env.sif: genfond_env.def genfond_env.tar
	apptainer build -F $@ $<
genfond_env.tar: Dockerfile Pipfile Pipfile.lock
	podman build --target=runtime -t genfond_env .
	rm -f $@
	podman save -o $@ genfond_env

.PHONY: genfond_env.tar

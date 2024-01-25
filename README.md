# Generalized FOND Planning

## Installation

There are two option to run `genfond`, in a local virtual env or in a container.

### Installing in a virtualenv

You can install all dependencies in a virtualenv managed by `pipenv`:


1. Install `pipenv`:
   ```
   pip install --user pipenv
   ```
1. Install dependencies:
   ```
   pipenv install
   ```
1. Activate a virtualenv shell (you will need to redo this every time you open a new terminal):
   ```
   pipenv shell
   ```

You can then run all the scripts with the virtualenv python, e.g.:
```
python genfond.py -h
```

### Building a Container Image

Alternatively, you can build a container image with docker or podman:

```
docker build -t genfond .
```

You can then use a container to run all the scripts, e.g.:
```
docker run --rm -ti genfond python genfond.py -h
```

## Learning Policies

In the following, we assume you are using a local virtualenv, but all commands work similarly by prefixing `docker run --rm -ti genfond`, as shown above.

To learn a policy for a domain, run the following, e.g., for `acrobatics`:
```
python genfond.py domains/non-deterministic/acrobatics/{domain.pddl,p*.pddl}
```

There are multiple options, e.g., you can use the transition-based variant by specifing `--constraints trans`. See `python genfond.py -h` for a full list of options.

### One-shot solver

The solver implemented in `genfond.py` iteratively solves the given problems by incrementally adding problems to the training set and by iteratively increasing the maximal feature complexity. If you instead want to run the solver once for a given feature complexity on a set of problems, you can use the oneshot solver implemented in `genfond_oneshot.py`, e.g.,:
```
python genfond_oneshot.py --complexity 6 --max-cost 10 domains/non-deterministic/acrobatics/{domain.pddl,p0002*}
```

## Executing a policy

After learning a policy and writing it to a file (by writing a policy file with `--output <policyfile>`, both for `genfond.py` and `genfond_oneshot.py`), you can execute the policy with `execute_policy.py`, e.g.,:
```
python execute_policy.py domains/non-deterministic/acrobatics/{domain.pddl,p0005.pddl} acrobatics.policy
```

# Copyright

All rights reserved by the authors.

Upon publication, this project will be made publicly available and licensed under an open-source license.

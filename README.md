# Generalized FOND Planning

## Installation

There are two option to run `genfond`, in a local virtual env or in a container.

### Installing in a virtualenv

You can install all dependencies in a virtualenv managed by `poetry`:


1. Install `poetry`:
   ```
   pip install --user poetry
   ```
1. Install dependencies:
   ```
   poetry install --no-root
   ```
1. Activate a virtualenv shell (you will need to redo this every time you open a new terminal):
   ```
   poetry shell
   ```

You can then run all the scripts with the virtualenv python, e.g.:
```
python -m genfond -h
```

### Building a Container Image

Alternatively, you can build a container image with docker or podman:

```
docker build -t genfond .
```

You can then use a container to run all the scripts, e.g.:
```
docker run --rm -ti genfond python -m genfond -h
```

## Learning Modes

This repository contains two fundamentally different ways of learning a policy:

1. *Unsupervised Learning*: Learn a rule-based policy by selecting policy transitions in the fully expanded state space and selecting features that distinguish the selected from the non-selected transitions.
2. *Supervised Learning*: Given example trajectories, learn a datalog policy that follows the input trajectories.

The two dimensions (unsupervised/supervised and rule-based/datalog) are independent in the implementation — supervision is enabled by the `use_example_plans` config option — but only the two combinations described below are configured and tested.

### Unsupervised Learning of Rule-Based Policies for FOND

The details of this approach are described in:

1. Hofmann, T. & Geffner, H. Learning generalized policies for fully observable non-deterministic planning domains. in Proceedings of the 33rd International Joint Conference on Artificial Intelligence vol. 7 6733–6742 (2024).

In this setting, policy rules are of the form C ↦ E, where
* C is a condition that must be satisfied in the current state
* E is a set of alternative descriptions of **feature changes** in the transition. For FOND, **one of the possible outcomes** must match one of the descriptions in E.

Additionally, a policy contains constraints that rule out actions leading towards dead-ends.
There are two variants, selected with `--type`:
* `state` (the default): state constraints B in the same form as C, with the following semantics: An action may only be applied if **none of the possible outcomes** matches any state constraint B.
* `trans`: transition constraints of the same form C ↦ E as the rules, with the following semantics: If C is satisfied in the current state, then the action may only be applied if **none of the possible outcomes** matches E.

For deterministic domains, `--type d2l` learns rules of the same form, but without constraints, and the single outcome must match E exactly.

#### Learning
From a high-level perspective, the approach works as follows:

1. Fully expand the state space of (several) small training problems
2. Select good transitions that describe a policy in the **all-outcome relaxation**:
   * every solvable state must have at least one outcome transition selected
   * and all trajectories must eventually end in a goal state

   The all-outcome relaxation treats all nondeterministic outcomes as separate actions.
   To deal with FOND dead-ends, we additionally require:
   * For at least one selected transition per state, no alternative outcome of the corresponding action may end in a FOND dead-end.
     Further transitions may be selected from other actions, as long as the selected outcome itself is not a dead-end.

   The dead-ends are precomputed during state space expansion.
3. Select features that distinguish
   * good from bad transitions
   * FOND dead-ends from alive states
   * goal from non-goal states
4. Extract a rule-based policy from the selected transitions and features.

Step (2) and (3) are done simultaneously in an ASP solver.
There is an outer loop around the solver that iteratively enables the unrestricted feature generators, increases the maximal feature complexity, and adds unsolved problems to the training set.
Each round must improve on the previous policy, i.e., the total complexity of the selected features must strictly decrease.

### Supervised learning of Datalog Policies for Deterministic Planning

**This approach is work in progress and does not fully work yet**; expect bugs and domains that it fails to solve.

It currently only supports deterministic domains, because the example plans are computed with a classical planner.

In contrast to the rule-based policy, a datalog policy does not have an action model.
Instead, rules are of the form

    action(X, Y, Z) ← state_feature, concept(X), role(Y, Z)

where:

* a state feature describes the current state
* a concept can be understood as a formula with a single free variable, which is bound to one action argument
* a role can be understood as a formula with two free variables, which is bound to two action arguments, and thus describes the relation of those arguments

State features, concepts, and roles are constructed from the same feature pool as in the rule-based approach.

#### Learning
From a high-level perspective, the approach works as follows:

1. Compute (several) example plans for each training problem with an external planner (by default `siw`, where diverse plans are obtained by branching over the serializations and by restarting with permuted action and goal orders)
2. Create a partial state space that contains the example trajectories and, for each state on a trajectory, every possible successor state resulting from all applicable actions.
   Successors that are not on any trajectory are left unexpanded; they serve as the negative examples.
3. Select good transitions that describe a policy:
   * for every state on a trajectory, select at least one transition that leads to a state on a trajectory.
     Note that this is a condition on the successor state, not on the action: an action that occurs in no example plan may be selected if its successor lies on a trajectory.
   * all trajectories induced by the selected transitions must eventually end in a goal state
   * select features, concepts, and roles that distinguish the selected from the non-selected actions.
     Only actions with the same action name need to be distinguished, because the action name is part of the rule head.
     The successor state is not used here, as a datalog policy has no action model.
4. Extract a datalog policy from the selected actions, features, concepts, and roles.

Step (3) is done in the ASP solver.
There is an outer loop around the solver that iteratively adds example plans, enables the unrestricted feature generators, increases the maximal feature complexity, and adds unsolved problems to the training set, again requiring each round to strictly improve on the previous policy.

##### Frontier expansion

Restricting the state space to the example trajectories can make a round unsatisfiable even though a small amount of extra state space would suffice.
With `frontier_expansion` (enabled by default for `--type datalog`), the solver may additionally select a transition into an unexpanded successor, assuming optimistically that it is solvable.
Such a transition carries a cost at a *higher* optimization priority than the feature complexity, so it is only used when the round is otherwise unsatisfiable.

The states selected this way are then handed back to the planner, which is asked to solve the problem starting from that state.
If it finds a plan, the plan is appended to the path that reaches the state and added as a new example plan, so the state space grows exactly where the solver needed it, rather than blindly.
If it finds no plan, the state is marked as a dead end and the solver may no longer select transitions into it.
Either way the round is retried; a policy is only accepted once the solver no longer relies on any unexpanded state.

Frontier expansion is only available while no policy has been found for the current training set.
Once a policy exists and the outer loop is merely trying to improve on its cost, the frontier is closed off again, since growing the state space there buys cheaper features rather than solvability.

The relevant options are `frontier_expansion`, `max_frontier_expansions` (a loop guard on the number of expansion rounds), `max_frontier_states_per_round`, and `max_frontier_transitions` (a hard cap on the frontier transitions in a single model).
Note that the planner is incomplete, so failing to find a plan does not prove that a state is a dead end; this can only prevent a policy from being found, never yield an incorrect one, because the final policy is verified on all problems anyway.


## Learning Policies

In the following, we assume you are using a local virtualenv, but all commands work similarly by prefixing `docker run --rm -ti genfond`, as shown above.

To learn a policy for a domain, run the following, e.g., for `acrobatics`:
```
python -m genfond domains/non-deterministic/acrobatics/{domain.pddl,p*.pddl}
```

There are multiple options, e.g., you can select the type of policy to learn with `--type`, such as the transition-based variant with `--type trans`. See `python -m genfond -h` for a full list of options.

### One-shot solver

The main solver iteratively solves the given problems by incrementally adding problems to the training set and by iteratively increasing the maximal feature complexity. If you instead want to run the solver once for a given feature complexity on a set of problems, pass `--one-shot`, e.g.,:
```
python -m genfond --one-shot --max-complexity 6 domains/non-deterministic/acrobatics/{domain.pddl,p0002*}
```

## Executing a policy

After learning a policy and writing it to a file (with `--output <policyfile>`), you can execute the policy with `execute_policy.py`, e.g.,:
```
python execute_policy.py domains/non-deterministic/acrobatics/{domain.pddl,p0005.pddl} acrobatics.policy
```

# Copyright

All rights reserved by the authors.

Upon publication, this project will be made publicly available and licensed under an open-source license.

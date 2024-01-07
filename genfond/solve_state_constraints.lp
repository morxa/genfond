#include "solve_constraints.lp".

% An action is safe if every possible outcome is alive.
safe_action(I, S1, A) :- state(I, S1), trans(I, S1, A, _), alive(I, S2) : trans(I, S1, A, S2).
% Choose at least one good transition from the safe actions.
1 { good_trans(I, S1, S2) : trans(I, S1, A, S2), safe_action(I, S1, A) } :- alive(I, S1), not goal(I, S1).
% Allow additional good transitions, possibly from non-safe action.
{ good_trans(I, S1, S2) : trans(I, S1, A, S2), alive(I, S2) } :- alive(I, S1), not goal(I, S1).

crit_state(I, S2) :- alive(I, S1), trans(I, S1, _, S2), not alive(I, S2).
:- alive(I1, S1), crit_state(I2, S2), not bool_dist(I1, S1, I2, S2).

#show crit_state/2.

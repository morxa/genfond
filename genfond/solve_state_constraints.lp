#include "solve_constraints.lp".

1 { good_trans(I, S1, A, S2) : trans(I, S1, A, S2), alive(I, S2) } :- alive(I, S1), not goal(I, S1).
good_trans(I, S1, S2) :- good_trans(I, S1, _, S2).

:- good_trans(I, S1, A, _), trans(I, S1, A, S2), not alive(I, S2).
crit_state(I, S2) :- alive(I, S1), trans(I, S1, _, S2), not alive(I, S2).
:- alive(I1, S1), crit_state(I2, S2), not bool_dist(I1, S1, I2, S2).

#show crit_state/2.

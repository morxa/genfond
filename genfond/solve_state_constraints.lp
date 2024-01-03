#include "solve_constraints.lp".

crit_state(I, S2) :- alive(I, S1), trans(I, S1, _, S2), not alive(I, S2).
:- alive(I1, S1), crit_state(I2, S2), bool_equiv(I1, S1, I2, S2).

#show crit_state/2.

#include "solve_constraints.lp". 

% There must be at least one good transition without a bad transition for the same action.
1 { good_trans(I, S1, S2) : trans(I, S1, A, S2), not bad_trans(I, S1, A, _) } :- alive(I, S1), not goal(I, S1).
% Allow additional good actions, possibly with matching bad actions.
{ good_trans(I, S1, S2) : trans(I, S1, A, S2) } :- alive(I, S1), not goal(I, S1).

bad_trans(I, S1, A, S2) :- alive(I, S1), trans(I, S1, A, S2), not alive(I, S2).
bad_trans(I, S1, S2) :- bad_trans(I, S1, _, S2).
:- alive(I1, S11), alive(I2, S21), not bool_dist(I1, S11, I2, S21), bad_trans(I1, S11, S12), trans(I2, S21, _, S22), not bad_trans(I2, S21, S22), not distinguished(I1, S11, S12, I2, S21, S22).

bad_trans_delta(I, S1, S2, F, D) :- bad_trans(I, S1, S2), selected(F), trans_delta(I, S1, S2, F, D).

#show bad_trans/3.
#show bad_trans_delta/5.

% Every consumer of trans_delta requires an alive *source*: distinguished/6 is only used by
% constraints guarded on alive(I1,S11) and alive(I2,S21), good_trans/3 and bad_trans/3 both
% imply it. The target is deliberately left unguarded -- a non-good transition into a dead
% state still has to be separated from a good one. (solve_d2l.lp must not do this: its
% constraint ranges over trans/4 from any state, alive or not.)
trans_delta(I, S1, S2, F, -1) :- alive(I, S1), trans(I, S1, _, S2), eval(I, S1, F, V1), eval(I, S2, F, V2), V2 < V1.
trans_delta(I, S1, S2, F, 0)  :- alive(I, S1), trans(I, S1, _, S2), eval(I, S1, F, V),  eval(I, S2, F, V).
trans_delta(I, S1, S2, F, 1)  :- alive(I, S1), trans(I, S1, _, S2), eval(I, S1, F, V1), eval(I, S2, F, V2), V2 > V1.

good_trans_delta(I, S1, S2, F, D) :- good_trans(I, S1, S2), selected(F), trans_delta(I, S1, S2, F, D).

% There must be a good transition for every alive non-goal state that ends ins an alive state.
:- alive(I, S), not good_trans(I, S, _), not goal(I, S).
:- good_trans(I, _, S), not alive(I, S).

{ selected(F) } :- feature(F).

% Compute boolean evaluations
bool_eval(I, S, F, 1) :- eval(I, S, F, V), V > 0.
bool_eval(I, S, F, 0) :- eval(I, S, F, V), V = 0.

bool_dist(I1, S1, I2, S2) :- state(I1, S1), state(I2, S2), bool_eval(I1, S1, F, V1), bool_eval(I2, S2, F, V2), selected(F), V1 != V2.
% D2
% This used to materialise trans_diff/7 and then project it over selected/1. Deriving
% distinguished/6 in one step grounds the same rule bodies but keeps |T|^2 atoms rather than
% |F|*|T|^2 + |T|^2 -- trans_diff was the single largest predicate in the ground program.
distinguished(I1, S11, S12, I2, S21, S22) :-
    selected(F), trans_delta(I1, S11, S12, F, D1), trans_delta(I2, S21, S22, F, D2), D1 != D2.
:- alive(I1, S11), alive(I2, S21), not bool_dist(I1, S11, I2, S21), good_trans(I1, S11, S12), trans(I2, S21, _, S22), not good_trans(I2, S21, S22), not distinguished(I1, S11, S12, I2, S21, S22).
:- state(I1, S1), state(I2, S2), not bool_dist(I1, S1, I2, S2), goal(I1, S1), not goal(I2, S2).


safe_state(I, S) :- goal(I, S).
safe_state(I, S1) :- alive(I, S1), safe_state(I, S2) : good_trans(I, S1, S2).
:- alive(I, S), not safe_state(I, S).

#minimize { C@3,F : selected(F), feature_complexity(F, C) }.

#show good_trans/3.
#show good_action(I, S1, A) : good_trans(I, S1, S2), trans(I, S1, A, S2).
#show selected/1.
#show bool_eval(I, S, F, V) : bool_eval(I, S, F, V), selected(F).
#show trans_delta(I, S1, S2, F, D) : trans_delta(I, S1, S2, F, D), selected(F).
#show good_trans_delta(I, S1, A, S2, F, D) : good_trans(I, S1, S2), selected(F), trans_delta(I, S1, S2, F, D), trans(I, S1, A, S2).

#program limit_feature_cost(c).
:- #sum { C,F : selected(F), feature_complexity(F, C) } > c.

#program min_feature_complexity(c).
:- C < c : selected(F), feature_complexity(F, C).


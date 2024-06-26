
trans_delta(I, S1, A, S2, F, -1) :- trans(I, S1, A, S2), eval(I, S1, F, V1), eval(I, S2, F, V2), V2 < V1.
trans_delta(I, S1, A, S2, F, 0) :-  trans(I, S1, A, S2), eval(I, S1, F, V), eval(I, S2, F, V).
trans_delta(I, S1, A, S2, F, 1) :-  trans(I, S1, A, S2), eval(I, S1, F, V1), eval(I, S2, F, V2), V2 > V1.
trans_delta(I, S1, S2, F, D) :- trans_delta(I, S1, _, S2, F, D).

good_trans_delta(I, S1, S2, F, D) :- good_trans(I, S1, S2), selected(F), trans_delta(I, S1, S2, F, D).

% Select good actions among the possible transition.
% Select at least one for each alive non-goal state.
1 { good_action(I, S, A) : trans(I, S, A, _) } :- alive(I, S), not goal(I, S).
% Any transition resulting from a good action is good.
good_trans(I, S1, A, S2) :- good_action(I, S1, A), trans(I, S1, A, S2).
good_trans(I, S1, S2) :- good_trans(I, S1, _, S2).

% There must be a good transition for every alive non-goal state that ends ins an alive state.
:- alive(I, S), not good_trans(I, S, _), not goal(I, S).
:- good_trans(I, _, S), not alive(I, S).

{ selected(F) } :- feature(F).

% Compute boolean evaluations
bool_eval(I, S, F, 1) :- eval(I, S, F, V), V > 0.
bool_eval(I, S, F, 0) :- eval(I, S, F, V), V = 0.

bool_dist(I1, S1, I2, S2) :- state(I1, S1), state(I2, S2), bool_eval(I1, S1, F, V1), bool_eval(I2, S2, F, V2), selected(F), V1 != V2.
% For two alive and boolean-equivalent states S11 and S21, if A1 is a good action in S11 but A2 is not good in S21, then there must be a transition that distinguishes the two actions.
:- alive(I1, S11), alive(I2, S21), not bool_dist(I1, S11, I2, S21), good_action(I1, S11, A1), trans(I2, S21,  A2, _), not good_action(I2, S21, A2), trans_delta(I1, S11, A1, S12, F, C) : trans_delta(I2, S21, A2, S22, F, C), selected(F); trans_delta(I2, S21, A2, S22, F, C) : trans_delta(I1, S11, A1, S12, F, C), selected(F).
% Equivalence must distinguish between goal and non-goal states.
:- state(I1, S1), state(I2, S2), not bool_dist(I1, S1, I2, S2), goal(I1, S1), not goal(I2, S2).

safe_state(I, S) :- goal(I, S).
safe_action(I, S, A) :- trans(I, S, A, S2), safe_state(I, S2).
safe_state(I, S) :- alive(I, S), safe_action(I, S, A) : good_action(I, S, A).
:- alive(I, S), not safe_state(I, S).

#minimize { C@3,F : selected(F), feature_complexity(F, C) }.

#show good_trans/3.
#show selected/1.
#show state/2.
#show bool_eval(I, S, F, V) : bool_eval(I, S, F, V), selected(F).
#show trans_delta(I, S1, S2, F, D) : trans_delta(I, S1, A, S2, F, D), selected(F).
#show good_trans_delta/5.

#program limit_feature_cost(c).
:- #sum { C,F : selected(F), feature_complexity(F, C) } > c.

#program min_feature_complexity(c).
:- C < c : selected(F), feature_complexity(F, C).

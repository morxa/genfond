
trans_delta(I, S1, A, S2, F, D) :- trans(I, S1, A, S2), eval(I, S1, F, V1), eval(I, S2, F, V2), D=(-1;0;1), D = -1 : V2 < V1; D = 0 : V2 = V1; D = 1 : V2 > V1.
trans_delta(I, S1, S2, F, D) :- trans_delta(I, S1, _, S2, F, D).

good_trans_delta(I, S1, S2, F, D) :- good_trans(I, S1, S2), selected(F), trans_delta(I, S1, S2, F, D).
%bad_trans_delta(I, S1, S2, F, D) :- bad_trans(I, S1, S2), selected(F), trans_delta(I, S1, S2, F, D).

% Select good actions among the possible transition.
% Select at least one for each alive non-goal state.
1 { good_trans(I, S1, A, S2) : trans(I, S1, A, S2) } :- alive(I, S1), not goal(I, S1).
good_trans(I, S1, S2) :- good_trans(I, S1, _, S2).

% There must be a good transition for every alive non-goal state that ends ins an alive state.
:- alive(I, S), not good_trans(I, S, _), not goal(I, S).
:- good_trans(I, _, S), not alive(I, S).

{ selected(F) } :- feature(F).

% Compute boolean evaluations
bool_eval(I, S, F, 1) :- eval(I, S, F, V), V > 0.
bool_eval(I, S, F, 0) :- eval(I, S, F, V), V = 0.

bool_equiv(I1, S1, I2, S2) :- state(I1, S1), state(I2, S2), bool_eval(I1, S1, F, V) : bool_eval(I2, S2, F, V), selected(F).
% A state is a representative if it is the smallest state in its equivalence class.
%repr(I1, S1) :- state(I1, S1), not equiv(I1, S1, I2, S2) : state(I2, S2), (I2, S2) < (I1, S1).
bool_repr(I1, S1) :- state(I1, S1), not bool_equiv(I1, S1, I2, S2) : state(I2, S2), (I2, S2) < (I1, S1).
% D2
:- alive(I1, S11), alive(I2, S21), bool_equiv(I1, S11, I2, S21), good_trans(I, S11, S12), trans(I, S21, _, S22), not good_trans(I, S21, S22), trans_delta(I, S11, S12, F, D) : trans_delta(I, S21, S22, F, D), selected(F).
% Equivalence must distinguish between goal and non-goal states.
:- bool_equiv(I1, S1, I2, S2), goal(I1, S1), not goal(I2, S2).


safe_state(I, S) :- goal(I, S).
safe_state(I, S1) :- alive(I, S1), safe_state(I, S2) : good_trans(I, S1, S2).
:- alive(I, S), not safe_state(I, S).

%bad_trans(I, S1, A, S2) :- alive(I, S1), trans(I, S1, A, S2), not alive(I, S2).
%bad_trans(I, S1, S2) :- bad_trans(I, S1, _, S2).
%:- good_trans(I, S1, A, _), bad_trans(I, S1, A, _).
%:- alive(I1, S11), alive(I2, S21), bool_equiv(I1, S11, I2, S21), bad_trans(I, S11, S12), trans(I, S21, _, S22), not bad_trans(I, S21, S22), trans_delta(I, S11, S12, F, D) : trans_delta(I, S21, S22, F, D), selected(F).

crit_state(I, S2) :- alive(I, S1), trans(I, S1, _, S2), not alive(I, S2).
:- alive(I1, S1), crit_state(I2, S2), bool_equiv(I1, S1, I2, S2).


#minimize { C@3,F : selected(F), feature_complexity(F, C) }.

#show good_trans/3.
%#show bad_trans/3.
#show selected/1.
#show state/2.
#show crit_state/2.
#show bool_eval(I, S, F, V) : bool_eval(I, S, F, V), selected(F).
#show trans_delta(I, S1, S2, F, D) : trans_delta(I, S1, A, S2, F, D), selected(F).
#show good_trans_delta/5.
%#show bad_trans_delta/5.
%#show rank/3.

#program limit_feature_cost(c).
:- #sum { C,F : selected(F), feature_complexity(F, C) } > c.

#program min_feature_complexity(c).
:- C < c : selected(F), feature_complexity(F, C).


trans_delta(I, S1, S2, F, D) :- trans(I, S1, _, S2), selected(F), eval(I, S1, F, V1), eval(I, S2, F, V2), D=(-1;0;1), D = -1 : V2 < V1; D = 0 : V2 = V1; D = 1 : V2 > V1.

good_trans_delta(I, S1, S2, F, D) :- good_trans(I, S1, S2), trans_delta(I, S1, S2, F, D).

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

% Two states are equivalent if they have the same valuation for all selected features.
equiv(I1, S1, I2, S2) :- state(I1, S1), state(I2, S2), eval(I1, S1, F, V) : eval(I2, S2, F, V), selected(F).
bool_equiv(I1, S1, I2, S2) :- state(I1, S1), state(I2, S2), bool_eval(I1, S1, F, V) : bool_eval(I2, S2, F, V), selected(F).
% A state is a representative if it is the smallest state in its equivalence class.
repr(I1, S1) :- state(I1, S1), not equiv(I1, S1, I2, S2) : state(I2, S2), (I2, S2) < (I1, S1).
bool_repr(I1, S1) :- state(I1, S1), not bool_equiv(I1, S1, I2, S2) : state(I2, S2), (I2, S2) < (I1, S1).
% For two equivalent states, the outgoing good transitions must match and result in equivalent states.
:- equiv(I1, S11, I2, S21), good_trans(I1, S11, S12), not equiv(I1, S12, I2, S22) : good_trans(I2, S21, S22).
% For two boolean-equivalent states, the outgoing good transitions must match and result in equal transition deltas.
:- bool_equiv(I1, S11, I2, S21), good_trans(I1, S11, S12), trans(I2, S21, A, S22), not good_trans(I2, S21, S22), trans_delta(I2, S21, S22, F, C) : trans_delta(I1, S11, S12, F, C).
% Equivalence must distinguish between goal and non-goal states.
:- bool_equiv(I1, S1, I2, S2), goal(I1, S1), not goal(I2, S2).

good_state(I, S) :- goal(I, S).
good_state(I, S1) :- good_state(I, S2), good_trans(I, S1, _, S2).
:- alive(I, S), not good_state(I, S).

#minimize { C@3,F : selected(F), feature_complexity(F, C) }.

#show good_trans/3.
#show selected/1.
#show repr/2.
#show equiv/4.
#show bool_repr/2.
#show bool_eval(I, S, F, V) : repr(I, S), selected(F), bool_eval(I, S, F, V).
#show trans_delta/5.
#show good_trans_delta/5.

% A state is alive it it is a goal or has a path to a goal.
alive(I, S) :- goal(I, S).
alive(I, S2) :- alive(I, S1), trans(I, S2, _, S1).

trans_delta(I, S1, S2, F, D) :- trans(I, S1, _, S2), selected(F), eval(I, S1, F, V1), eval(I, S2, F, V2), D=(-1;0;1), D = -1 : V2 < V1; D = 0 : V2 = V1; D = 1 : V2 > V1.

% Select good transitions among the transitions starting from a representative.
{ good_trans(I, S1, S2) } :- repr(I, S1), trans(I, S1, _, S2).
% Infer the rest of good transitions.
good_trans(I2, S1, S2) :- good_trans(I1, R1, R2), equiv(I1, R1, I2, S1), equiv(I1, R2, I2, S2), trans(I2, S1, _, S2).
good_trans_delta(I, S1, S2, F, D) :- good_trans(I, S1, S2), trans_delta(I, S1, S2, F, D).

% There must be a good transition for every alive non-goal state that ends ins an alive state.
:- alive(I, S), not good_trans(I, S, _), not goal(I, S).
:- good_trans(I, _, S), not alive(I, S).

% For a fixed state, there must be one action that only results in good transitions.
good_action(I, S1, A) :- good_trans(I, S1, S2), trans(I, S1, A, S2), good_trans(I, S1, S3) : trans(I, S1, A, S3).
% There must be at least one good action for each alive non-goal state.
:- alive(I, S), not good_action(I, S, _), not goal(I, S).

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
% For two equivalent states, the outgoing good transitions must match.
:- equiv(I1, S11, I2, S21), good_trans(I1, S11, S12), not equiv(I1, S12, I2, S22) : good_trans(I2, S21, S22).
:- bool_equiv(I1, S11, I2, S21), good_trans(I1, S11, S12), trans(I2, S21, _, S22), not good_trans(I2, S21, S22), trans_delta(I2, S21, S22, F, C) : trans_delta(I1, S11, S12, F, C).
% Equivalence must distinguish between goal and non-goal states.
:- equiv(I1, S1, I2, S2), goal(I1, S1), not goal(I2, S2).


% Count the number of states.
num_states(I, N) :- state(I, _), N = #count { S : state(I, S) }.
max_num_states(N) :- N = #max { M : num_states(_, M) }.

#const delta = 1.
% Assign a rank to each state between 0 and delta*<max_num_states>.
{ rank(I, S, N) :  N=0..delta*M } = 1 :- max_num_states(M), repr(I, S).
% Infer the rank for the other states.
rank(I, S, N) :- equiv(I, S, I2, R), repr(I2, R), rank(I2, R, N).
% Goals must have rank 0.
:- goal(I, S), rank(I, S, N), N > 0.
% For every alive state, there must be one good transition that decreases the rank.
:- state(I, S), alive(I, S), not goal(I, S), rank(I, S, N), N <= N2 : good_trans(I, S, S2), rank(I, S2, N2).

#minimize { C@2,F : selected(F), feature_complexity(F, C) }.
#minimize { 1@1,I,S1,S2 : good_trans(I, S1, S2) }.


#show good_trans/3.
#show selected/1.
#show repr/2.
#show equiv/4.
#show bool_repr/2.
#show bool_eval(I, S, F, V) : bool_repr(I, S), selected(F), bool_eval(I, S, F, V).
#show trans_delta/5.
#show good_trans_delta/5.

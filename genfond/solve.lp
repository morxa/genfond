% A state is alive it it is a goal or has a path to a goal.
alive(I, S) :- goal(I, S).
alive(I, S2) :- alive(I, S1), trans(I, S2, _, S1).

% Select good transitions among the transitions starting from a representative.
{ good_trans(I, S1, S2) } :- repr(I, S1), trans(I, S1, _, S2).
% Infer the rest of good transitions.
good_trans(I2, S1, S2) :- good_trans(I1, R1, R2), equiv(I1, R1, I2, S1), equiv(I1, R2, I2, S2), trans(I2, S1, _, S2).

% There must be a good transition for every alive non-goal state that ends ins an alive state.
:- alive(I, S), not good_trans(I, S, _), not goal(I, S).
:- good_trans(I, _, S), not alive(I, S).

% For a fixed state, there must be one action that only results in good transitions.
good_action(I, S1, A) :- good_trans(I, S1, S2), trans(I, S1, A, S2), good_trans(I, S1, S3) : trans(I, S1, A, S3).
% There must be at least one good action for each alive non-goal state.
:- alive(I, S), not good_action(I, S, _), not goal(I, S).

{ selected(F) } :- feature(F).

% Two states are equivalent if they have the same valuation for all selected features.
equiv(I1, S1, I2, S2) :- state(I1, S1), state(I2, S2), eval(I1, S1, F, V) : eval(I2, S2, F, V), selected(F).
% A state is a representative if it is the smallest state in its equivalence class.
repr(I1, S1) :- state(I1, S1), not equiv(I1, S1, I2, S2) : state(I2, S2), (I2, S2) < (I1, S1).
% For two equivalent states, the outgoing good transitions must match.
:- equiv(I1, S11, I2, S21), good_trans(I1, S11, S12), not equiv(I1, S12, I2, S22) : good_trans(I2, S21, S22).
% Equivalence must distinguish between goal and non-goal states.
:- equiv(I1, S1, I2, S2), goal(I1, S1), not goal(I2, S2).

#minimize { C,F : selected(F), feature_complexity(F, C) }.

#show good_trans/3.
#show selected/1.
#show repr/2.

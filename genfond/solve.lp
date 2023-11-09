% A state is alive it it is a goal or has a path to a goal.
alive(I, S) :- goal(I, S).
alive(I, S2) :- alive(I, S1), trans(I, S2, _, S1).

{ good_trans(I, S1, S2) } :- trans(I, S1, _, S2).

% There must be a good transition for every alive non-goal state that ends ins an alive state.
:- alive(I, S), not good_trans(I, S, _), not goal(I, S).
:- good_trans(I, _, S), not alive(I, S).

% For a fixed state, there must be one action that only results in good transitions.
good_action(I, S1, A) :- good_trans(I, S1, S2), trans(I, S1, A, S2), good_trans(I, S1, S3) : trans(I, S1, A, S3).
% There must be at least one good action for each alive non-goal state.
:- alive(I, S), not good_action(I, S, _), not goal(I, S).

#show good_trans/3.

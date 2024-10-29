bool_eval(I, S, F, 1) :- eval(I, S, F, V), V > 0.
bool_eval(I, S, F, 0) :- eval(I, S, F, V), V = 0.

trans_delta(I, S1, S2, F, -1) :- trans(I, S1, A, S2), eval(I, S1, F, V1), eval(I, S2, F, V2), V2 < V1.
trans_delta(I, S1, S2, F, 0) :-  trans(I, S1, A, S2), eval(I, S1, F, V), eval(I, S2, F, V).
trans_delta(I, S1, S2, F, 1) :-  trans(I, S1, A, S2), eval(I, S1, F, V1), eval(I, S2, F, V2), V2 > V1.

trans_diff(F, I1, S11, S12, I2, S21, S22) :- trans_delta(I1, S11, S12, F, D1), trans_delta(I2, S21, S22, F, D2), D1 != D2.

{ selected(F) } :- feature(F).
1 { good_trans(I, S1, S2) : trans(I, S1, _, S2), alive(I, S2) } :- alive(I, S1), not goal(I, S1).

safe_state(I, S) :- goal(I, S).
safe_state(I, S1) :- alive(I, S1), good_trans(I, S1, _), safe_state(I, S2) : good_trans(I, S1, S2).
:- alive(I, S), not safe_state(I, S).

:- goal(I1, S1), state(I2, S2), not goal(I2, S2), bool_eval(I2, S2, F, V) : bool_eval(I1, S1, F, V), selected(F).
:- good_trans(I1, S11, S12), trans(I2, S21, _, S22), not good_trans(I2, S21, S22),
   bool_eval(I2, S21, F, V) : bool_eval(I1, S11, F, V), selected(F);
   not trans_diff(F, I1, S11, S12, I2, S21, S22) : selected(F).

#minimize { C,F : selected(F), feature_complexity(F, C) }.

#show good_action(I, S1, A) : good_trans(I, S1, S2), trans(I, S1, A, S2).
#show selected/1.
#show bool_eval(I, S, F, V) : bool_eval(I, S, F, V), selected(F).
#show good_trans_delta(I, S1, A, S2, F, D) : good_trans(I, S1, S2), trans_delta(I, S1, S2, F, D), selected(F), trans(I, S1, A, S2).

#program limit_feature_cost(c).
:- #sum { C,F : selected(F), feature_complexity(F, C) } > c.

#program min_feature_complexity(c).
:- C < c : selected(F), feature_complexity(F, C).
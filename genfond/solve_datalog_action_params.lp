%#defined eval/4.

{ selected(F) } :- feature(F).

safe_state(I, S) :- goal(I, S).
safe_state(I, S1) :- alive(I, S1), good_trans(I, S1, _, _), safe_state(I, S2) : good_trans(I, S1, _, S2).
:- alive(I, S), not safe_state(I, S).

1 { good_trans(I, S1, A, S2) : trans(I, S1, A, S2), alive(I, S2) } :- alive(I, S1), not goal(I, S1).
%good_action(I, S, A) :- good_trans(I, S, A, _).

fdiff(I, S, A, P1, P2, F, -1) :- aug_state(I, S, A, P1, T1), aug_state(I, S, A, P2, T2), P1 < P2, aug_eval(T1, F, V1), aug_eval(T2, F, V2), V1 > V2.
fdiff(I, S, A, P1, P2, F,  1) :- aug_state(I, S, A, P1, T1), aug_state(I, S, A, P2, T2), P1 < P2, aug_eval(T1, F, V1), aug_eval(T2, F, V2), V1 < V2.
fdiff(I, S, A, P1, P2, F,  0) :- aug_state(I, S, A, P1, T1), aug_state(I, S, A, P2, T2), P1 < P2, aug_eval(T1, F, V),  aug_eval(T2, F, V).

fdist(I1, S1, I2, S2, F) :- alive(I1, S1), alive(I2, S2), eval(I1, S1, F, V1), eval(I2, S2, F, V2), V1 = 0, V2 != 0.
fdist(I2, S2, I1, S1, F) :- fdist(I1, S1, I2, S2, F).
aug_fdist(I1, S1, A1, P, I2, S2, A2, F) :- alive(I1, S1), alive(I2, S2), aug_state(I1, S1, A1, P, T1), aug_state(I2, S2, A2, P, T2), aug_eval(T1, F, V1), aug_eval(T2, F, V2), V1 = 0, V2 != 0.
aug_fdist(I2, S2, A2, P, I1, S1, A1, F) :- aug_fdist(I1, S1, A1, P, I2, S2, A2, F).
aug_d2(I1, S1, A1, P1, P2, I2, S2, A2, F) :- alive(I1, S1), alive(I2, S2), aname(A1, AName), aname(A2, AName), fdiff(I1, S1, A1, P1, P2, F, D1), fdiff(I2, S2, A2, P1, P2, F, D2), D1 != D2.

:- alive(I1, S1), not goal(I1, S1), good_trans(I1, S1, A1, _),
   alive(I2, S2), not goal(I2, S2), trans(I2, S2, A2, _), not good_trans(I2, S2, A2, _),
   aname(A1, AName), aname(A2, AName),
   not aug_d2(I1, S1, A1, _, _, I2, S2, A2, F) : selected(F);
   not aug_fdist(I1, S1, A1, _, I2, S2, A2, F) : selected(F);
   not fdist(I1, S1, I2, S2, F) : selected(F).
   %not fdist(I2, S2, I1, S1, F) : selected(F).

#show aug_d2(I, S, A, P1, P2, F, D) : alive(I, S), selected(F), aug_d2(I, S, A, P1, P2, _, _, _, F), fdiff(I, S, A, P1, P2, F, D).
#show bool_eval(I, S, F, 0) : alive(I, S), eval(I, S, F, 0), selected(F).
#show bool_eval(I, S, F, 1) : alive(I, S), eval(I, S, F, V), V > 0, selected(F).
#show aug_bool_eval(I, S, A, P, F, 0) : alive(I, S), aug_state(I, S, A, P, T), aug_eval(T, F, 0), selected(F).
#show aug_bool_eval(I, S, A, P, F, 1) : alive(I, S), aug_state(I, S, A, P, T), aug_eval(T, F, V), V > 0, selected(F).
#show param_aug_dist(I1, S1, A1, P, F) : aug_fdist(I1, S1, A1, P, I2, S2, A2, F), selected(F).
#show f_distinguished(I1, S1, I2, S2, F) : fdist(I1, S1, I2, S2, F), selected(F).
%#show f_distinguished(I2, S2, I1, S1, F) : fdist(I1, S1, I2, S2, F), selected(F).
#show good_trans/4.
#show trans/4.
#show safe_state/2.
#show goal/2.
#show good_action(I, S, A) : good_trans(I, S, A, _).

#minimize { C,F : selected(F), feature_complexity(F, C) }.


#program limit_feature_cost(c).
:- #sum { C,F : selected(F), feature_complexity(F, C) } > c.

#program min_feature_complexity(c).
:- C < c : selected(F), feature_complexity(F, C).
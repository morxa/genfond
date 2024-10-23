{ selected(F) } :- feature(F).

%bool_eval(I, S, F, 1) :- eval(I, S, F, V), V > 0.
%bool_eval(I, S, F, 0) :- eval(I, S, F, V), V = 0.
%f_distinguished(I1, S1, I2, S2, F) :- state(I1, S1), state(I2, S2), bool_eval(I1, S1, F, V), not bool_eval(I2, S2, F, V).

aug_bool_eval(T, F, 1) :- eval(T, F, V), V > 0.
aug_bool_eval(T, F, 0) :- eval(T, F, 0).

1 { good_trans(I, S1, A, S2) : trans(I, S1, A, S2), alive(I, S2) } :- alive(I, S1), not goal(I, S1).
good_action(I, S, A) :- good_trans(I, S, A, _).

safe_state(I, S) :- goal(I, S).
safe_state(I, S1) :- alive(I, S1), safe_state(I, S2) : good_trans(I, S1, _, S2).
:- alive(I, S), not safe_state(I, S).

aug_dist(I1, S1, A1, I2, S2, A2, F) :- alive(I1, S1), alive(I2, S2), aug_state(I1, S1, A1, T1), aug_state(I2, S2, A2, T2),
                                       %aname(A1, AName), aname(A2, AName),
                                       aug_bool_eval(T1, F, V), not aug_bool_eval(T2, F, V).

%:- good_trans(I1, S1, A1, _), trans(I2, S2, A2, _), not good_trans(I2, S2, A2, _),
%   aname(A1, AName1), aname(A2, AName2), AName1 != AName2,
%   not f_distinguished(I1, S1, I2, S2, F) : selected(F).
:- good_trans(I1, S1, A1, _), trans(I2, S2, A2, _), not good_trans(I2, S2, A2, _),
   %aname(A1, AName), aname(A2, AName),
   not aug_dist(I1, S1, A1, I2, S2, A2, F) : selected(F).

#minimize { Cost,F : selected(F), feature_complexity(F, Cost) }.

#show state_aug_dist(I1, S1, A1, I2, S2, A2, F) : aug_dist(I1, S1, A1, I2, S2, A2, F), selected(F).
#show good_action/3.
#show state_aug_bool_eval(I, S, A, F, V) : aug_state(I, S, A, T), aug_bool_eval(T, F, V), selected(F).
%#show bool_eval(I, S, F, V) : alive(I, S), bool_eval(I, S, F, V), selected(F).
%#show f_distinguished(I1, S1, I2, S2, F) : f_distinguished(I1, S1, I2, S2, F), selected(F).

#program limit_feature_cost(c).
:- #sum { C,F : selected(F), feature_complexity(F, C) } > c.

#program min_feature_complexity(c).
:- C < c : selected(F), feature_complexity(F, C).
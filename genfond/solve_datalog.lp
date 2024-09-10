#defined c_eval/4.
#defined r_eval/5.

{ c_selected(C) } :- concept(C).
c_selected(name).

{ f_selected(F) } :- feature(F).

{ r_selected(R) } :- role(R).

1 { good_trans(I, S, A, S2) : trans(I, S, A, S2) } :- alive(I, S), not goal(I, S).
:- good_trans(I, S, A, _), trans(I, S, A, S2), not alive(I, S2).
good_action(I, S, A) :- good_trans(I, S, A, _).

safe_state(I, S) :- goal(I, S).
safe_state(I, S1) :- alive(I, S1), safe_state(I, S2) : good_trans(I, S1, _, S2).
:- alive(I, S), not safe_state(I, S).

c_distinguished(I1, S1, A1, I2, S2, A2, name, pos, 0) :- alive(I1, S1), alive(I2, S2), amap(A1, AName1, _, _, _, _), amap(A2, AName2, _, _, _, _), AName1 < AName2.
c_distinguished(I1, S1, A1, I2, S2, A2, C,    pos, 1) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, P1, _, _, _), amap(A2, AName, P2, _, _, _), c_eval(I1, S1, C, P1), not c_eval(I2, S2, C, P2).
c_distinguished(I1, S1, A1, I2, S2, A2, C,    pos, 2) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, _, P1, _, _), amap(A2, AName, _, P2, _, _), c_eval(I1, S1, C, P1), not c_eval(I2, S2, C, P2).
c_distinguished(I1, S1, A1, I2, S2, A2, C,    pos, 3) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, _, _, P1, _), amap(A2, AName, _, _, P2, _), c_eval(I1, S1, C, P1), not c_eval(I2, S2, C, P2).
c_distinguished(I1, S1, A1, I2, S2, A2, C,    pos, 4) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, _, _, _, P1), amap(A2, AName, _, _, _, P2), c_eval(I1, S1, C, P1), not c_eval(I2, S2, C, P2).
c_distinguished(I1, S1, A1, I2, S2, A2, C, neg, N) :- c_distinguished(I2, S2, A2, I1, S1, A1, C, pos, N).

r_distinguished(I1, S1, A1, I2, S2, A2, R,    pos, 1, 2) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, P11, P12, _, _), amap(A2, AName, P21, P22, _, _), r_eval(I1, S1, R, P11, P12), not r_eval(I2, S2, R, P21, P22).
r_distinguished(I1, S1, A1, I2, S2, A2, R,    pos, 1, 3) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, P11, _, P12, _), amap(A2, AName, P21, _, P22, _), r_eval(I1, S1, R, P11, P12), not r_eval(I2, S2, R, P21, P22).
r_distinguished(I1, S1, A1, I2, S2, A2, R,    pos, 1, 4) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, P11, _, _, P12), amap(A2, AName, P21, _, _, P22), r_eval(I1, S1, R, P11, P12), not r_eval(I2, S2, R, P21, P22).
r_distinguished(I1, S1, A1, I2, S2, A2, R,    pos, 2, 3) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, _, P11, P12, _), amap(A2, AName, _, P21, P22, _), r_eval(I1, S1, R, P11, P12), not r_eval(I2, S2, R, P21, P22).
r_distinguished(I1, S1, A1, I2, S2, A2, R,    pos, 2, 4) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, _, P11, _, P12), amap(A2, AName, _, P21, _, P22), r_eval(I1, S1, R, P11, P12), not r_eval(I2, S2, R, P21, P22).
r_distinguished(I1, S1, A1, I2, S2, A2, R,    pos, 3, 4) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, _, _, P11, P12), amap(A2, AName, _, _, P21, P22), r_eval(I1, S1, R, P11, P12), not r_eval(I2, S2, R, P21, P22).

r_distinguished(I1, S1, A1, I2, S2, A2, R,    pos, 2, 1) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, P12, P11, _, _), amap(A2, AName, P22, P21, _, _), r_eval(I1, S1, R, P11, P12), not r_eval(I2, S2, R, P21, P22).
r_distinguished(I1, S1, A1, I2, S2, A2, R,    pos, 3, 1) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, P12, _, P11, _), amap(A2, AName, P22, _, P21, _), r_eval(I1, S1, R, P11, P12), not r_eval(I2, S2, R, P21, P22).
r_distinguished(I1, S1, A1, I2, S2, A2, R,    pos, 3, 2) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, _, P12, P11, _), amap(A2, AName, _, P22, P21, _), r_eval(I1, S1, R, P11, P12), not r_eval(I2, S2, R, P21, P22).
r_distinguished(I1, S1, A1, I2, S2, A2, R,    pos, 4, 1) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, P12, _, _, P11), amap(A2, AName, P22, _, _, P21), r_eval(I1, S1, R, P11, P12), not r_eval(I2, S2, R, P21, P22).
r_distinguished(I1, S1, A1, I2, S2, A2, R,    pos, 4, 2) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, _, P12, _, P11), amap(A2, AName, _, P22, _, P21), r_eval(I1, S1, R, P11, P12), not r_eval(I2, S2, R, P21, P22).
r_distinguished(I1, S1, A1, I2, S2, A2, R,    pos, 4, 3) :- alive(I1, S1), alive(I2, S2), amap(A1, AName, _, _, P12, P11), amap(A2, AName, _, _, P22, P21), r_eval(I1, S1, R, P11, P12), not r_eval(I2, S2, R, P21, P22).

r_distinguished(I1, S1, A1, I2, S2, A2, R, neg, N1, N2) :- r_distinguished(I2, S2, A2, I1, S1, A1, R, pos, N1, N2).


bool_eval(I, S, F, 1) :- eval(I, S, F, V), V > 0.
bool_eval(I, S, F, 0) :- eval(I, S, F, V), V = 0.
f_distinguished(I1, S1, I2, S2, F) :- state(I1, S1), state(I2, S2), bool_eval(I1, S1, F, V), not bool_eval(I2, S2, F, V).

% TODO test whether using sets is cheaper { c_distinguished(I1, S1, A1, I2, S2, A2, C, _, _) : c_selected(C) } = 0.
:- good_action(I1, S1, A1), alive(I2, S2), not goal(I2, S2), trans(I2, S2, A2, _), not good_action(I2, S2, A2),  not c_distinguished(I1, S1, A1, I2, S2, A2, C, _, _) : c_selected(C); not f_distinguished(I1, S1, I2, S2, F) : f_selected(F);
not r_distinguished(I1, S1, A1, I2, S2, A2, R, _, _, _) : r_selected(R).

#show f_distinguished(I1, S1, I2, S2, F) : f_distinguished(I1, S1, I2, S2, F), f_selected(F).
#show c_distinguished(I1, S1, A1, I2, S2, A2, C, D, N) : c_distinguished(I1, S1, A1, I2, S2, A2, C, D, N), c_selected(C), alive(I1, S1), amap(A1, AName, _, _, _, _), amap(A2, AName, _, _, _, _), good_action(I1, S1, A1), trans(I2, S2, A2, _), not good_action(I2, S2, A2), not f_distinguished(I1, S1, I2, S2, F) : f_selected(F).
#show r_distinguished(I1, S1, A1, I2, S2, A2, R, D, N1, N2) : r_distinguished(I1, S1, A1, I2, S2, A2, R, D, N1, N2), r_selected(R), alive(I1, S1), amap(A1, AName, _, _, _, _), amap(A2, AName, _, _, _, _), good_action(I1, S1, A1), trans(I2, S2, A2, _), not good_action(I2, S2, A2), not f_distinguished(I1, S1, I2, S2, F) : f_selected(F); not c_distinguished(I1, S1, A1, I2, S2, A2, C, _, _) : c_selected(C).
#show good_action/3.
#show c_selected/1.
#show f_selected/1.
#show bool_eval/4.

#minimize { Cost,C : c_selected(C), concept_complexity(C, Cost) }.
#minimize { Cost,F : f_selected(F), feature_complexity(F, Cost) }.
#minimize { Cost,R : r_selected(R), role_complexity(R, Cost) }.

#program limit_feature_cost(c).
:- #sum { C,F : f_selected(F), feature_complexity(F, C); Cost,Concept : c_selected(Concept), concept_complexity(Concept, Cost); Cost,Role : r_selected(Role), role_complexity(Role, Cost) } > c.

#program min_feature_complexity(c).
:- C < c : f_selected(F), feature_complexity(F, C); Cost < c : c_selected(Concept), concept_complexity(Concept, Cost); Cost < c : r_selected(Role), role_complexity(Role, Cost).

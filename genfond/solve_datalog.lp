#defined c_eval/4.
#defined role/1.
#defined role_complexity/2.
#defined r_eval/5.
#defined amap/2.
#defined amap/3.
#defined amap/4.
#defined amap/5.
#defined amap/6.

{ c_selected(C) } :- concept(C).
c_selected(name).

{ f_selected(F) } :- feature(F).

{ r_selected(R) } :- role(R).

1 { good_trans(I, S, A, S2) : trans(I, S, A, S2) } :- alive(I, S), not goal(I, S).
:- good_trans(I, S, A, _), trans(I, S, A, S2), not alive(I, S2), not pruned(I, S2).
good_action(I, S, A) :- good_trans(I, S, A, _).

safe_state(I, S) :- goal(I, S).
safe_state(I, S) :- pruned(I, S).
safe_state(I, S1) :- alive(I, S1), good_trans(I, S1, _, _), safe_state(I, S2) : good_trans(I, S1, _, S2).
:- alive(I, S), not safe_state(I, S).


c_distinguished(I1, S1, A1, I2, S2, A2, C) :-
    alive(I1, S1), alive(I2, S2),
    aname(A1, AName), aname(A2, AName), aparam(A1, N, P1), aparam(A2, N, P2),
    c_eval(I1, S1, C, P1), not c_eval(I2, S2, C, P2).
r_distinguished(I1, S1, A1, I2, S2, A2, R) :-
    alive(I1, S1), alive(I2, S2),
    aname(A1, AName), aname(A2, AName),
    aparam(A1, N1, P11), aparam(A1, N2, P12), aparam(A2, N1, P21), aparam(A2, N2, P22),
    r_eval(I1, S1, R, P11, P12), not r_eval(I2, S2, R, P21, P22).

bool_eval(I, S, F, 1) :- eval(I, S, F, V), V > 0.
bool_eval(I, S, F, 0) :- eval(I, S, F, V), V = 0.
f_distinguished(I1, S1, I2, S2, F) :- state(I1, S1), state(I2, S2), bool_eval(I1, S1, F, V), not bool_eval(I2, S2, F, V).

% TODO test whether using sets is cheaper { c_distinguished(I1, S1, A1, I2, S2, A2, C, _, _) : c_selected(C) } = 0.
:- good_action(I1, S1, A1), alive(I2, S2), not goal(I2, S2), trans(I2, S2, A2, _), not good_action(I2, S2, A2),
   aname(A1, AName), aname(A2, AName),
   not c_distinguished(I1, S1, A1, I2, S2, A2, C) : c_selected(C); not c_distinguished(I2, S2, A2, I1, S1, A1, C) : c_selected(C);
   not f_distinguished(I1, S1, I2, S2, F) : f_selected(F);
   not r_distinguished(I1, S1, A1, I2, S2, A2, R) : r_selected(R); not r_distinguished(I2, S2, A2, I1, S1, A1, R) : r_selected(R).

#show f_distinguished(I1, S1, I2, S2, F) : f_distinguished(I1, S1, I2, S2, F), f_selected(F).
#show c_distinguished(I1, S1, A1, I2, S2, A2, C, pos, N) :
    c_distinguished(I1, S1, A1, I2, S2, A2, C), c_selected(C),
    aparam(A1, N, P1), c_eval(I1, S1, C, P1),
    aparam(A2, N, P2), not c_eval(I2, S2, C, P2),
    good_action(I1, S1, A1), trans(I2, S2, A2, _), not good_action(I2, S2, A2),
    not f_distinguished(I1, S1, I2, S2, F) : f_selected(F).
#show c_distinguished(I1, S1, A1, I2, S2, A2, C, neg, N) :
    c_distinguished(I2, S2, A2, I1, S1, A1, C), c_selected(C),
    aparam(A1, N, P1), not c_eval(I1, S1, C, P1),
    aparam(A2, N, P2), c_eval(I2, S2, C, P2),
    good_action(I1, S1, A1), trans(I2, S2, A2, _), not good_action(I2, S2, A2),
    not f_distinguished(I1, S1, I2, S2, F) : f_selected(F).
#show r_distinguished(I1, S1, A1, I2, S2, A2, R, pos, N1, N2) :
    r_distinguished(I1, S1, A1, I2, S2, A2, R), r_selected(R),
    aparam(A1, N1, P11), aparam(A1, N2, P12), r_eval(I1, S1, R, P11, P12),
    aparam(A2, N1, P21), aparam(A2, N2, P22), not r_eval(I2, S2, R, P21, P22),
    good_action(I1, S1, A1), trans(I2, S2, A2, _), not good_action(I2, S2, A2),
    not f_distinguished(I1, S1, I2, S2, F) : f_selected(F);
    not c_distinguished(I1, S1, A1, I2, S2, A2, C) : c_selected(C);
    not c_distinguished(I2, S2, A2, I1, S1, A1, C) : c_selected(C).
#show r_distinguished(I1, S1, A1, I2, S2, A2, R, neg, N1, N2) :
    r_distinguished(I2, S2, A2, I1, S1, A1, R), r_selected(R),
    aparam(A1, N1, P11), aparam(A1, N2, P12), not r_eval(I1, S1, R, P11, P12),
    aparam(A2, N1, P21), aparam(A2, N2, P22), r_eval(I2, S2, R, P21, P22),
    good_action(I1, S1, A1), trans(I2, S2, A2, _), not good_action(I2, S2, A2),
    not f_distinguished(I1, S1, I2, S2, F) : f_selected(F);
    not c_distinguished(I1, S1, A1, I2, S2, A2, C) : c_selected(C);
    not c_distinguished(I2, S2, A2, I1, S1, A1, C) : c_selected(C).
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

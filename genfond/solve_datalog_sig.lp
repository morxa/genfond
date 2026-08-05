% Datalog policies with the separation layer quotiented by action signature.
%
% Semantically identical to solve_datalog.lp. The difference is that the pairwise separation
% constraint ranges over signature classes (asig/4, emitted by feature_generator) instead of
% over (instance, state, action) triples. The constraint's body observes a triple only through
% the action name, the concept membership of each argument position, the role membership of
% each ordered position pair, and the source state's boolean feature vector -- exactly what a
% signature records -- so two triples with the same signature can never be told apart by any
% selection, and goodness is forced to be a function of the signature either way.
%
% The graph layer (state/trans/alive/goal/pruned/good_trans/safe_state/frontier) is unchanged,
% so frontier expansion works exactly as before.

#defined c_eval/4.
#defined role/1.
#defined role_complexity/2.
#defined concept/1.
#defined concept_complexity/2.
#defined r_eval/5.
#defined pruned/2.
#defined asig/4.
#defined sig_aname/2.
#defined sig_pos/2.
#defined sig_c_eval/3.
#defined sig_r_eval/4.
#defined sig_bool_eval/3.

{ c_selected(C) } :- concept(C).
c_selected(name).

{ f_selected(F) } :- feature(F).

{ r_selected(R) } :- role(R).

1 { good_trans(I, S, A, S2) : trans(I, S, A, S2) } :- alive(I, S), not goal(I, S).
:- good_trans(I, S, A, _), trans(I, S, A, S2), not alive(I, S2), not pruned(I, S2).
good_action(I, S, A) :- good_trans(I, S, A, _).

frontier(I, S) :- good_trans(I, _, _, S), pruned(I, S).

safe_state(I, S) :- goal(I, S).
safe_state(I, S) :- pruned(I, S).
safe_state(I, S1) :- alive(I, S1), good_trans(I, S1, _, _), safe_state(I, S2) : good_trans(I, S1, _, S2).
:- alive(I, S), not safe_state(I, S).

% ---------------------------------------------------------------- separation layer, quotiented

good_sig(K) :- good_action(I, S, A), asig(I, S, A, K).
bad_sig(K) :- alive(I, S), not goal(I, S), trans(I, S, A, _), not good_action(I, S, A), asig(I, S, A, K).

sig_c_dist(K1, K2, C) :-
    sig_aname(K1, AName), sig_aname(K2, AName),
    sig_c_eval(K1, C, N), sig_pos(K2, N), not sig_c_eval(K2, C, N).
sig_r_dist(K1, K2, R) :-
    sig_aname(K1, AName), sig_aname(K2, AName),
    sig_r_eval(K1, R, N1, N2), sig_pos(K2, N1), sig_pos(K2, N2), not sig_r_eval(K2, R, N1, N2).
% Both sides carry a total boolean vector, so this needs no aliveness guard: a signature only
% exists for a state that emits actions, which is exactly an alive non-goal state.
sig_f_dist(K1, K2, F) :- sig_bool_eval(K1, F, V1), sig_bool_eval(K2, F, V2), V1 != V2.

% K1 = K2 derives none of the three, so a signature can never be both good and bad.
:- good_sig(K1), bad_sig(K2), sig_aname(K1, AName), sig_aname(K2, AName),
   not sig_c_dist(K1, K2, C) : c_selected(C); not sig_c_dist(K2, K1, C) : c_selected(C);
   not sig_f_dist(K1, K2, F) : f_selected(F);
   not sig_r_dist(K1, K2, R) : r_selected(R); not sig_r_dist(K2, K1, R) : r_selected(R).

% ------------------------------------------------------------------------- policy extraction

#show sig_action(K, A) : good_action(I, S, A), asig(I, S, A, K).
#show sig_f_dist(K1, K2, F) : sig_f_dist(K1, K2, F), f_selected(F), good_sig(K1).
#show sig_bool_eval(K, F, V) : sig_bool_eval(K, F, V), f_selected(F), good_sig(K).
#show sig_c_dist(K1, K2, C, pos, N) :
    sig_c_dist(K1, K2, C), c_selected(C),
    sig_c_eval(K1, C, N), sig_pos(K2, N), not sig_c_eval(K2, C, N),
    good_sig(K1), bad_sig(K2),
    not sig_f_dist(K1, K2, F) : f_selected(F).
#show sig_c_dist(K1, K2, C, neg, N) :
    sig_c_dist(K2, K1, C), c_selected(C),
    sig_pos(K1, N), not sig_c_eval(K1, C, N), sig_c_eval(K2, C, N),
    good_sig(K1), bad_sig(K2),
    not sig_f_dist(K1, K2, F) : f_selected(F).
#show sig_r_dist(K1, K2, R, pos, N1, N2) :
    sig_r_dist(K1, K2, R), r_selected(R),
    sig_r_eval(K1, R, N1, N2), sig_pos(K2, N1), sig_pos(K2, N2), not sig_r_eval(K2, R, N1, N2),
    good_sig(K1), bad_sig(K2),
    not sig_f_dist(K1, K2, F) : f_selected(F);
    not sig_c_dist(K1, K2, C) : c_selected(C);
    not sig_c_dist(K2, K1, C) : c_selected(C).
#show sig_r_dist(K1, K2, R, neg, N1, N2) :
    sig_r_dist(K2, K1, R), r_selected(R),
    sig_pos(K1, N1), sig_pos(K1, N2), not sig_r_eval(K1, R, N1, N2), sig_r_eval(K2, R, N1, N2),
    good_sig(K1), bad_sig(K2),
    not sig_f_dist(K1, K2, F) : f_selected(F);
    not sig_c_dist(K1, K2, C) : c_selected(C);
    not sig_c_dist(K2, K1, C) : c_selected(C).
#show good_action/3.
#show frontier/2.
#show c_selected/1.
#show f_selected/1.
#show bool_eval/4.

bool_eval(I, S, F, 1) :- eval(I, S, F, V), V > 0.
bool_eval(I, S, F, 0) :- eval(I, S, F, V), V = 0.

#minimize { Cost,C : c_selected(C), concept_complexity(C, Cost) }.
#minimize { Cost,F : f_selected(F), feature_complexity(F, Cost) }.
#minimize { Cost,R : r_selected(R), role_complexity(R, Cost) }.

#minimize { 1@2,I,S1,A,S2 : good_trans(I, S1, A, S2), pruned(I, S2) }.

#program limit_feature_cost(c).
:- #sum { C,F : f_selected(F), feature_complexity(F, C); Cost,Concept : c_selected(Concept), concept_complexity(Concept, Cost); Cost,Role : r_selected(Role), role_complexity(Role, Cost) } > c.

#program limit_prune_cost(c).
:- #count { I,S1,A,S2 : good_trans(I, S1, A, S2), pruned(I, S2) } > c.

#program min_feature_complexity(c).
:- C < c : f_selected(F), feature_complexity(F, C); Cost < c : c_selected(Concept), concept_complexity(Concept, Cost); Cost < c : r_selected(Role), role_complexity(Role, Cost).

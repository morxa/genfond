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
% Which elements distinguish a given pair of classes is a property of the instance, not of the
% model, so it is not derived here: feature_generator emits sig_pair(K1, K2, D) once per
% unordered pair of classes with the same action name, and dist(D, E) once per element of each
% *distinct* distinguishing set D. Separation then becomes a hitting-set condition whose
% grounding is linear in the emitted facts, instead of |K|^2 * (|F| + |C|*a + |R|*a^2) rules.
% E is f("..."), c("...") or r("..."), matching f_selected/c_selected/r_selected.
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
#defined sig_pair/3.
#defined dist/2.

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

sig_hit(D) :- dist(D, f(F)), f_selected(F).
sig_hit(D) :- dist(D, c(C)), c_selected(C).
sig_hit(D) :- dist(D, r(R)), r_selected(R).

% A class is never distinguished from itself, so it can never be both good and bad. sig_pair/3
% only covers K1 != K2, hence the separate constraint.
:- good_sig(K), bad_sig(K).
:- good_sig(K1), bad_sig(K2), sig_pair(K1, K2, D), not sig_hit(D).
:- good_sig(K2), bad_sig(K1), sig_pair(K1, K2, D), not sig_hit(D).

% ------------------------------------------------------------------------- policy extraction

% The conditions of a rule are reconstructed in Python from the selected elements and the good
% and bad classes (see generate_datalog_policy), which needs nothing quadratic from the model.
#show sig_action(K, A) : good_action(I, S, A), asig(I, S, A, K).
#show bad_sig/1.
#show good_action/3.
#show frontier/2.
#show c_selected/1.
#show f_selected/1.
#show r_selected/1.
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

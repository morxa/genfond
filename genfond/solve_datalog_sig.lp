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
% With lazy_pairs the same relation arrives one batch at a time, tagged with the batch index.
#defined sig_pair/4.
#defined dist/3.
% Labels the graph layer fixes before search (fix_forced_labels; see forced_labels.py) and the
% on-plan actions of the optional decision heuristic (plan_label_heuristic). Both families are
% empty unless the config asks for them, and an empty #defined predicate is simplified away by
% the grounder, so the default ground program is the one this file produced before.
#defined forced_good/3.
#defined forced_bad/3.
#defined plan_action/3.
% The transitions the best policy so far took on the problems it solved (anchor_policy_labels;
% see feature_generator._emit_anchors). Facts only -- the constraint that reads them lives in
% the `anchor` #program part at the bottom, so the very same instance can be re-solved without
% them when the anchored attempt is unsatisfiable.
#defined anchor/3.

{ c_selected(C) } :- concept(C).
c_selected(name).

{ f_selected(F) } :- feature(F).

{ r_selected(R) } :- role(R).

% forced_bad(I,S,A) holds exactly for the actions that are bad in *every* model, so leaving them
% out of the choice removes choice points, not models. Most of them are already excluded by the
% constraint on the next line; the rest come from the cross-state propagation through the
% signature classes (see forced_labels.py), which the grounder cannot see.
1 { good_trans(I, S, A, S2) : trans(I, S, A, S2), not forced_bad(I, S, A) } :- alive(I, S), not goal(I, S).
:- good_trans(I, S, A, _), trans(I, S, A, S2), not alive(I, S2), not pruned(I, S2).
good_action(I, S, A) :- good_trans(I, S, A, _).

% The mirror image: forced_good(I,S,A) holds exactly where good_action does in every model. This
% is stated as a constraint rather than as a rule deriving good_action, because making
% good_action true without a supporting good_trans would *add* models -- the good class would no
% longer oblige the policy to actually take the action, and the obligations that come with taking
% it (all outcomes safe, safe_state) would be dropped.
:- forced_good(I, S, A), not good_action(I, S, A).

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

#minimize { 1@3,I,S1,A,S2 : good_trans(I, S1, A, S2), pruned(I, S2) }.

% -------------------------------------------------- Occam biases over good signatures / arity
%
% Two optional, independent Occam biases (see minimize_good_signatures and
% minimize_selected_count in config/default.yaml), each grounded only when Solver is asked for
% it, so the default "none"/"none" setting reproduces the previous program byte for byte.
%
% minimize_good_signatures minimizes the number of good signature classes, i.e. the number of
% rules the policy ends up with, since a rule is the projection of a good signature onto the
% selected elements.
%
% minimize_selected_count minimizes the number of *selected elements* (concepts, features and
% roles together -- one #minimize statement per kind, all three at the same priority so they
% form a single level) instead of their total complexity. A hand-picked pool that generalizes on
% blocks3ops-heldout selects few, general elements; the plain complexity-cost objective instead
% tends to prefer several cheap, narrow elements over one more expensive general one, and every
% extra selected element grows the number of distinguishable object classes -- and hence rules
% -- exponentially. See docs/min-count-results.md.
%
% Priority layout (descending in model.cost, see solver.py's cost-vector gotcha and
% cost_utils.feature_cost() below): the frontier-transition count is the highest priority (@3)
% regardless of which of these are active, so a frontier model is never preferred just to save a
% rule or an element. The two "above" variants (@2, @1) both sit strictly between the
% feature/concept/role complexity minimize (@0, unspecified priority) and the frontier count, so
% complexity stays the lowest priority *present* and remains model.cost[-1] whenever neither
% "below" variant is active. The two "below" variants (@-1, @-2) sit under complexity, so each
% one active pushes complexity one further position from the end of model.cost -- callers must
% go through cost_utils.feature_cost() (which computes the offset from the config), never index
% cost[-1] directly, whenever either "below" setting may be active.
#program minimize_good_sigs_below.
#minimize { 1@-1,K : good_sig(K) }.

#program minimize_good_sigs_above.
#minimize { 1@2,K : good_sig(K) }.

#program minimize_selected_count_below.
#minimize { 1@-2,E : c_selected(E) }.
#minimize { 1@-2,E : f_selected(E) }.
#minimize { 1@-2,E : r_selected(E) }.

#program minimize_selected_count_above.
#minimize { 1@1,E : c_selected(E) }.
#minimize { 1@1,E : f_selected(E) }.
#minimize { 1@1,E : r_selected(E) }.

#program limit_feature_cost(c).
:- #sum { C,F : f_selected(F), feature_complexity(F, C); Cost,Concept : c_selected(Concept), concept_complexity(Concept, Cost); Cost,Role : r_selected(Role), role_complexity(Role, Cost) } > c.

#program limit_prune_cost(c).
:- #count { I,S1,A,S2 : good_trans(I, S1, A, S2), pruned(I, S2) } > c.

#program min_feature_complexity(c).
:- C < c : f_selected(F), feature_complexity(F, C); Cost < c : c_selected(Concept), concept_complexity(Concept, Cost); Cost < c : r_selected(Role), role_complexity(Role, Cost).

% ------------------------------------------------------- separation layer, added lazily

% With lazy_pairs the instance carries no sig_pair/dist facts at all: the pairs are added one
% batch at a time, in response to models of the relaxation that violate them (see lazy_pairs.py
% and action_signatures.LazyPairs). The batch index k tags the facts so that grounding this part
% for a new batch does not re-ground the rules over the facts of the earlier ones. sig_hit/1 is
% a plain atom, so a batch may refer to a distinguishing set whose dist/3 facts -- and hence the
% rule deriving sig_hit for it -- were grounded in an earlier batch.
#program pairs(k).
sig_hit(D) :- dist(k, D, f(F)), f_selected(F).
sig_hit(D) :- dist(k, D, c(C)), c_selected(C).
sig_hit(D) :- dist(k, D, r(R)), r_selected(R).
:- good_sig(K1), bad_sig(K2), sig_pair(k, K1, K2, D), not sig_hit(D).
:- good_sig(K2), bad_sig(K1), sig_pair(k, K1, K2, D), not sig_hit(D).

% --------------------------------------------------------------- plan-based decision heuristic
%
% plan_label_heuristic: prefer to label a transition that lies on an example plan good. Purely a
% search-order hint -- #heuristic changes no model and no cost -- and it needs clingo's
% --heuristic=Domain, which Solver adds along with this part.
#program plan_heuristic.
#heuristic good_trans(I, S, A, S2) : trans(I, S, A, S2), plan_action(I, S, A). [1,true]

% ------------------------------------------------------------------- policy-anchored labels
%
% anchor_policy_labels: the action the best policy so far took at this state must be labelled
% good, so the round's model agrees with that policy wherever it already worked and the
% #minimize only decides how to cover what it did not. Stated exactly like forced_good/3 --
% as a constraint on good_action rather than as a rule deriving it, so the obligations that
% come with taking an action (all outcomes safe, safe_state) are not dropped.
%
% Grounded only when Solver is asked for it: the anchored solve gets this part, and the
% fallback re-solve of the same round does not (see iterative_solver.solve). An anchor can be
% contradictory -- with the signature quotient, anchoring one occurrence forces its whole class
% good, and that class may have to be bad at some other state -- so the fallback is not an
% optimisation but the thing that keeps an anchored round from ever losing a policy.
#program anchor.
:- anchor(I, S, A), not good_action(I, S, A).

from genfond.solver import Solver
from genfond.feature_generator import FeaturePool
from genfond.generate_policy import generate_policy, PolicyType
import pddl
import pygraphviz
import pickle
import sys
import os.path

import argparse
import logging
from genfond.state_space_generator import Alive

log = logging.getLogger(__name__)


def _get_repr(reprs, equiv, i, s):
    for i1, s1, i2, s2 in equiv:
        if i1 == i and s1 == s and (i2, s2) in reprs:
            return i2, s2
    raise ValueError(f'No repr found for {i}, {s}')


def _state_to_str(node):
    state = node.state
    s = f'{node.id}: '
    return s + ','.join([f'{p.name}({",".join([str(p) for p in p.terms])})' for p in sorted(state)])


def draw_state_graph(state_graph, filename):
    graph = pygraphviz.AGraph(directed=True)
    graph.node_attr['shape'] = 'box'
    for node in state_graph.nodes.values():
        graph.add_node(node.id, label=_state_to_str(node), color='green' if node.alive == Alive.ALIVE else 'red')
        for action, children in node.children.items():
            action_str = f'{action.name}({",".join([str(p) for p in action.parameters])})'
            for child in children:
                graph.add_edge(node.id, child.id, label=action_str)
    graph.layout(prog='dot')
    graph.draw(filename)


def draw_graph(feature_gen, solution, filename):
    graph = pygraphviz.AGraph(directed=True)
    graph.node_attr['shape'] = 'record'
    equivnodes = dict()
    for problem, state_graph in feature_gen.state_graphs.items():
        instance = feature_gen.problem_name_to_id[problem]
        for node in state_graph.nodes.values():
            for i1, s1, i2, s2 in solution['equiv']:
                if (i1, s1) in solution['repr']:
                    if instance == i2 and node.id == s2:
                        equivnodes.setdefault((i1, s1), []).append(
                            (instance, _state_to_str(node.state | feature_gen.goal_states[problem])))
                        break
    for equiv, nodes in equivnodes.items():
        graph.add_node(equiv, label=f'{{{"|".join([str(node) for _, node in nodes])}}}')
    for i, s1, s2 in solution['good_trans']:
        if (i, s1) in solution['repr'] or (i, s2) in solution['repr']:
            graph.add_edge(_get_repr(solution['repr'], solution['equiv'], i, s1),
                           _get_repr(solution['repr'], solution['equiv'], i, s2))

    graph.layout(prog='dot')
    graph.draw(filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('domain_file')
    parser.add_argument('problem_file', nargs='*')
    parser.add_argument('--program-file', '-p', help='Output file for the generated program')
    parser.add_argument('--dump', help='Output for a pickle dump of the solution')
    parser.add_argument('--output', '-o', help='Output file for the resulting policy (as pickle dump)')
    parser.add_argument('--input', '-i', help='Input for a pickle dump of the solution')
    parser.add_argument('--draw', '-d', help='Output path for the resulting state graph')
    parser.add_argument('--draw-input', help='Output path for drawing the input state graph')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-c', '--complexity', type=int, default=5, help='max complexity of the used features')
    parser.add_argument('--min-complexity', type=int, help='force minimum complexity of the used features')
    parser.add_argument('--max-cost', type=int, help='max cost of the policy')
    parser.add_argument('--type',
                        choices=['none', 'state', 'trans', 'datalog'],
                        default='state',
                        help='the type of the generated policy')
    parser.add_argument('-n',
                        '--num-threads',
                        type=int,
                        default=None,
                        help='number of threads to use; "None" uses all available threads')
    parser.add_argument('--no-solve', action='store_true', help='only generate the program, do not solve it')
    parser.add_argument('--restrict-features', nargs='+', help='restrict features to the given ones')
    args = parser.parse_args()
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logging.basicConfig(level=loglevel)
    if args.type == 'state':
        solve_prog = 'solve_state_type.lp'
        policy_type = PolicyType.CONSTRAINED
    elif args.type == 'trans':
        solve_prog = 'solve_trans_type.lp'
        policy_type = PolicyType.CONSTRAINED
    elif args.type == 'none':
        solve_prog = 'solve.lp'
        policy_type = PolicyType.EXACT
    elif args.type == 'datalog':
        solve_prog = 'solve_datalog.lp'
        policy_type = PolicyType.DATALOG
    else:
        raise ValueError(f'Unknown constraint type {args.type}')
    domain = pddl.parse_domain(args.domain_file)
    problems = []
    for problem_file in args.problem_file:
        problems.append(pddl.parse_problem(problem_file))
    feature_pool = FeaturePool(domain,
                               problems,
                               args.complexity,
                               preset_features=args.restrict_features,
                               include_boolean_features=True,
                               include_numerical_features=(args.type != 'datalog'),
                               include_concepts=(args.type == 'datalog'),
                               include_roles=False)
    if args.draw_input:
        for p, g in feature_pool.state_graphs.items():
            base, suffix = os.path.splitext(args.draw_input)
            draw_state_graph(g, f'{base}_{p}{suffix}')
    if args.input:
        solution, policy = pickle.load(open(args.input, 'rb'))
    else:
        asp_instance = feature_pool.to_clingo()
        if args.program_file:
            with open(args.program_file, 'w') as f:
                f.write(asp_instance)
        if args.no_solve:
            sys.exit(0)
        log.info(f'Starting solver for domain {domain.name} and problems {", ".join([p.name for p in problems])}')
        solver = Solver(asp_instance,
                        args.num_threads,
                        solve_prog=solve_prog,
                        max_cost=args.max_cost,
                        min_feature_complexity=args.min_complexity)
        solver.solve()
        solution = solver.solution
        if args.dump:
            pickle.dump(solution, open(args.dump, 'wb'))
            log.info(f'Saved (raw) solution to {args.dump}')
    if not solution:
        log.error('No solution found')
        sys.exit(1)
    policy = generate_policy(solution, policy_type=policy_type)
    log.debug(f'Solution:\n{solution}')
    log.info(f'Policy:\n{policy}')
    if args.output:
        pickle.dump(policy, open(args.output, 'wb'))
    if args.draw:
        draw_graph(feature_pool, solution, args.draw)


if __name__ == '__main__':
    main()

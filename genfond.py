from genfond.solver import Solver
from genfond.feature_generator import FeaturePool
import pddl
import pygraphviz
import pickle
import sys

import argparse
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _get_repr(reprs, equiv, i, s):
    for i1, s1, i2, s2 in equiv:
        if i1 == i and s1 == s and (i2, s2) in reprs:
            return i2, s2
    raise ValueError(f'No repr found for {i}, {s}')


def _state_to_str(state):
    return ','.join([f'{p.name}({",".join([str(p) for p in p.terms])})' for p in state])


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
    parser.add_argument('--output', '-o', help='Output file for the resulting policy (as text)')
    parser.add_argument('--input', '-i', help='Input for a pickle dump of the solution')
    parser.add_argument('--draw', '-d', help='Output path for the resulting state graph')
    args = parser.parse_args()
    domain = pddl.parse_domain(args.domain_file)
    problems = []
    for problem_file in args.problem_file:
        problems.append(pddl.parse_problem(problem_file))
    feature_pool = FeaturePool(domain, problems, 5)
    if not args.input:
        asp_instance = feature_pool.to_clingo()
        if args.program_file:
            with open(args.program_file, 'w') as f:
                f.write(asp_instance)
        solver = Solver(asp_instance)
        solver.solve()
        solution = solver.solution
        if args.output:
            pickle.dump(solution, open(args.output, 'wb'))
    else:
        solution = pickle.load(open(args.input, 'rb'))
    if not solution:
        log.error('No solution found')
        sys.exit(1)
    log.debug(f'Solution:\n{solution}')
    rules = set()
    for instance, state in solution['bool_repr']:
        conds = set()
        for i, s, f, v in solution['bool_eval']:
            if i == instance and s == state:
                if v == 1:
                    if f.startswith('b_'):
                        conds.add(f)
                    elif f.startswith('n_'):
                        conds.add(f'{f} > 0')
                    else:
                        raise ValueError(f'Unknown feature type {f}')
                elif v == 0:
                    if f.startswith('b_'):
                        conds.add(f'¬{f}')
                    elif f.startswith('n_'):
                        conds.add(f'{f} = 0')
                    else:
                        raise ValueError(f'Unknown feature type {f}')
                else:
                    raise ValueError(f'Unknown value {v}')
        effs = dict()
        for i, s1, s2, f, v in solution['good_trans_delta']:
            if i == instance and s1 == state:
                if v == 1:
                    if f.startswith('b_'):
                        eff = f
                    elif f.startswith('n_'):
                        eff = f'↑{f}'
                    else:
                        raise ValueError(f'Unknown feature type {f}')
                elif v == -1:
                    if f.startswith('b_'):
                        eff = f'¬{f}'
                    elif f.startswith('n_'):
                        eff = f'↓{f}'
                    else:
                        raise ValueError(f'Unknown feature type {f}')
                elif v == 0:
                    continue
                else:
                    raise ValueError(f'Unknown value {v}')
                effs.setdefault(s2, set()).add(eff)
        if effs:
            cond = ' ∧ '.join(conds)
            eff_string = ' ; '.join({' ∧ '.join(eff) for _, eff in effs.items()})
            rules.add(f'{{ {cond} }}  ⇒  {{ {eff_string} }}')
    log.info('Policy:')
    for rule in rules:
        log.info(rule)
    with open(args.output, 'w') as f:
        for rule in rules:
            f.write(f'{rule}\n')
    if args.draw:
        draw_graph(feature_pool, solution, args.draw)


if __name__ == '__main__':
    main()

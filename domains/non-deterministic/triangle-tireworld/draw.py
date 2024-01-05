import argparse
import pddl
import pygraphviz


def draw(name, locations, facts, goal):
    g = pygraphviz.AGraph(strict=False, directed=True)
    g.node_attr['shape'] = 'circle'
    for location in locations:
        g.add_node(location)
    for fact in facts:
        if fact.name == 'road':
            if fact.terms[0] < fact.terms[1] or True:
                g.add_edge(fact.terms[0], fact.terms[1])
        elif fact.name == 'spare-in':
            n = g.get_node(fact.terms[0])
            n.attr['color'] = 'green'
        elif fact.name == 'vehicle-at':
            n = g.get_node(fact.terms[0])
            n.attr['shape'] = 'cds'
    gnode = g.get_node(goal.terms[0])
    gnode.attr['peripheries'] = 2
    g.layout(prog='neato')
    g.draw(f'{name}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', help='problem file')
    args = parser.parse_args()
    problem = pddl.parse_problem(args.problem)
    draw(problem.name, problem.objects, problem.init, problem.goal)


if __name__ == '__main__':
    main()

import argparse
import pddl
import pygraphviz

from collections import defaultdict


def draw(name, objects, facts, goal):
    g = pygraphviz.AGraph(strict=False, directed=True)
    g.node_attr['shape'] = 'circle'
    for location in objects:
        if 'location' in location._type_tags:
            g.add_node(location)
    tires = defaultdict(int)
    for fact in facts:
        if fact.name == 'road':
            g.add_edge(fact.terms[0], fact.terms[1])
        elif fact.name == 'spiky-road':
            g.add_edge(fact.terms[0], fact.terms[1], color='red')
        elif fact.name == 'spare-in':
            n = g.get_node(fact.terms[0])
        elif fact.name == 'tire-at':
            tires[fact.terms[1]] += 1
            n = g.get_node(fact.terms[1])
            n.attr['color'] = 'green'
        elif fact.name == 'vehicle-at':
            n = g.get_node(fact.terms[0])
            n.attr['shape'] = 'cds'
    for tire, count in tires.items():
        n = g.get_node(tire)
        n.attr['label'] = f'{n} ({count})'
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

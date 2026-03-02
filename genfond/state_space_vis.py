import pygraphviz

from .state_space_generator import Alive, StateSpaceNode


def _state_to_str(node):
    state = node.state
    s = f"{node.id}: "
    return s + ",".join([f'{p.name}({",".join([str(p) for p in p.terms])})' for p in sorted(state)])


def _node_color(node: StateSpaceNode) -> str:
    if node.goal:
        return "green"
    alive = node.alive
    if alive == Alive.ALIVE:
        return "black"
    elif alive == Alive.DEAD:
        return "red"
    elif alive in [Alive.PRUNED, Alive.NUM_PRUNED]:
        return "blue"
    else:
        return "gray"


def draw_state_graph(state_graph, filename):
    graph = pygraphviz.AGraph(directed=True)
    graph.node_attr["shape"] = "box"
    for node in state_graph.nodes.values():
        graph.add_node(
            node.id,
            label=_state_to_str(node),
            color=_node_color(node),
        )
        for action, children in node.children.items():
            action_str = f'{action.name}({",".join([str(p) for p in action.parameters])})'
            for child in children:
                graph.add_edge(node.id, child.id, label=action_str)
    graph.layout(prog="dot")
    graph.draw(filename)

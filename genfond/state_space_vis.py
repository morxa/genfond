import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Mapping, Optional

import pygraphviz

from .ground import action_string, state_atoms
from .state_space_generator import Alive, StateSpaceGraph, StateSpaceNode

log = logging.getLogger(__name__)

# Fill color per state type; the model overlay below only touches the border, so the two never clash.
NODE_FILL = {
    Alive.ALIVE: "#ffffff",
    Alive.DEAD: "#ffc7ce",
    Alive.PRUNED: "#dbe9f7",
    Alive.NUM_PRUNED: "#ede1f5",
    Alive.UNKNOWN: "#eeeeee",
}
GOAL_FILL = "#c6efce"
FRONTIER_COLOR = "#1565c0"
CRIT_COLOR = "#c62828"
DEFAULT_BORDER = "#333333"
SELECTED_COLOR = "#2e7d32"
EXCLUDED_COLOR = "#bdbdbd"
NEUTRAL_COLOR = "#616161"


class EdgeMark(Enum):
    SELECTED = auto()
    EXCLUDED = auto()
    NEUTRAL = auto()


@dataclass(frozen=True)
class TransitionSelection:
    """Which transitions a model marks, normalised over the different `solve_prog` encodings.

    The encodings disagree on both shape and polarity: most report the chosen transitions
    (`good_*`), `solve_trans_constraints.lp` reports the rejected ones (`bad_*`), and the action
    is either explicit (`good_trans/4`), only implied (`good_action/3`), or absent altogether
    (`good_trans/3`).  They do share the choice-rule body `alive(I,S), not goal(I,S)`, which is
    what makes a state a decision point and therefore its outgoing transitions meaningful.
    """

    inverted: bool = False
    edges: frozenset[tuple[str, int, str, int]] = frozenset()  # (problem, src, action, dst)
    pairs: frozenset[tuple[str, int, int]] = frozenset()  # (problem, src, dst)
    actions: frozenset[tuple[str, int, str]] = frozenset()  # (problem, src, action)
    node_marks: Mapping[tuple[str, int], frozenset[str]] = field(default_factory=dict)
    has_model: bool = False

    def _matches(self, problem: str, src: int, action: str, dst: int) -> bool:
        if (problem, src, action, dst) in self.edges:
            return True
        # `good_trans/3` abstracts over the action, `good_action/3` over the successor. Only one of
        # them is populated per encoding, so whichever exists is the authority.
        if self.pairs:
            return (problem, src, dst) in self.pairs
        if self.actions:
            return (problem, src, action) in self.actions
        return False

    def mark(self, problem: str, src: int, action: str, dst: int, decision_point: bool) -> EdgeMark:
        if not self.has_model:
            return EdgeMark.NEUTRAL
        if self._matches(problem, src, action, dst):
            return EdgeMark.EXCLUDED if self.inverted else EdgeMark.SELECTED
        if not decision_point:
            # No encoding says anything about transitions out of goal or non-alive states, so
            # claiming they were rejected would be a lie.
            return EdgeMark.NEUTRAL
        return EdgeMark.SELECTED if self.inverted else EdgeMark.EXCLUDED

    def marks_for(self, problem: str, node_id: int) -> frozenset[str]:
        return self.node_marks.get((problem, node_id), frozenset())


def resolve_transitions(
    problem_id_to_name: Mapping[int, str], solution: Optional[Mapping[str, Any]]
) -> TransitionSelection:
    """Turn a clingo model into a `TransitionSelection`, whichever encoding produced it."""
    if not solution:
        # Solver.solution is an empty dict when unsatisfiable, which must not be mistaken for a
        # model in which nothing was selected -- that would mark every transition as rejected.
        return TransitionSelection()
    inverted = "bad_trans" in solution or "bad_action" in solution
    prefix = "bad" if inverted else "good"
    edges: set[tuple[str, int, str, int]] = set()
    pairs: set[tuple[str, int, int]] = set()
    actions: set[tuple[str, int, str]] = set()
    for entry in solution.get(f"{prefix}_trans", set()):
        # Arity is encoded as tuple length by Solver.on_model; a run never mixes the two shapes.
        if len(entry) == 4:
            problem_id, src, action, dst = entry
            if problem_id in problem_id_to_name:
                edges.add((problem_id_to_name[problem_id], src, action, dst))
        elif len(entry) == 3:
            problem_id, src, dst = entry
            if problem_id in problem_id_to_name:
                pairs.add((problem_id_to_name[problem_id], src, dst))
    for problem_id, src, action, dst, _feature, _delta in solution.get(f"{prefix}_trans_delta", set()):
        # Only supplements the action attribution of the /3 encodings: a transition on which no
        # selected feature changes produces no delta at all, so this can never be the authority.
        if problem_id in problem_id_to_name:
            edges.add((problem_id_to_name[problem_id], src, action, dst))
    for problem_id, src, action in solution.get(f"{prefix}_action", set()):
        if problem_id in problem_id_to_name:
            actions.add((problem_id_to_name[problem_id], src, action))
    node_marks: dict[tuple[str, int], set[str]] = dict()
    for key, name in (("frontier", "frontier"), ("crit_state", "crit")):
        for problem_id, node_id in solution.get(key, set()):
            if problem_id in problem_id_to_name:
                node_marks.setdefault((problem_id_to_name[problem_id], node_id), set()).add(name)
    return TransitionSelection(
        inverted=inverted,
        edges=frozenset(edges),
        pairs=frozenset(pairs),
        actions=frozenset(actions),
        node_marks={key: frozenset(value) for key, value in node_marks.items()},
        has_model=True,
    )


def _state_tooltip(node: StateSpaceNode) -> str:
    header = f"state {node.id} ({node.alive.name.lower()}{', goal' if node.goal else ''})"
    # pygraphviz quotes for DOT and graphviz escapes for SVG, so real newlines are correct here.
    return "\n".join([header, ""] + state_atoms(node.state))


def _node_style(node: StateSpaceNode, is_root: bool, marks: frozenset[str], label: str) -> dict[str, str]:
    style = "filled,dashed" if node.alive in (Alive.PRUNED, Alive.NUM_PRUNED) else "filled"
    attrs = {
        "label": label,
        "tooltip": _state_tooltip(node),
        "fillcolor": GOAL_FILL if node.goal else NODE_FILL[node.alive],
        "style": style,
        "color": DEFAULT_BORDER,
        "penwidth": "1",
        "id": f"s{node.id}",
    }
    if node.goal:
        attrs["peripheries"] = "2"
    if "frontier" in marks:
        attrs["color"] = FRONTIER_COLOR
        attrs["penwidth"] = "3"
    elif "crit" in marks:
        attrs["color"] = CRIT_COLOR
        attrs["penwidth"] = "3"
    if is_root:
        attrs["xlabel"] = "root"
    return attrs


def _edge_style(mark: EdgeMark) -> dict[str, str]:
    if mark == EdgeMark.SELECTED:
        return {"color": SELECTED_COLOR, "penwidth": "2.5", "fontcolor": SELECTED_COLOR}
    if mark == EdgeMark.EXCLUDED:
        return {"color": EXCLUDED_COLOR, "style": "dashed", "fontcolor": EXCLUDED_COLOR}
    return {"color": NEUTRAL_COLOR, "penwidth": "1"}


def _show_edge_labels(state_graph: StateSpaceGraph, config: Mapping[str, Any]) -> bool:
    mode = config["state_graph_edge_labels"]
    if mode == "always":
        return True
    if mode == "never":
        return False
    num_edges = sum(len(children) for node in state_graph.nodes.values() for children in node.children.values())
    return num_edges <= config["state_graph_edge_label_limit"]


def _add_legend(graph: pygraphviz.AGraph) -> None:
    # rank=same lays the entries out as one strip instead of a column as tall as the graph itself.
    legend = graph.add_subgraph(name="cluster_legend", label="legend", style="dotted", fontsize="10", rank="same")
    entries = [
        ("legend_goal", "goal", {"fillcolor": GOAL_FILL, "style": "filled", "peripheries": "2"}),
        ("legend_alive", "alive", {"fillcolor": NODE_FILL[Alive.ALIVE], "style": "filled"}),
        ("legend_dead", "dead", {"fillcolor": NODE_FILL[Alive.DEAD], "style": "filled"}),
        ("legend_pruned", "pruned", {"fillcolor": NODE_FILL[Alive.PRUNED], "style": "filled,dashed"}),
        (
            "legend_frontier",
            "frontier",
            {"fillcolor": NODE_FILL[Alive.PRUNED], "style": "filled,dashed", "color": FRONTIER_COLOR, "penwidth": "3"},
        ),
        (
            "legend_crit",
            "critical",
            {"fillcolor": NODE_FILL[Alive.ALIVE], "style": "filled", "color": CRIT_COLOR, "penwidth": "3"},
        ),
    ]
    for name, label, attrs in entries:
        legend.add_node(name, label=label, fontsize="9", **{"color": DEFAULT_BORDER, **attrs})
    # constraint=false: the invisible edges only fix the left-to-right order, they must not rank
    # the entries, or the legend grows into a column as tall as the graph.
    for (source, _, _), (target, _, _) in zip(entries, entries[1:]):
        legend.add_edge(source, target, style="invis", constraint="false")
    for mark, label in ((EdgeMark.SELECTED, "selected"), (EdgeMark.EXCLUDED, "not selected")):
        source, target = f"legend_{mark.name.lower()}_a", f"legend_{mark.name.lower()}_b"
        for name in (source, target):
            legend.add_node(name, label="", shape="point", width="0.05", color=DEFAULT_BORDER)
        legend.add_edge(source, target, label=label, fontsize="9", constraint="false", **_edge_style(mark))
    legend.add_edge(entries[-1][0], f"legend_{EdgeMark.SELECTED.name.lower()}_a", style="invis", constraint="false")


def build_state_graph(
    state_graph: StateSpaceGraph,
    problem_name: str,
    selection: TransitionSelection,
    config: Mapping[str, Any],
    *,
    title: str = "",
) -> pygraphviz.AGraph:
    """Render one state space graph, highlighting what the model selected.

    `strict=False` is essential: a strict graph merges parallel edges, which silently drops the
    nondeterministic branches and the alternative actions between the same pair of states.
    """
    graph = pygraphviz.AGraph(name=problem_name, directed=True, strict=False)
    graph.graph_attr.update(label=title, labelloc="t", fontname="Helvetica", fontsize="12", rankdir="TB")
    graph.node_attr.update(shape="box", style="filled", fontname="Helvetica", fontsize="10", margin="0.06,0.03")
    graph.edge_attr.update(fontname="Helvetica", fontsize="9", arrowsize="0.7")
    # `root` is only set when the graph was not restricted to selected states.
    root = getattr(state_graph, "root", None)
    show_labels = _show_edge_labels(state_graph, config)
    full_state_labels = config["state_graph_node_label"] == "state"
    # Node ids are gappy after prune_nodes(), so never enumerate: the ids are what the model refers to.
    for node in state_graph.nodes.values():
        label = "\n".join(state_atoms(node.state)) if full_state_labels else f"s{node.id}"
        graph.add_node(
            f"s{node.id}",
            **_node_style(node, node is root, selection.marks_for(problem_name, node.id), label),
        )
    for node in state_graph.nodes.values():
        decision_point = node.alive == Alive.ALIVE and not node.goal
        for action, children in node.children.items():
            action_str = action_string(action)
            for child in children:
                mark = selection.mark(problem_name, node.id, action_str, child.id, decision_point)
                graph.add_edge(
                    f"s{node.id}",
                    f"s{child.id}",
                    key=f"{action_str}->{child.id}",
                    label=action_str if show_labels else "",
                    tooltip=action_str,
                    labeltooltip=action_str,
                    **_edge_style(mark),
                )
    if config["state_graph_legend"]:
        _add_legend(graph)
    return graph


def dump_state_graphs(
    state_graphs: Mapping[str, StateSpaceGraph],
    problem_id_to_name: Mapping[int, str],
    solution: Optional[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    round_index: int,
    complexity: int,
) -> list[Path]:
    """Write one visualisation per problem for a single solver round.

    `solution` is None for a round that found no model; the graph is still drawn, without
    highlights, because the state space it searched is exactly what one wants to look at then.
    """
    selection = resolve_transitions(problem_id_to_name, solution)
    directory = Path(config["state_graph_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for problem_name, state_graph in state_graphs.items():
        stem = f"{round_index:03d}-{problem_name}-c{complexity}"
        title = f"{problem_name} | round {round_index} | complexity {complexity}"
        if solution is None:
            title += " | no solution"
        graph = build_state_graph(state_graph, problem_name, selection, config, title=title)
        max_nodes = config["max_state_graph_nodes"]
        too_large = bool(max_nodes) and len(state_graph.nodes) > max_nodes
        if too_large:
            log.warning(
                f"State graph for {problem_name} has {len(state_graph.nodes)} nodes"
                f" (> max_state_graph_nodes={max_nodes}), writing only the graphviz source"
            )
        if config["state_graph_dot"] or too_large:
            dot_path = directory / f"{stem}.dot"
            graph.write(dot_path)
            written.append(dot_path)
        if not too_large:
            # dot's layout is the superlinear part, so only this is skipped for huge graphs.
            image_path = directory / f"{stem}.{config['state_graph_format']}"
            graph.draw(image_path, prog="dot")
            written.append(image_path)
    log.info(f"Wrote {len(written)} state graph file(s) to {directory}")
    return written

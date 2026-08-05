import xml.etree.ElementTree as ElementTree

import pytest

from genfond.config_handler import ConfigHandler
from genfond.feature_generator import FeaturePool
from genfond.ground import state_atoms
from genfond.solver import Solver
from genfond.state_space_generator import Alive, generate_state_space
from genfond.state_space_vis import (
    EdgeMark,
    TransitionSelection,
    build_state_graph,
    dump_state_graphs,
    resolve_transitions,
)


def vis_config(**overrides):
    """The real default config; tests that inspect every edge switch the legend off."""
    config = ConfigHandler()
    config.update(overrides)
    return config


def no_legend(**overrides):
    return vis_config(state_graph_legend=False, **overrides)


def num_transitions(state_graph):
    return sum(len(children) for node in state_graph.nodes.values() for children in node.children.values())


def edge_colors(graph):
    return {edge.attr["color"] for edge in graph.edges()}


def test_node_styles_follow_the_state_type(doors):
    domain, problem = doors
    state_graph = generate_state_space(domain, problem)
    graph = build_state_graph(state_graph, problem.name, TransitionSelection(), vis_config())
    goal_nodes = [node for node in state_graph.nodes.values() if node.goal]
    assert goal_nodes, "fixture is expected to contain a goal state"
    for node in goal_nodes:
        assert graph.get_node(f"s{node.id}").attr["peripheries"] == "2"
        assert graph.get_node(f"s{node.id}").attr["fillcolor"] == "#c6efce"
    for node in state_graph.nodes.values():
        if node.alive == Alive.DEAD and not node.goal:
            assert graph.get_node(f"s{node.id}").attr["fillcolor"] == "#ffc7ce"
    assert graph.get_node(f"s{state_graph.root.id}").attr["xlabel"] == "root"


def test_tooltip_contains_every_atom(simple_blocks):
    domain, problem = simple_blocks
    state_graph = generate_state_space(domain, problem)
    graph = build_state_graph(state_graph, problem.name, TransitionSelection(), vis_config())
    for node in state_graph.nodes.values():
        tooltip = graph.get_node(f"s{node.id}").attr["tooltip"]
        assert f"state {node.id}" in tooltip
        for atom in state_atoms(node.state):
            assert atom in tooltip


def test_edges_carry_the_action(simple_blocks):
    domain, problem = simple_blocks
    state_graph = generate_state_space(domain, problem)
    graph = build_state_graph(state_graph, problem.name, TransitionSelection(), no_legend())
    for edge in graph.edges():
        assert edge.attr["tooltip"]
        assert edge.attr["label"] == edge.attr["tooltip"]


def test_edge_labels_switch_off_above_the_limit(simple_blocks):
    domain, problem = simple_blocks
    state_graph = generate_state_space(domain, problem)
    config = no_legend(state_graph_edge_labels="auto", state_graph_edge_label_limit=0)
    graph = build_state_graph(state_graph, problem.name, TransitionSelection(), config)
    for edge in graph.edges():
        assert edge.attr["label"] == ""
        assert edge.attr["tooltip"], "the action must stay available in the tooltip"
    always = build_state_graph(
        state_graph, problem.name, TransitionSelection(), no_legend(state_graph_edge_labels="always")
    )
    assert all(edge.attr["label"] for edge in always.edges())


def test_legend_is_drawn_alongside_the_graph(simple_blocks):
    domain, problem = simple_blocks
    state_graph = generate_state_space(domain, problem)
    graph = build_state_graph(state_graph, problem.name, TransitionSelection(), vis_config(state_graph_legend=True))
    assert graph.get_subgraph("cluster_legend") is not None
    # The legend must not disturb the graph it documents.
    assert {f"s{node.id}" for node in state_graph.nodes.values()} <= set(graph.nodes())


def test_parallel_transitions_are_kept(fond_blocks):
    """A strict graph would merge parallel edges and silently drop nondeterministic branches."""
    domain, problem = fond_blocks
    state_graph = generate_state_space(domain, problem)
    graph = build_state_graph(state_graph, problem.name, TransitionSelection(), no_legend())
    assert len(graph.edges()) == num_transitions(state_graph)


@pytest.mark.parametrize("solution", [None, {}])
def test_resolve_transitions_without_a_model(solution):
    """Solver.solution is an empty dict when unsat -- that is not a model selecting nothing."""
    selection = resolve_transitions({0: "p"}, solution)
    assert not selection.has_model
    assert selection.mark("p", 0, "pick(a)", 1, decision_point=True) == EdgeMark.NEUTRAL


def test_resolve_transitions_from_good_trans_pairs():
    solution = {"good_trans": {(0, 0, 1), (1, 0, 5)}, "good_action": {(0, 0, "pick(a)")}}
    selection = resolve_transitions({0: "p", 1: "q"}, solution)
    assert not selection.inverted
    assert selection.mark("p", 0, "pick(a)", 1, True) == EdgeMark.SELECTED
    # The /3 shape abstracts over the action, so any action reaching that successor counts.
    assert selection.mark("p", 0, "put(a)", 1, True) == EdgeMark.SELECTED
    assert selection.mark("p", 0, "pick(a)", 2, True) == EdgeMark.EXCLUDED
    assert selection.mark("q", 0, "pick(a)", 5, True) == EdgeMark.SELECTED
    # Transitions out of a non-decision-point state are not claimed either way.
    assert selection.mark("p", 0, "pick(a)", 2, False) == EdgeMark.NEUTRAL


def test_resolve_transitions_from_good_trans_edges():
    solution = {"good_trans": {(0, 0, "pick(a)", 1)}}
    selection = resolve_transitions({0: "p"}, solution)
    assert selection.mark("p", 0, "pick(a)", 1, True) == EdgeMark.SELECTED
    assert selection.mark("p", 0, "put(a)", 1, True) == EdgeMark.EXCLUDED


def test_resolve_transitions_from_trans_delta():
    solution = {
        "good_trans": {(0, 0, 1)},
        "good_trans_delta": {(0, 0, "pick(a)", 1, "b_empty(x)", 1)},
    }
    selection = resolve_transitions({0: "p"}, solution)
    assert ("p", 0, "pick(a)", 1) in selection.edges
    # The delta only supplements the action: the pair stays authoritative, so an action with no
    # feature change on the same selected pair is still selected.
    assert selection.mark("p", 0, "put(a)", 1, True) == EdgeMark.SELECTED


def test_resolve_transitions_from_good_action_only():
    """The datalog encoding shows no good_trans at all, so all outcomes of the action are good."""
    solution = {"good_action": {(0, 0, "pick(a)")}, "frontier": {(0, 3)}}
    selection = resolve_transitions({0: "p"}, solution)
    assert selection.mark("p", 0, "pick(a)", 1, True) == EdgeMark.SELECTED
    assert selection.mark("p", 0, "pick(a)", 2, True) == EdgeMark.SELECTED
    assert selection.mark("p", 0, "put(a)", 1, True) == EdgeMark.EXCLUDED
    assert selection.marks_for("p", 3) == frozenset({"frontier"})


def test_resolve_transitions_inverted():
    solution = {"bad_trans": {(0, 0, 1)}, "bad_action": {(0, 0, "pick(a)")}}
    selection = resolve_transitions({0: "p"}, solution)
    assert selection.inverted
    assert selection.mark("p", 0, "pick(a)", 1, True) == EdgeMark.EXCLUDED
    assert selection.mark("p", 0, "pick(a)", 2, True) == EdgeMark.SELECTED
    assert selection.mark("p", 0, "pick(a)", 2, False) == EdgeMark.NEUTRAL


def test_crit_states_are_marked():
    selection = resolve_transitions({0: "p"}, {"crit_state": {(0, 2)}})
    assert selection.marks_for("p", 2) == frozenset({"crit"})
    assert selection.marks_for("p", 3) == frozenset()


@pytest.mark.parametrize("policy_type", ["state", "d2l", "exact", "datalog"])
def test_selected_edges_come_from_a_real_model(simple_blocks, policy_type):
    """Guards the join between ground.action_string and the action strings in the ASP instance.

    Every encoding reports its selected transitions differently, so run each of them end to end:
    if the action rendering ever drifts apart, highlighting silently degrades to zero matches.
    """
    domain, problem = simple_blocks
    config = ConfigHandler(type=policy_type)
    feature_pool = FeaturePool(domain, [problem], config=config, max_complexity=4, all_generators=True)
    solver = Solver(feature_pool.to_clingo(), num_threads=1, solve_prog=config["solve_prog"])
    assert solver.solve()
    selection = resolve_transitions(feature_pool.problem_id_to_name, solver.solution)
    assert selection.has_model
    graph = build_state_graph(feature_pool.state_graphs[problem.name], problem.name, selection, no_legend())
    colors = edge_colors(graph)
    assert "#2e7d32" in colors, "no transition was highlighted as selected"
    assert "#bdbdbd" in colors, "no alternative transition was marked as not selected"


def test_dump_writes_a_file_set_per_round(tmp_path, simple_blocks):
    domain, problem = simple_blocks
    state_graph = generate_state_space(domain, problem)
    config = vis_config(state_graph_dir=str(tmp_path))
    written = dump_state_graphs(
        {problem.name: state_graph}, {0: problem.name}, None, config, round_index=3, complexity=4
    )
    svg = tmp_path / f"003-{problem.name}-c4.svg"
    dot = tmp_path / f"003-{problem.name}-c4.dot"
    assert set(written) == {svg, dot}
    assert svg.exists() and dot.exists()
    root = ElementTree.parse(svg).getroot()
    tooltips = {
        element.get("{http://www.w3.org/1999/xlink}title") for element in root.iter("{http://www.w3.org/2000/svg}a")
    }
    some_node = next(iter(state_graph.nodes.values()))
    assert any(state_atoms(some_node.state)[0] in (tooltip or "") for tooltip in tooltips)
    dump_state_graphs({problem.name: state_graph}, {0: problem.name}, None, config, round_index=4, complexity=4)
    assert (tmp_path / f"004-{problem.name}-c4.svg").exists()
    assert svg.exists(), "an earlier round must not be overwritten"


def test_dump_without_a_model_is_labelled(tmp_path, simple_blocks):
    domain, problem = simple_blocks
    state_graph = generate_state_space(domain, problem)
    # No legend here: its sample edge legitimately carries the "selected" color.
    config = no_legend(state_graph_dir=str(tmp_path))
    dump_state_graphs({problem.name: state_graph}, {0: problem.name}, None, config, round_index=1, complexity=2)
    dot = (tmp_path / f"001-{problem.name}-c2.dot").read_text()
    assert "no solution" in dot
    assert "#2e7d32" not in dot, "an unsat round must not highlight anything"


def test_oversized_graphs_skip_the_layout(tmp_path, simple_blocks, caplog):
    domain, problem = simple_blocks
    state_graph = generate_state_space(domain, problem)
    config = vis_config(state_graph_dir=str(tmp_path), max_state_graph_nodes=1, state_graph_dot=False)
    written = dump_state_graphs(
        {problem.name: state_graph}, {0: problem.name}, None, config, round_index=1, complexity=2
    )
    assert written == [tmp_path / f"001-{problem.name}-c2.dot"]
    assert not (tmp_path / f"001-{problem.name}-c2.svg").exists()
    assert "max_state_graph_nodes" in caplog.text

def get_action(actions, name, parameters):
    try:
        return [action for action in actions if action.name == name and action.parameters == parameters][0]
    except IndexError as e:
        raise KeyError(f'Action {name}({",".join(parameters)}) not found') from e

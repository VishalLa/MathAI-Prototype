def evaluate_expression(predicted_tokens):
    """
    Evaluates a mathematical expression from a list of predicted tokens.

    Parameters:
        predicted_tokens (list of str): List containing digits and operators as strings (e.g., ['3', '+', '4']).

    Returns:
        result (int/float/str): The evaluated result of the expression or an error message.
    """

    if not predicted_tokens:
        return 'Error: Empty expression'
    
    # Convert token list to string expression
    expr = ''.join(predicted_tokens)

    # Replace sympols if needed
    expr = expr.replace('x', '*').replace('X', '*')
    expr = expr.replace('slash', '/')
    expr = expr.replace('plus', '+')
    expr = expr.replace('minus', '-')
    expr = expr.replace('dot', '.')

    try:
        result = eval(expr)
        return result
    except ZeroDivisionError:
        return 'Error: Division by Zero'
    except Exception as e:
        return f'Error: Invalid expression ({e})'


def evaluate_expression(predicted_tockens):
    if not predicted_tockens:
        return 'Error: Empty expression'

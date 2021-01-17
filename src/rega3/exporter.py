from textwrap import wrap
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model._base import LinearModel
from sklearn.pipeline import Pipeline

__all__ = [
    'export_estimator'
]

def export_estimator(estimator, function_name, decimal_places=6, indent=4*' ',
                     line_length=80):
    '''Export trained regression model from Scikit-learn to SQF code.

    :param estimator: Fitted estimator to export_estimator.
    :type estimator: sklearn.base.BaseEstimator
    :param function_name: Name for the SQF function.
    :type function_name: str
    :param decimal_places: Number of decimal places for coefficients.
        Defaults to 6.
    :type decimal_places: int, optional
    :param indent: The string used for indentations. Defaults to four spaces.
    :type indent: str, optional
    :param line_length: Maximum line length for the code. Defaults to 80.
    :type line_length: int, optional

    :retun: SQF code as
    :rtype: str
    '''
    if isinstance(estimator, Pipeline):
        return _export_pipeline(estimator, function_name,
                                decimal_places=decimal_places, indent=indent,
                                line_length=line_length)
    elif isinstance(estimator, PolynomialFeatures):
        return _export_polynomial_feature(estimator, function_name,
                                          decimal_places=decimal_places,
                                          indent=indent,
                                          line_length=line_length)
    elif isinstance(estimator, LinearModel):
        return _export_linear_model(estimator, function_name,
                                    decimal_places=decimal_places,
                                    indent=indent,
                                    line_length=line_length)
    else:
        return ''

def _export_pipeline(estimator, function_name, indent=4*' ', **kwargs):
    output = f'{function_name} = {{\n'
    step_output = ''
    for step_name, step_estimator in estimator.named_steps.items():
        step_function_name = f'{function_name}_{step_name}'
        output += f'{indent}_this = _this call {step_function_name};\n'
        step_output += export_estimator(step_estimator, step_function_name,
                                        indent=indent, **kwargs)
        step_output += '\n'
    output += f'{indent}_this\n'
    output += '};\n'
    return (step_output + output)

def _export_polynomial_feature(estimator, function_name, decimal_places=6,
                               indent=4*' ', line_length=80):
    output = f'{function_name} = {{[\n'
    output += indent
    elements = []
    for row in estimator.powers_:
        factors = []
        for i, power in enumerate(row):
            if power == 1:
                factors.append(f'(_this#{i})')
            elif power != 0:
                factors.append(f'(_this#{i})^{power}')
        if len(factors) > 0:
            elements.append("*".join(factors))
        else:
            elements.append('1')
    body = ', '.join(elements)
    body += ']'
    body = f'\n{indent}'.join(wrap(body, line_length-len(indent)))
    output += f'{body}\n'
    output += '};\n'
    return output

def _export_linear_model(estimator, function_name, decimal_places=6,
                         indent=4*' ', line_length=80):
    output = f'{function_name} = {{\n'
    output += indent
    summands = []
    for i, coef in enumerate(estimator.coef_):
        if coef != 0:
            summands.append(f'{coef:.{decimal_places}e}*(_this#{i})')
    summands.append(f'{estimator.intercept_:.{decimal_places}e}')
    body = ' + '.join(summands)
    body = body.replace('+ -', '- ')
    body = f'\n{indent}'.join(wrap(body, line_length-len(indent)))
    output += f'{body}\n'
    output += '};\n'
    return output

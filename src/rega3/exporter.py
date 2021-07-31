from textwrap import wrap
import sympy as sp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model._base import LinearModel
from sklearn.model_selection._search import BaseSearchCV
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

    :retun: SQF code
    :rtype: str
    '''
    _export_function = None
    if isinstance(estimator, BaseSearchCV):
        _export_function = _export_search
    elif isinstance(estimator, LinearModel):
        _export_function = _export_linear_model
    elif isinstance(estimator, Pipeline):
        _export_function = _export_pipeline
    elif isinstance(estimator, PolynomialFeatures):
        _export_function = _export_polynomial_feature
    elif isinstance(estimator, StandardScaler):
        _export_function = _export_standard_scaler

    if _export_function is None:
        raise NotImplementedError('Cannot export estimator of type '
                                  f'{type(estimator)}')

    return _export_function(estimator, function_name,
                            decimal_places=decimal_places, indent=indent,
                            line_length=line_length)

def _wrap(text, indent, line_length):
    return f'\n{indent}'.join(wrap(text, line_length-len(indent)))

def _export_search(estimator, function_name, **kwargs):
    return export_estimator(estimator.best_estimator_ , function_name,
                            **kwargs)


def _export_linear_model(estimator, function_name, decimal_places=6,
                         indent=4*' ', line_length=80):
    output = f'{function_name} = {{\n'
    output += indent
    bias = f'{estimator.intercept_:.{decimal_places}e}'
    coefs = (f'[{coef:.{decimal_places}e}]' for coef in estimator.coef_)
    coef_vec = f'[{", ".join(coefs)}]'
    body = f'([_this] matrixMultiply {coef_vec})#0#0 + {bias}'
    body = body.replace('+ -', '- ')
    body = _wrap(body, indent, line_length)
    output += f'{body}\n'
    output += '};\n'
    return output


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
    body = _wrap(body, indent, line_length)
    output += f'{body}\n'
    output += '};\n'
    return output


def _export_standard_scaler(estimator, function_name, decimal_places=6,
                               indent=4*' ', line_length=80):
    output = f'{function_name} = {{[\n'
    output += indent
    elements = []
    for i, (mean, scale) in enumerate(zip(estimator.mean_, estimator.scale_)):
        elements.append(f'((_this#{i})-{mean:.{decimal_places}e}) / '
                        f'{scale:.{decimal_places}e}')
    body = ', '.join(elements)
    body += ']'
    body = _wrap(body, indent, line_length)
    output += f'{body}\n'
    output += '};\n'
    return output

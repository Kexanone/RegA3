from textwrap import wrap
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model._base import LinearModel
from sklearn.pipeline import Pipeline

__all__ = [
    'export_estimator'
]

def export_estimator(estimator, function_name, tab=4*' '):
    if isinstance(estimator, Pipeline):
        return _export_pipeline(estimator, function_name, tab=tab)
    elif isinstance(estimator, PolynomialFeatures):
        return _export_polynomial_feature(estimator, function_name, tab=tab)
    elif isinstance(estimator, LinearModel):
        return _export_linear_model(estimator, function_name, tab=tab)
    else:
        return ''

def _export_pipeline(estimator, function_name, tab=4*' '):
    output = f'{function_name} = {{\n'
    step_output = ''
    for step_name, step_estimator in estimator.named_steps.items():
        step_function_name = f'{function_name}_{step_name}'
        output += f'{tab}_this = _this call {step_function_name};\n'
        step_output += export_estimator(step_estimator, step_function_name, tab=tab)
        step_output += '\n'
    output += f'{tab}_this\n'
    output += '};\n'
    return (step_output + output)

def _export_polynomial_feature(estimator, function_name, tab=4*' '):
    output = f'{function_name} = {{[\n'
    output += tab
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
    body = f'\n{tab}'.join(wrap(body, 76))
    output += f'{body}\n'
    output += '};\n'
    return output

def _export_linear_model(estimator, function_name, tab=4*' '):
    output = f'{function_name} = {{\n'
    output += tab
    summands = []
    for i, coef in enumerate(estimator.coef_):
        summands.append(f'{coef}*(_this#{i})')
    summands.append(str(estimator.intercept_))
    body = ' + '.join(summands)
    body = body.replace('+ -', '- ')
    body = f'\n{tab}'.join(wrap(body, 76))
    output += f'{body}\n'
    output += '};\n'
    return output

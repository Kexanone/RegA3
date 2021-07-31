# RegA3
Export trained regression models from Scikit-learn to SQF.

## Installation
```sh
git clone https://github.com/Kexanone/rega3
cd rega3
pip install .
```

## Usage
Exporting a simple polynomial model:
```py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from rega3 import export_estimator

# Generate toy data
x = np.arange(100)
X = x.reshape(-1, 1)
y = x**3

# Fit data
pipe = Pipeline([('polyfeat', PolynomialFeatures()), ('linreg', LinearRegression())])
param_grid = {'polyfeat__degree': range(1, 5)}
search = GridSearchCV(pipe, param_grid)
search.fit(X, y)

# Export model
print(export_estimator(search, 'rega_fnc_test'))
```
The printed output will look like this:
```sqf
rega_fnc_test_polyfeat = {[
    1, (_this#0), (_this#0)^2, (_this#0)^3]
};

rega_fnc_test_linreg = {
    ([_this] matrixMultiply [[0.000000e+00], [-5.226124e-14], [1.296163e-14],
    [1.000000e+00]])#0#0 + 5.820766e-11
};

rega_fnc_test = {
    _this = _this call rega_fnc_test_polyfeat;
    _this = _this call rega_fnc_test_linreg;
    _this
};
```
More advanced examples can be found in the [example folder](example).
## Currently supported classes
`sklearn.linear_model`:
- [x] `ElasticNet`
- [x] `Lasso`
- [x] `LinearRegression`
- [x] `Ridge`

`sklearn.model_selection`:
- [x] `GridSearchCV`
- [x] `HalvingGridSearchCV`
- [x] `HalvingRandomSearchCV`
- [x] `RandomizedSearchCV`

`sklearn.preprocessing`:
- [x] `PolynomialFeatures`
- [x] `StandardScaler`

`sklearn.pipeline`:
- [x] `Pipeline`

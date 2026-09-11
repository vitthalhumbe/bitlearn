# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Vitthal Humbe

import pytest
import numpy as np
from bitlearn import LinearRegression


# CURRENT TEST MILESTONE : 21/21 PASSED

# test dataset: y = 4 + 3x + noise
@pytest.fixture
def linear_data():
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y


def test_invalid_regul():
    with pytest.raises(ValueError):
        LinearRegression(regul="l3")

def test_negative_lambda():
    with pytest.raises(ValueError):
        LinearRegression(lambda_=-1.0)

def test_invalid_alpha():
    with pytest.raises(ValueError):
        LinearRegression(regul="elastic_net", alpha=1.5)

def test_normal_eq_with_l1_raises():
    with pytest.raises(ValueError):
        LinearRegression(method="normal_eq", regul="l1")

def test_normal_eq_with_elastic_net_raises():
    with pytest.raises(ValueError):
        LinearRegression(method="normal_eq", regul="elastic_net")




def test_fit_sets_attributes(linear_data):
    X, y = linear_data
    model = LinearRegression()
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert model.coef_.shape == (1,)

def test_fit_returns_self(linear_data):
    X, y = linear_data
    model = LinearRegression()
    result = model.fit(X, y)
    assert result is model



def test_normal_eq_coef_close(linear_data):
    X, y = linear_data
    model = LinearRegression(method="normal_eq")
    model.fit(X, y)
    # true slope is 3, intercept is 4 — allow reasonable tolerance
    assert abs(model.coef_[0] - 3.0) < 0.3
    assert abs(model.intercept_ - 4.0) < 0.3

def test_normal_eq_l2(linear_data):
    X, y = linear_data
    model = LinearRegression(method="normal_eq", regul="l2", lambda_=0.1)
    model.fit(X, y)
    assert model.coef_.shape == (1,)




def test_gd_r2(linear_data):
    X, y = linear_data
    model = LinearRegression(method="gd", lr=0.01, epochs=1000)
    model.fit(X, y)
    assert model.score(X, y) > 0.7

def test_gd_loss_history_length(linear_data):
    X, y = linear_data
    model = LinearRegression(method="gd", lr=0.01, epochs=500)
    model.fit(X, y)
    assert len(model.get_loss_history()) == 500

def test_gd_loss_decreases(linear_data):
    X, y = linear_data
    model = LinearRegression(method="gd", lr=0.01, epochs=1000)
    model.fit(X, y)
    history = model.get_loss_history()
    # loss in second half should be lower than first half on average
    mid = len(history) // 2
    assert np.mean(history[mid:]) < np.mean(history[:mid])




def test_sgd_r2(linear_data):
    X, y = linear_data
    model = LinearRegression(method="sgd", lr=0.01, epochs=1000)
    model.fit(X, y)
    assert model.score(X, y) > 0.7

def test_sgd_loss_history_length(linear_data):
    X, y = linear_data
    model = LinearRegression(method="sgd", lr=0.01, epochs=300)
    model.fit(X, y)
    assert len(model.get_loss_history()) == 300



@pytest.mark.parametrize("regul", ["l1", "l2", "elastic_net"])
def test_gd_with_regul(linear_data, regul):
    X, y = linear_data
    model = LinearRegression(method="gd", regul=regul, lambda_=0.1, alpha=0.5, epochs=1000)
    model.fit(X, y)
    assert model.score(X, y) > 0.7



def test_predict_shape(linear_data):
    X, y = linear_data
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)

def test_predict_reasonable_values(linear_data):
    X, y = linear_data
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.min() > 0
    assert preds.max() < 20



@pytest.mark.parametrize("method", ["gd", "sgd", "normal_eq"])
def test_l2_shrinks_coef(linear_data, method):
    X, y = linear_data

    model_plain = LinearRegression(method=method, lr=0.01, epochs=1000, regul="l2", lambda_=0.0)
    model_plain.fit(X, y)

    model_reg = LinearRegression(method=method, lr=0.01, epochs=1000, regul="l2", lambda_=10.0)
    model_reg.fit(X, y)

    assert not np.allclose(model_plain.coef_, model_reg.coef_)
    assert abs(model_reg.coef_[0]) < abs(model_plain.coef_[0])


@pytest.mark.parametrize("regul", ["l1", "elastic_net"])
def test_l1_and_elastic_net_shrink_coef(linear_data, regul):
    X, y = linear_data

    model_plain = LinearRegression(method="gd", lr=0.01, epochs=1000, regul=regul, lambda_=0.0, alpha=0.5)
    model_plain.fit(X, y)

    model_reg = LinearRegression(method="gd", lr=0.01, epochs=1000, regul=regul, lambda_=10.0, alpha=0.5)
    model_reg.fit(X, y)

    assert not np.allclose(model_plain.coef_, model_reg.coef_)

def test_score_perfect():
    

    X = np.linspace(0, 1, 100).reshape(-1, 1)
    y = 2 * X + 1
    model = LinearRegression(method="normal_eq")
    model.fit(X, y)
    assert model.score(X, y) > 0.999

def test_score_range(linear_data):
    X, y = linear_data
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    assert 0.0 <= r2 <= 1.0
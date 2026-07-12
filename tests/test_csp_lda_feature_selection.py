import numpy as np

from modern_bci_v2 import CSPFilter, FeatureSelector, SimpleLDA


def test_csp_filters_shape_after_fit():
    rng = np.random.default_rng(42)
    n_trials, n_channels, n_timepoints = 20, 2, 500
    class1 = rng.normal(0, 1, size=(n_trials, n_channels, n_timepoints))
    class2 = rng.normal(0, 1, size=(n_trials, n_channels, n_timepoints))

    csp = CSPFilter(n_filters=2)
    csp.fit(class1, class2)

    assert csp.filters.shape == (n_channels, 2)


def test_csp_transform_logvar_discriminates_variance_classes():
    # Class 1: channel 0 has much higher variance than channel 1.
    # Class 2: the reverse. CSP should find a spatial filter that separates
    # the two classes by log-variance, since that is exactly what CSP optimizes.
    rng = np.random.default_rng(7)
    n_trials, n_timepoints = 30, 500

    def make_trial(hi_var_channel):
        ch0 = rng.normal(0, 3.0 if hi_var_channel == 0 else 0.5, n_timepoints)
        ch1 = rng.normal(0, 3.0 if hi_var_channel == 1 else 0.5, n_timepoints)
        return np.vstack([ch0, ch1])

    class1_trials = np.array([make_trial(hi_var_channel=0) for _ in range(n_trials)])
    class2_trials = np.array([make_trial(hi_var_channel=1) for _ in range(n_trials)])

    csp = CSPFilter(n_filters=2)
    csp.fit(class1_trials, class2_trials)

    logvar1 = csp.transform_logvar(class1_trials)
    logvar2 = csp.transform_logvar(class2_trials)

    assert logvar1.shape == (n_trials, 2)
    # The first CSP component's log-variance should clearly separate the two
    # classes (CSP maximizes exactly this variance-ratio contrast).
    assert abs(logvar1[:, 0].mean() - logvar2[:, 0].mean()) > 1.0


def test_csp_transform_without_fit_returns_input_unchanged():
    csp = CSPFilter(n_filters=2)
    X = np.ones((2, 10))
    assert np.array_equal(csp.transform(X), X)


def test_feature_selector_ranks_discriminative_feature_first():
    rng = np.random.default_rng(0)
    n_samples, n_features = 100, 10
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    X = rng.normal(0, 1, size=(n_samples, n_features))
    X[y == 1, 3] += 6.0  # feature 3: strongly discriminative
    X[y == 1, 7] += 0.5  # feature 7: mildly discriminative

    selector = FeatureSelector(k=3)
    selector.fit(X, y)

    assert selector.selected_idx[0] == 3
    assert 3 in selector.selected_idx

    X_sel = selector.transform(X)
    assert X_sel.shape == (n_samples, 3)


def test_feature_selector_k_capped_at_n_features():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(20, 4))
    y = np.array([0] * 10 + [1] * 10)
    selector = FeatureSelector(k=100)  # k larger than n_features
    selector.fit(X, y)
    assert len(selector.selected_idx) == 4


def test_simple_lda_separates_well_separated_classes():
    rng = np.random.default_rng(1)
    n = 60
    X0 = rng.normal(loc=[-3, -3], scale=0.5, size=(n, 2))
    X1 = rng.normal(loc=[3, 3], scale=0.5, size=(n, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)

    lda = SimpleLDA(shrinkage=0.0)
    lda.fit(X, y)
    preds = lda.predict(X)

    accuracy = (preds == y).mean()
    assert accuracy > 0.95

    proba = lda.predict_proba(X)
    assert proba.shape == (2 * n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    # class-1 samples should get high P(class=1)
    assert proba[y == 1, 1].mean() > 0.9


def test_simple_lda_shrinkage_changes_covariance_toward_identity_scale():
    rng = np.random.default_rng(2)
    n = 40
    X0 = rng.normal(loc=[-2, 0], scale=[3.0, 0.2], size=(n, 2))
    X1 = rng.normal(loc=[2, 0], scale=[3.0, 0.2], size=(n, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)

    lda_no_shrink = SimpleLDA(shrinkage=0.0)
    lda_no_shrink.fit(X, y)
    lda_full_shrink = SimpleLDA(shrinkage=1.0)
    lda_full_shrink.fit(X, y)

    # With shrinkage=1.0, pooled_cov must equal avg_var * I exactly (fully
    # shrunk toward the isotropic target), so off-diagonal terms vanish.
    off_diag = lda_full_shrink.pooled_cov[0, 1]
    assert abs(off_diag) < 1e-8
    # Without shrinkage, the raw covariance is not required to be isotropic
    # given these anisotropic-scale classes.
    assert not np.allclose(lda_no_shrink.pooled_cov, lda_full_shrink.pooled_cov)

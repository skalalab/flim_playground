import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.classify import run_classification


class ClassificationHyperparamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_classification(
            n_samples=180,
            n_features=8,
            n_informative=5,
            n_redundant=1,
            n_classes=2,
            random_state=17,
        )
        cls.df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        cls.df["classes"] = np.where(y == 1, "positive", "negative")

    def _run(self, method, classifier_params=None, class_weight="None"):
        error_msg, results = run_classification(
            self.df,
            method,
            splits=0.7,
            sampling_method="None",
            class_weight=class_weight,
            threshold_method="None",
            classifier_params=classifier_params,
            random_state=42,
        )
        self.assertEqual(error_msg, "", msg=error_msg)
        self.assertIsNotNone(results)
        return results

    def test_default_params_equivalent_to_empty_dict(self):
        methods = ["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression"]
        for method in methods:
            with self.subTest(method=method):
                default_results = self._run(method, classifier_params=None)
                empty_results = self._run(method, classifier_params={})
                np.testing.assert_array_equal(default_results["y_pred"], empty_results["y_pred"])

    def test_random_forest_custom_params_are_applied(self):
        results = self._run(
            "Random Forest",
            classifier_params={
                "n_estimators": 250,
                "max_depth": 6,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
                "max_features": "log2",
                "criterion": "entropy",
                "bootstrap": False,
            },
            class_weight="Balanced",
        )
        clf = results["classifier"]
        self.assertEqual(clf.n_estimators, 250)
        self.assertEqual(clf.max_depth, 6)
        self.assertEqual(clf.min_samples_split, 4)
        self.assertEqual(clf.min_samples_leaf, 2)
        self.assertEqual(clf.max_features, "log2")
        self.assertEqual(clf.criterion, "entropy")
        self.assertFalse(clf.bootstrap)
        self.assertEqual(clf.class_weight, "balanced")

    def test_svm_custom_params_are_applied(self):
        results = self._run(
            "SVM",
            classifier_params={
                "kernel": "rbf",
                "C": 2.5,
                "gamma": "auto",
                "tol": 0.0005,
            },
            class_weight="Balanced",
        )
        svc = results["classifier"].named_steps["svc"]
        self.assertEqual(svc.kernel, "rbf")
        self.assertAlmostEqual(svc.C, 2.5)
        self.assertEqual(svc.gamma, "auto")
        self.assertAlmostEqual(svc.tol, 0.0005)
        self.assertEqual(svc.class_weight, "balanced")

    def test_logistic_regularization_params_are_applied(self):
        results = self._run(
            "Logistic Regression",
            classifier_params={
                "solver": "lbfgs",
                "regularization": "l2",
                "l1_ratio": 0.7,
                "C": 0.8,
                "max_iter": 3000,
            },
        )
        lr = results["classifier"].named_steps["logisticregression"]
        self.assertEqual(lr.solver, "lbfgs")
        self.assertAlmostEqual(lr.C, 0.8)
        self.assertEqual(lr.max_iter, 3000)
        self.assertAlmostEqual(lr.l1_ratio, 0.0)

    def test_logistic_regularization_l1_maps_to_l1_ratio(self):
        results = self._run(
            "Logistic Regression",
            classifier_params={
                "solver": "saga",
                "regularization": "l1",
                "C": 0.6,
                "max_iter": 2000,
            },
        )
        lr = results["classifier"].named_steps["logisticregression"]
        self.assertEqual(lr.solver, "saga")
        self.assertAlmostEqual(lr.C, 0.6)
        self.assertEqual(lr.max_iter, 2000)
        self.assertAlmostEqual(lr.l1_ratio, 1.0)

    def test_logistic_regularization_none_mapping(self):
        results = self._run(
            "Logistic Regression",
            classifier_params={
                "solver": "lbfgs",
                "regularization": "none",
                "C": 0.5,
            },
        )
        lr = results["classifier"].named_steps["logisticregression"]
        self.assertGreaterEqual(lr.C, 1e12)
        self.assertAlmostEqual(lr.l1_ratio, 0.0)

    def test_gradient_boosting_custom_params_are_applied(self):
        results = self._run(
            "Gradient Boosting",
            classifier_params={
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 2,
                "subsample": 0.8,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "n_iter_no_change": 5,
                "validation_fraction": 0.15,
                "tol": 0.0002,
            },
        )
        clf = results["classifier"]
        self.assertEqual(clf.n_estimators, 200)
        self.assertAlmostEqual(clf.learning_rate, 0.05)
        self.assertEqual(clf.max_depth, 2)
        self.assertAlmostEqual(clf.subsample, 0.8)
        self.assertEqual(clf.min_samples_split, 5)
        self.assertEqual(clf.min_samples_leaf, 2)
        self.assertEqual(clf.max_features, "sqrt")
        self.assertEqual(clf.n_iter_no_change, 5)
        self.assertAlmostEqual(clf.validation_fraction, 0.15)
        self.assertAlmostEqual(clf.tol, 0.0002)


if __name__ == "__main__":
    unittest.main()

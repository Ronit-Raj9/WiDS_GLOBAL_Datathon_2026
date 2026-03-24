from __future__ import annotations

import argparse
import importlib
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

HAS_MATPLOTLIB = False
HAS_SEABORN = False
HAS_LIFELINES = False
HAS_SCIPY = False
plt = None
sns = None
KaplanMeierFitter = None
logrank_test = None
ks_2samp = None

try:
	matplotlib = importlib.import_module("matplotlib")
	matplotlib.use("Agg")
	plt = importlib.import_module("matplotlib.pyplot")
	HAS_MATPLOTLIB = True
except Exception:
	HAS_MATPLOTLIB = False

try:
	sns = importlib.import_module("seaborn")
	HAS_SEABORN = HAS_MATPLOTLIB
except Exception:
	HAS_SEABORN = False

try:
	lifelines_module = importlib.import_module("lifelines")
	KaplanMeierFitter = getattr(lifelines_module, "KaplanMeierFitter")
	logrank_mod = importlib.import_module("lifelines.statistics")
	logrank_test = getattr(logrank_mod, "logrank_test")
	HAS_LIFELINES = True
except Exception:
	HAS_LIFELINES = False

try:
	scipy_stats = importlib.import_module("scipy.stats")
	ks_2samp = getattr(scipy_stats, "ks_2samp")
	HAS_SCIPY = True
except Exception:
	HAS_SCIPY = False


warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
META_PATH = DATA_DIR / "metaData.csv"
SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

HORIZONS = [12, 24, 48, 72]
EPS = 1e-9


def make_output_dirs(base_dir: Path) -> Dict[str, Path]:
	tables_dir = base_dir / "tables"
	figs_dir = base_dir / "figures"
	json_dir = base_dir / "json"
	for directory in [base_dir, tables_dir, figs_dir, json_dir]:
		directory.mkdir(parents=True, exist_ok=True)
	return {
		"base": base_dir,
		"tables": tables_dir,
		"figures": figs_dir,
		"json": json_dir,
	}


def setup_plot_style() -> None:
	if not HAS_MATPLOTLIB:
		return
	if HAS_SEABORN:
		sns.set_theme(style="whitegrid", context="talk")
	else:
		plt.style.use("ggplot")


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	train = pd.read_csv(TRAIN_PATH)
	test = pd.read_csv(TEST_PATH)
	metadata = pd.read_csv(META_PATH)
	sample_submission = pd.read_csv(SUBMISSION_PATH)
	return train, test, metadata, sample_submission


def extract_feature_groups(metadata: pd.DataFrame) -> Dict[str, List[str]]:
	groups = {}
	for category, chunk in metadata.groupby("category"):
		groups[category] = chunk["column"].tolist()
	return groups


def validate_schema(
	train: pd.DataFrame,
	test: pd.DataFrame,
	metadata: pd.DataFrame,
	sample_submission: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
	identifier_cols = metadata.loc[metadata["type"] == "identifier", "column"].tolist()
	feature_cols = metadata.loc[metadata["type"] == "feature", "column"].tolist()
	target_cols = metadata.loc[metadata["type"] == "target", "column"].tolist()

	expected_train = set(identifier_cols + feature_cols + target_cols)
	expected_test = set(identifier_cols + feature_cols)
	expected_submission = ["event_id", "prob_12h", "prob_24h", "prob_48h", "prob_72h"]

	rows = []

	train_missing = sorted(expected_train - set(train.columns))
	train_extra = sorted(set(train.columns) - expected_train)
	test_missing = sorted(expected_test - set(test.columns))
	test_extra = sorted(set(test.columns) - expected_test)
	submission_columns_match = sample_submission.columns.tolist() == expected_submission

	rows.append(
		{
			"dataset": "train",
			"n_rows": int(train.shape[0]),
			"n_cols": int(train.shape[1]),
			"missing_columns": ", ".join(train_missing),
			"extra_columns": ", ".join(train_extra),
			"ok": len(train_missing) == 0,
		}
	)
	rows.append(
		{
			"dataset": "test",
			"n_rows": int(test.shape[0]),
			"n_cols": int(test.shape[1]),
			"missing_columns": ", ".join(test_missing),
			"extra_columns": ", ".join(test_extra),
			"ok": len(test_missing) == 0,
		}
	)
	rows.append(
		{
			"dataset": "sample_submission",
			"n_rows": int(sample_submission.shape[0]),
			"n_cols": int(sample_submission.shape[1]),
			"missing_columns": "",
			"extra_columns": "",
			"ok": submission_columns_match,
		}
	)

	schema_summary = {
		"identifier_cols": identifier_cols,
		"feature_cols": feature_cols,
		"target_cols": target_cols,
		"train_missing": train_missing,
		"test_missing": test_missing,
		"submission_columns_match": submission_columns_match,
	}
	return pd.DataFrame(rows), schema_summary


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
	engineered = df.copy()
	if "closing_speed_m_per_h" in engineered.columns:
		engineered["closing_speed_m_per_h"] = engineered["closing_speed_m_per_h"].fillna(
			engineered["closing_speed_m_per_h"].median()
		)
	if "radial_growth_rate_m_per_h" in engineered.columns:
		engineered["radial_growth_rate_m_per_h"] = engineered["radial_growth_rate_m_per_h"].fillna(
			engineered["radial_growth_rate_m_per_h"].median()
		)
	if "area_growth_rel_0_5h" in engineered.columns:
		cap = engineered["area_growth_rel_0_5h"].quantile(0.99)
		engineered["area_growth_rel_0_5h"] = engineered["area_growth_rel_0_5h"].clip(upper=cap)

	engineered["persistence"] = np.log1p(engineered["num_perimeters_0_5h"])
	engineered["dist"] = (engineered["dist_min_ci_0_5h"] - 5000).clip(lower=10)

	engineered["v_stable"] = (
		(engineered["closing_speed_m_per_h"] + engineered["radial_growth_rate_m_per_h"]).clip(lower=1.0)
		* (engineered["alignment_abs"] ** 1.5)
		* engineered["persistence"]
	)
	engineered["eta_kinetic"] = engineered["dist"] / engineered["v_stable"].clip(lower=0.1)
	engineered["density_metric"] = engineered["area_first_ha"] / (engineered["dist"] + 1)
	engineered["speed_alignment"] = engineered["closing_speed_m_per_h"] * engineered["alignment_abs"]

	engineered["kinetic_energy"] = engineered["area_first_ha"] * engineered["v_stable"] ** 2
	engineered["approach_rate"] = engineered["closing_speed_m_per_h"] * engineered["alignment_abs"]
	engineered["lateral_motion"] = engineered["centroid_speed_m_per_h"] * (1 - engineered["alignment_abs"])

	engineered["dist_to_initial_ratio"] = engineered["dist_min_ci_0_5h"] / (
		engineered["dist_min_ci_0_5h"].mean() + 1
	)
	engineered["dist_std_normalized"] = engineered["dist_std_ci_0_5h"] / (
		engineered["dist_min_ci_0_5h"] + 1
	)
	engineered["growth_efficiency"] = engineered["area_growth_rate_ha_per_h"] / (
		engineered["area_first_ha"] + 1
	)
	engineered["radial_to_area_ratio"] = engineered["radial_growth_rate_m_per_h"] / (
		np.sqrt(engineered["area_first_ha"]) + 1
	)

	engineered["is_night"] = (
		(engineered["event_start_hour"] >= 20) | (engineered["event_start_hour"] <= 5)
	).astype(int)
	engineered["is_weekend"] = (engineered["event_start_dayofweek"] >= 5).astype(int)
	engineered["hour_sin"] = np.sin(2 * np.pi * engineered["event_start_hour"] / 24)
	engineered["hour_cos"] = np.cos(2 * np.pi * engineered["event_start_hour"] / 24)

	engineered["dist_squared"] = engineered["dist"] ** 2 / 1e10
	engineered["v_stable_squared"] = engineered["v_stable"] ** 2
	engineered["alignment_squared"] = engineered["alignment_abs"] ** 2
	engineered["closing_to_centroid_ratio"] = engineered["closing_speed_m_per_h"] / (
		engineered["centroid_speed_m_per_h"].clip(lower=0.1) + 1
	)
	return engineered


def add_horizon_labels(train: pd.DataFrame) -> pd.DataFrame:
	out = train.copy()
	for horizon in HORIZONS:
		out[f"y_{horizon}"] = ((out["time_to_hit_hours"] <= horizon) & (out["event"] == 1)).astype(int)
	return out


def describe_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
	return pd.DataFrame(
		[
			{
				"dataset": name,
				"rows": int(df.shape[0]),
				"cols": int(df.shape[1]),
				"memory_mb": float(df.memory_usage(deep=True).sum() / (1024 ** 2)),
				"duplicate_rows": int(df.duplicated().sum()),
				"duplicate_event_id": int(df["event_id"].duplicated().sum()) if "event_id" in df.columns else np.nan,
			}
		]
	)


def quality_audit(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
	rows = []
	numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

	for col in df.columns:
		series = df[col]
		n_missing = int(series.isna().sum())
		missing_pct = float(n_missing / len(df) * 100)
		nunique = int(series.nunique(dropna=True))
		row = {
			"dataset": dataset_name,
			"column": col,
			"dtype": str(series.dtype),
			"n_missing": n_missing,
			"missing_pct": missing_pct,
			"n_unique": nunique,
			"is_constant": nunique <= 1,
			"is_near_constant": False,
			"zero_count": np.nan,
			"zero_pct": np.nan,
			"q01": np.nan,
			"q05": np.nan,
			"q50": np.nan,
			"q95": np.nan,
			"q99": np.nan,
			"iqr": np.nan,
			"iqr_outliers": np.nan,
		}

		if col in numeric_cols:
			non_na = series.dropna()
			if len(non_na) > 0:
				row["zero_count"] = int((non_na == 0).sum())
				row["zero_pct"] = float((non_na == 0).mean() * 100)
				q01, q05, q50, q95, q99 = np.nanquantile(non_na, [0.01, 0.05, 0.5, 0.95, 0.99])
				q25, q75 = np.nanquantile(non_na, [0.25, 0.75])
				iqr = q75 - q25
				if iqr > 0:
					lower = q25 - 1.5 * iqr
					upper = q75 + 1.5 * iqr
					iqr_outliers = int(((non_na < lower) | (non_na > upper)).sum())
				else:
					iqr_outliers = 0
				row.update(
					{
						"q01": float(q01),
						"q05": float(q05),
						"q50": float(q50),
						"q95": float(q95),
						"q99": float(q99),
						"iqr": float(iqr),
						"iqr_outliers": iqr_outliers,
					}
				)
				value_counts = non_na.value_counts(normalize=True)
				if len(value_counts) > 0:
					row["is_near_constant"] = bool(value_counts.iloc[0] >= 0.98)

		rows.append(row)

	return pd.DataFrame(rows)


def target_audit(train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	event_table = (
		train["event"].value_counts(dropna=False)
		.rename_axis("event")
		.reset_index(name="count")
		.sort_values("event")
	)
	event_table["pct"] = event_table["count"] / len(train) * 100

	horizon_rows = []
	for h in HORIZONS:
		y_h = ((train["time_to_hit_hours"] <= h) & (train["event"] == 1)).astype(int)
		horizon_rows.append(
			{
				"horizon": h,
				"positive": int(y_h.sum()),
				"negative": int((1 - y_h).sum()),
				"positive_pct": float(y_h.mean() * 100),
			}
		)
	horizon_table = pd.DataFrame(horizon_rows)

	y12 = ((train["time_to_hit_hours"] <= 12) & (train["event"] == 1)).astype(int)
	y24 = ((train["time_to_hit_hours"] <= 24) & (train["event"] == 1)).astype(int)
	y48 = ((train["time_to_hit_hours"] <= 48) & (train["event"] == 1)).astype(int)
	y72 = ((train["time_to_hit_hours"] <= 72) & (train["event"] == 1)).astype(int)
	monotonic_violations = ((y12 > y24) | (y24 > y48) | (y48 > y72)).sum()

	checks = pd.DataFrame(
		[
			{
				"metric": "event_rate_pct",
				"value": float(train["event"].mean() * 100),
			},
			{
				"metric": "censor_rate_pct",
				"value": float((1 - train["event"]).mean() * 100),
			},
			{
				"metric": "time_min",
				"value": float(train["time_to_hit_hours"].min()),
			},
			{
				"metric": "time_max",
				"value": float(train["time_to_hit_hours"].max()),
			},
			{
				"metric": "time_outside_0_72",
				"value": int(((train["time_to_hit_hours"] < 0) | (train["time_to_hit_hours"] > 72)).sum()),
			},
			{
				"metric": "monotonic_label_violations",
				"value": int(monotonic_violations),
			},
		]
	)

	return event_table, horizon_table, checks


def subgroup_horizon_prevalence(train: pd.DataFrame) -> pd.DataFrame:
	temp = train.copy()
	temp["perimeter_band"] = pd.cut(
		temp["num_perimeters_0_5h"],
		bins=[-np.inf, 1, 3, 6, np.inf],
		labels=["<=1", "2-3", "4-6", "7+"],
	)

	rows = []
	for subgroup in ["low_temporal_resolution_0_5h", "perimeter_band", "event_start_month", "event_start_hour"]:
		for key, chunk in temp.groupby(subgroup, observed=False):
			for h in HORIZONS:
				y_h = ((chunk["time_to_hit_hours"] <= h) & (chunk["event"] == 1)).astype(int)
				rows.append(
					{
						"subgroup": subgroup,
						"group_value": str(key),
						"horizon": h,
						"n": int(len(chunk)),
						"positive": int(y_h.sum()),
						"positive_pct": float(y_h.mean() * 100),
					}
				)
	return pd.DataFrame(rows)


def _psi(train_series: pd.Series, test_series: pd.Series, bins: int = 10) -> float:
	train_values = train_series.dropna().values
	test_values = test_series.dropna().values
	if len(train_values) < 5 or len(test_values) < 5:
		return np.nan

	quantiles = np.linspace(0, 1, bins + 1)
	edges = np.quantile(train_values, quantiles)
	edges = np.unique(edges)
	if len(edges) < 3:
		return np.nan

	train_counts, _ = np.histogram(train_values, bins=edges)
	test_counts, _ = np.histogram(test_values, bins=edges)
	train_pct = train_counts / max(train_counts.sum(), 1)
	test_pct = test_counts / max(test_counts.sum(), 1)

	train_pct = np.clip(train_pct, EPS, 1)
	test_pct = np.clip(test_pct, EPS, 1)
	return float(np.sum((train_pct - test_pct) * np.log(train_pct / test_pct)))


def drift_analysis(train: pd.DataFrame, test: pd.DataFrame, numeric_cols: Iterable[str]) -> pd.DataFrame:
	rows = []
	for col in numeric_cols:
		train_col = train[col]
		test_col = test[col]

		train_mean = float(train_col.mean())
		test_mean = float(test_col.mean())
		train_std = float(train_col.std(ddof=1))
		test_std = float(test_col.std(ddof=1))

		pooled_std = max(np.sqrt((train_std ** 2 + test_std ** 2) / 2), EPS)
		mean_diff_z = (test_mean - train_mean) / pooled_std

		if HAS_SCIPY:
			ks_stat, ks_p = ks_2samp(train_col.dropna(), test_col.dropna())
		else:
			ks_stat, ks_p = np.nan, np.nan

		rows.append(
			{
				"column": col,
				"train_mean": train_mean,
				"test_mean": test_mean,
				"train_std": train_std,
				"test_std": test_std,
				"mean_diff_z": float(mean_diff_z),
				"median_diff": float(test_col.median() - train_col.median()),
				"q95_diff": float(test_col.quantile(0.95) - train_col.quantile(0.95)),
				"ks_stat": float(ks_stat),
				"ks_pvalue": float(ks_p),
				"psi": _psi(train_col, test_col),
			}
		)

	drift = pd.DataFrame(rows)
	drift["psi_flag"] = pd.cut(
		drift["psi"],
		bins=[-np.inf, 0.1, 0.25, np.inf],
		labels=["low", "moderate", "high"],
	)
	drift = drift.sort_values(["psi", "ks_stat"], ascending=False)
	return drift


def association_analysis(train: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
	rows = []
	for col in numeric_cols:
		for target_col in ["event", "time_to_hit_hours", "y_12", "y_24", "y_48", "y_72"]:
			if target_col not in train.columns:
				continue
			corr = train[col].corr(train[target_col], method="spearman")
			rows.append(
				{
					"feature": col,
					"target": target_col,
					"spearman_corr": float(corr) if pd.notna(corr) else np.nan,
				}
			)
	out = pd.DataFrame(rows)
	out["abs_spearman_corr"] = out["spearman_corr"].abs()
	return out.sort_values("abs_spearman_corr", ascending=False)


def redundancy_analysis(train: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
	corr = train[numeric_cols].corr(method="spearman")
	pairs = []
	cols = corr.columns.tolist()
	for i in range(len(cols)):
		for j in range(i + 1, len(cols)):
			value = corr.iloc[i, j]
			if pd.notna(value) and abs(value) >= 0.95:
				pairs.append(
					{
						"feature_1": cols[i],
						"feature_2": cols[j],
						"spearman_corr": float(value),
						"abs_spearman_corr": float(abs(value)),
					}
				)

	deterministic_checks = []
	if {"relative_growth_0_5h", "area_growth_rel_0_5h"}.issubset(train.columns):
		diff = train["relative_growth_0_5h"] - train["area_growth_rel_0_5h"]
		deterministic_checks.append(
			{
				"check": "relative_growth_vs_area_growth_rel",
				"mean_abs_diff": float(np.nanmean(np.abs(diff))),
				"max_abs_diff": float(np.nanmax(np.abs(diff))),
			}
		)
	if {"projected_advance_m", "dist_change_ci_0_5h"}.issubset(train.columns):
		diff = train["projected_advance_m"] + train["dist_change_ci_0_5h"]
		deterministic_checks.append(
			{
				"check": "projected_advance_vs_negative_dist_change",
				"mean_abs_diff": float(np.nanmean(np.abs(diff))),
				"max_abs_diff": float(np.nanmax(np.abs(diff))),
			}
		)
	if {"closing_speed_abs_m_per_h", "closing_speed_m_per_h"}.issubset(train.columns):
		diff = train["closing_speed_abs_m_per_h"] - train["closing_speed_m_per_h"].abs()
		deterministic_checks.append(
			{
				"check": "closing_speed_abs_consistency",
				"mean_abs_diff": float(np.nanmean(np.abs(diff))),
				"max_abs_diff": float(np.nanmax(np.abs(diff))),
			}
		)
	if {"spread_bearing_sin", "spread_bearing_cos"}.issubset(train.columns):
		norm_diff = (train["spread_bearing_sin"] ** 2 + train["spread_bearing_cos"] ** 2) - 1
		deterministic_checks.append(
			{
				"check": "bearing_sin_cos_unit_circle",
				"mean_abs_diff": float(np.nanmean(np.abs(norm_diff))),
				"max_abs_diff": float(np.nanmax(np.abs(norm_diff))),
			}
		)

	return pd.DataFrame(pairs).sort_values("abs_spearman_corr", ascending=False), pd.DataFrame(deterministic_checks)


def save_table(df: pd.DataFrame, path: Path) -> None:
	df.to_csv(path, index=False)


def plot_event_distribution(event_table: pd.DataFrame, figures_dir: Path) -> None:
	if not HAS_MATPLOTLIB:
		return
	plt.figure(figsize=(7, 5))
	plt.bar(event_table["event"].astype(str), event_table["count"], color=["#1f77b4", "#d62728"])
	plt.title("Event Distribution (0 = censored, 1 = hit)")
	plt.xlabel("event")
	plt.ylabel("count")
	plt.tight_layout()
	plt.savefig(figures_dir / "target_event_distribution.png", dpi=200)
	plt.close()


def plot_time_distribution(train: pd.DataFrame, figures_dir: Path) -> None:
	if not HAS_MATPLOTLIB:
		return
	plt.figure(figsize=(9, 6))
	if HAS_SEABORN:
		sns.histplot(data=train, x="time_to_hit_hours", hue="event", bins=30, kde=True, multiple="layer")
	else:
		for event_val in [0, 1]:
			subset = train.loc[train["event"] == event_val, "time_to_hit_hours"]
			plt.hist(subset, bins=30, alpha=0.55, label=f"event={event_val}")
		plt.legend()
	plt.title("time_to_hit_hours Distribution by Event")
	plt.tight_layout()
	plt.savefig(figures_dir / "target_time_distribution_by_event.png", dpi=200)
	plt.close()


def plot_horizon_prevalence(horizon_table: pd.DataFrame, figures_dir: Path) -> None:
	if not HAS_MATPLOTLIB:
		return
	plt.figure(figsize=(8, 5))
	plt.plot(horizon_table["horizon"], horizon_table["positive_pct"], marker="o")
	plt.title("Positive Class Prevalence by Horizon")
	plt.xlabel("Horizon (hours)")
	plt.ylabel("Positive %")
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig(figures_dir / "target_horizon_prevalence.png", dpi=200)
	plt.close()


def plot_feature_distributions(
	train: pd.DataFrame,
	test: pd.DataFrame,
	numeric_cols: List[str],
	figures_dir: Path,
	prefix: str,
	max_features: int | None = None,
) -> None:
	if not HAS_MATPLOTLIB:
		return
	cols = numeric_cols if max_features is None else numeric_cols[:max_features]
	for col in cols:
		fig, axes = plt.subplots(1, 2, figsize=(14, 5))
		if HAS_SEABORN:
			sns.histplot(train[col], bins=30, kde=True, ax=axes[0], color="#1f77b4")
		else:
			axes[0].hist(train[col].dropna(), bins=30, color="#1f77b4", alpha=0.75)
		axes[0].set_title(f"Train Distribution: {col}")

		if HAS_SEABORN:
			sns.histplot(train[col], bins=30, color="#1f77b4", alpha=0.45, stat="density", ax=axes[1], label="train")
			sns.histplot(test[col], bins=30, color="#d62728", alpha=0.45, stat="density", ax=axes[1], label="test")
			axes[1].legend()
		else:
			axes[1].hist(train[col].dropna(), bins=30, alpha=0.5, label="train")
			axes[1].hist(test[col].dropna(), bins=30, alpha=0.5, label="test")
			axes[1].legend()
		axes[1].set_title(f"Train vs Test: {col}")
		plt.tight_layout()
		plt.savefig(figures_dir / f"{prefix}_dist_{col}.png", dpi=180)
		plt.close(fig)

		if train[col].dropna().skew() > 2 and (train[col] > 0).all():
			plt.figure(figsize=(8, 5))
			transformed = np.log1p(train[col].clip(lower=0))
			if HAS_SEABORN:
				sns.histplot(transformed, bins=30, kde=True)
			else:
				plt.hist(transformed.dropna(), bins=30)
			plt.title(f"Log1p Train Distribution: {col}")
			plt.tight_layout()
			plt.savefig(figures_dir / f"{prefix}_logdist_{col}.png", dpi=180)
			plt.close()


def plot_correlation_heatmap(train: pd.DataFrame, numeric_cols: List[str], figures_dir: Path, name: str) -> None:
	if not HAS_MATPLOTLIB:
		return
	corr = train[numeric_cols].corr(method="spearman")
	plt.figure(figsize=(18, 14))
	if HAS_SEABORN:
		sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1)
	else:
		plt.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
		plt.colorbar()
	plt.title(f"Spearman Correlation Heatmap ({name})")
	plt.tight_layout()
	plt.savefig(figures_dir / f"correlation_heatmap_{name}.png", dpi=220)
	plt.close()


def plot_top_drift(drift: pd.DataFrame, figures_dir: Path, top_n: int = 25) -> None:
	if not HAS_MATPLOTLIB:
		return
	top = drift.head(top_n).copy().sort_values("psi", ascending=True)
	plt.figure(figsize=(10, max(8, int(0.35 * top_n))))
	plt.barh(top["column"], top["psi"], color="#9467bd")
	plt.title(f"Top {top_n} Features by PSI Drift")
	plt.xlabel("PSI")
	plt.tight_layout()
	plt.savefig(figures_dir / "drift_top_features_psi.png", dpi=200)
	plt.close()


def run_survival_analysis(train: pd.DataFrame, figures_dir: Path) -> pd.DataFrame:
	if not HAS_MATPLOTLIB:
		return pd.DataFrame(
			[
				{
					"analysis": "matplotlib_available",
					"status": "no",
					"details": "matplotlib not installed; skipping survival plotting",
				}
			]
		)
	if not HAS_LIFELINES:
		return pd.DataFrame(
			[
				{
					"analysis": "lifelines_available",
					"status": "no",
					"details": "lifelines not installed; skipping Kaplan-Meier plots and log-rank tests",
				}
			]
		)

	kmf = KaplanMeierFitter()
	summary_rows = []

	plt.figure(figsize=(9, 6))
	kmf.fit(train["time_to_hit_hours"], event_observed=train["event"], label="Overall survival")
	kmf.plot_survival_function()
	plt.title("Kaplan-Meier Survival Curve (Overall)")
	plt.xlabel("Hours since t0+5h")
	plt.ylabel("Survival probability (no hit yet)")
	plt.tight_layout()
	plt.savefig(figures_dir / "survival_km_overall.png", dpi=220)
	plt.close()
	summary_rows.append(
		{
			"analysis": "km_overall",
			"status": "ok",
			"details": f"median_survival={kmf.median_survival_time_}",
		}
	)

	strat_features = {
		"dist_min_ci_bin": pd.qcut(train["dist_min_ci_0_5h"], q=3, duplicates="drop").astype(str),
		"alignment_abs_bin": pd.qcut(train["alignment_abs"], q=3, duplicates="drop").astype(str),
		"num_perimeters_bin": pd.cut(
			train["num_perimeters_0_5h"],
			bins=[-np.inf, 1, 3, 6, np.inf],
			labels=["<=1", "2-3", "4-6", "7+"],
		).astype(str),
	}

	for name, strata in strat_features.items():
		temp = train.copy()
		temp[name] = strata
		plt.figure(figsize=(10, 7))
		valid_groups = [g for g, c in temp[name].value_counts().items() if c >= 8 and g != "nan"]

		for group in valid_groups:
			chunk = temp[temp[name] == group]
			kmf.fit(chunk["time_to_hit_hours"], event_observed=chunk["event"], label=str(group))
			kmf.plot_survival_function(ci_show=False)
		plt.title(f"KM Curves by {name}")
		plt.xlabel("Hours")
		plt.ylabel("Survival probability")
		plt.tight_layout()
		plt.savefig(figures_dir / f"survival_km_{name}.png", dpi=220)
		plt.close()

		if len(valid_groups) >= 2:
			g1 = temp[temp[name] == valid_groups[0]]
			g2 = temp[temp[name] == valid_groups[1]]
			lr = logrank_test(g1["time_to_hit_hours"], g2["time_to_hit_hours"], g1["event"], g2["event"])
			details = f"first_two_groups={valid_groups[0]} vs {valid_groups[1]}, p={lr.p_value:.6g}"
		else:
			details = "insufficient group count for log-rank"

		summary_rows.append({"analysis": f"km_{name}", "status": "ok", "details": details})

	return pd.DataFrame(summary_rows)


def export_json_report(path: Path, payload: Dict[str, object]) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, default=str)


def run_eda(output_dir: Path, max_feature_plots: int | None = None, skip_plots: bool = False) -> None:
	print("=== WiDS 2026: Comprehensive Survival-Aware EDA ===")
	print(f"Data directory: {DATA_DIR}")
	print(f"Output directory: {output_dir}")

	setup_plot_style()
	if not HAS_MATPLOTLIB and not skip_plots:
		print("matplotlib not available: forcing --skip-plots mode")
		skip_plots = True
	dirs = make_output_dirs(output_dir)

	train, test, metadata, sample_submission = load_data()
	train = add_horizon_labels(train)

	train_enh = build_engineered_features(train)
	test_enh = build_engineered_features(test)
	engineered_cols = sorted(set(train_enh.columns) - set(train.columns))

	print("[1/7] Running schema validation and dataset profiling...")
	schema_df, schema_summary = validate_schema(train, test, metadata, sample_submission)
	data_profile_df = pd.concat([describe_dataset(train, "train"), describe_dataset(test, "test")], ignore_index=True)

	print("[2/7] Running quality audits (raw + engineered)...")
	quality_train_raw = quality_audit(train, "train_raw")
	quality_test_raw = quality_audit(test, "test_raw")
	quality_train_eng = quality_audit(train_enh[engineered_cols], "train_engineered")
	quality_test_eng = quality_audit(test_enh[engineered_cols], "test_engineered")

	print("[3/7] Running target, horizon, and subgroup analyses...")
	event_table, horizon_table, target_checks = target_audit(train)
	subgroup_table = subgroup_horizon_prevalence(train)

	print("[4/7] Running association and redundancy analysis...")
	raw_numeric_cols = [
		c for c in train.columns
		if pd.api.types.is_numeric_dtype(train[c]) and c not in ["event_id"]
	]
	eng_numeric_cols = [c for c in engineered_cols if pd.api.types.is_numeric_dtype(train_enh[c])]

	assoc_raw = association_analysis(train, raw_numeric_cols)
	assoc_eng = association_analysis(train_enh.assign(**{k: train[k] for k in ["event", "time_to_hit_hours", "y_12", "y_24", "y_48", "y_72"]}), eng_numeric_cols)

	high_corr_pairs_raw, deterministic_checks_raw = redundancy_analysis(train, raw_numeric_cols)
	high_corr_pairs_eng, deterministic_checks_eng = redundancy_analysis(train_enh, eng_numeric_cols)

	print("[5/7] Running train-vs-test drift analysis...")
	drift_raw = drift_analysis(train, test, [c for c in raw_numeric_cols if c in test.columns])
	drift_eng = drift_analysis(train_enh, test_enh, eng_numeric_cols)

	print("[6/7] Running survival analyses...")
	survival_summary = run_survival_analysis(train, dirs["figures"])

	print("[7/7] Exporting tables, figures, and JSON report...")
	save_table(schema_df, dirs["tables"] / "schema_validation.csv")
	save_table(data_profile_df, dirs["tables"] / "dataset_profile.csv")
	save_table(quality_train_raw, dirs["tables"] / "quality_train_raw.csv")
	save_table(quality_test_raw, dirs["tables"] / "quality_test_raw.csv")
	save_table(quality_train_eng, dirs["tables"] / "quality_train_engineered.csv")
	save_table(quality_test_eng, dirs["tables"] / "quality_test_engineered.csv")
	save_table(event_table, dirs["tables"] / "target_event_distribution.csv")
	save_table(horizon_table, dirs["tables"] / "target_horizon_prevalence.csv")
	save_table(target_checks, dirs["tables"] / "target_consistency_checks.csv")
	save_table(subgroup_table, dirs["tables"] / "subgroup_horizon_prevalence.csv")
	save_table(assoc_raw, dirs["tables"] / "association_raw_features.csv")
	save_table(assoc_eng, dirs["tables"] / "association_engineered_features.csv")
	save_table(high_corr_pairs_raw, dirs["tables"] / "redundancy_high_corr_pairs_raw.csv")
	save_table(high_corr_pairs_eng, dirs["tables"] / "redundancy_high_corr_pairs_engineered.csv")
	save_table(deterministic_checks_raw, dirs["tables"] / "deterministic_checks_raw.csv")
	save_table(deterministic_checks_eng, dirs["tables"] / "deterministic_checks_engineered.csv")
	save_table(drift_raw, dirs["tables"] / "drift_raw_features.csv")
	save_table(drift_eng, dirs["tables"] / "drift_engineered_features.csv")
	save_table(survival_summary, dirs["tables"] / "survival_summary.csv")

	if not skip_plots:
		plot_event_distribution(event_table, dirs["figures"])
		plot_time_distribution(train, dirs["figures"])
		plot_horizon_prevalence(horizon_table, dirs["figures"])

		plot_feature_distributions(
			train,
			test,
			[c for c in raw_numeric_cols if c in test.columns],
			dirs["figures"],
			prefix="raw",
			max_features=max_feature_plots,
		)
		plot_feature_distributions(
			train_enh,
			test_enh,
			eng_numeric_cols,
			dirs["figures"],
			prefix="engineered",
			max_features=max_feature_plots,
		)
		plot_correlation_heatmap(train, [c for c in raw_numeric_cols if c not in ["event_id"]], dirs["figures"], "raw")
		plot_correlation_heatmap(train_enh, eng_numeric_cols, dirs["figures"], "engineered")
		plot_top_drift(drift_raw, dirs["figures"], top_n=25)
		plot_top_drift(drift_eng, dirs["figures"], top_n=25)

	summary_payload = {
		"script": "eda.py",
		"dataset": {
			"train_rows": int(train.shape[0]),
			"test_rows": int(test.shape[0]),
			"raw_feature_count": int(len(schema_summary["feature_cols"])),
			"engineered_feature_count": int(len(eng_numeric_cols)),
		},
		"target": {
			"event_rate": float(train["event"].mean()),
			"censor_rate": float((1 - train["event"]).mean()),
			"time_min": float(train["time_to_hit_hours"].min()),
			"time_max": float(train["time_to_hit_hours"].max()),
		},
		"schema_ok": bool(schema_df["ok"].all()),
		"tools": {
			"seaborn": HAS_SEABORN,
			"scipy": HAS_SCIPY,
			"lifelines": HAS_LIFELINES,
		},
		"outputs": {
			"tables_dir": str(dirs["tables"]),
			"figures_dir": str(dirs["figures"]),
		},
	}
	export_json_report(dirs["json"] / "eda_summary.json", summary_payload)

	print("=== EDA Completed Successfully ===")
	print(f"Train rows: {train.shape[0]}, Test rows: {test.shape[0]}")
	print(f"Raw features (metadata): {len(schema_summary['feature_cols'])}")
	print(f"Engineered numeric features (from hi.py-inspired transforms): {len(eng_numeric_cols)}")
	print(f"Event rate: {train['event'].mean():.4f}, Censor rate: {(1 - train['event']).mean():.4f}")
	print(f"Tables saved to: {dirs['tables']}")
	print(f"Figures saved to: {dirs['figures']}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Comprehensive EDA for WiDS Global Datathon 2026")
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(ROOT / "eda_outputs"),
		help="Directory where all EDA artifacts are written",
	)
	parser.add_argument(
		"--max-feature-plots",
		type=int,
		default=None,
		help="Optional cap on number of features for per-feature distribution plots",
	)
	parser.add_argument(
		"--skip-plots",
		action="store_true",
		help="Skip all plotting and only produce tables/json",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	run_eda(
		output_dir=Path(args.output_dir),
		max_feature_plots=args.max_feature_plots,
		skip_plots=args.skip_plots,
	)


if __name__ == "__main__":
	main()

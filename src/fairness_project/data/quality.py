"""Deterministic data-semantics audits for the UCI Adult benchmark.

This module inspects evidence quality without cleaning, imputing, or inferring
protected attributes. Missing protected values remain explicit ``null`` groups
in the JSON-ready output. ``fnlwgt`` is enforced as an audit-only sampling
weight and is never permitted in the predictive feature identity.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from itertools import combinations
from numbers import Integral
from typing import Any

import pandas as pd

from fairness_project.data.download import COLUMN_NAMES
from fairness_project.data.schema import (
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    SAMPLE_WEIGHT_COLUMN,
    VALID_INCOME,
    VALID_SEX,
    VALID_SPLIT,
)

QUALITY_SCHEMA_VERSION = "1.0"
DEFAULT_MISSING_MARKERS = ("?", "")
PROTECTED_DIMENSIONS = (("sex",), ("race",), ("sex", "race"))
_TARGET_COLUMN = "income"
_SPLIT_COLUMN = "split"
_RAW_SPLIT_COLUMN = "_audit_input_split"
_PROTECTED_COLUMNS = frozenset({"sex", "race", "race_binary"})

__all__ = [
    "DataQualityValidationError",
    "audit_processed_quality",
    "audit_raw_attrition",
]


class DataQualityValidationError(ValueError):
    """Raised when an audit input or audit configuration is structurally invalid."""


def _column_list(value: Sequence[str], name: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DataQualityValidationError(f"{name} must be a sequence of column names")
    columns = list(value)
    if not columns:
        raise DataQualityValidationError(f"{name} must contain at least one column")
    if any(not isinstance(column, str) or not column.strip() for column in columns):
        raise DataQualityValidationError(f"{name} must contain non-empty strings")
    if len(set(columns)) != len(columns):
        raise DataQualityValidationError(f"{name} must not contain duplicate columns")
    return columns


def _markers(value: Sequence[str]) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DataQualityValidationError("missing_markers must be a sequence of strings")
    markers: list[str] = []
    for marker in value:
        if not isinstance(marker, str):
            raise DataQualityValidationError("missing_markers must contain only strings")
        markers.append(marker.strip())
    if len(set(markers)) != len(markers):
        raise DataQualityValidationError(
            "missing_markers must remain unique after whitespace normalization"
        )
    return sorted(markers)


def _small_group_threshold(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise DataQualityValidationError("small_group_threshold must be a positive integer")
    return int(value)


def _validate_frame(frame: pd.DataFrame, name: str, required: set[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise DataQualityValidationError(f"{name} must be a pandas DataFrame")
    if frame.empty:
        raise DataQualityValidationError(f"{name} must contain at least one row")
    columns = list(frame.columns)
    if any(not isinstance(column, str) or not column for column in columns):
        raise DataQualityValidationError(f"{name} must use non-empty string column names")
    if len(set(columns)) != len(columns):
        raise DataQualityValidationError(f"{name} must not contain duplicate column names")
    missing = sorted(required - set(columns))
    if missing:
        raise DataQualityValidationError(f"{name} is missing required columns: {missing}")


def _validate_feature_contract(feature_columns: list[str]) -> None:
    forbidden = sorted(
        set(feature_columns)
        & ({_TARGET_COLUMN, _SPLIT_COLUMN, SAMPLE_WEIGHT_COLUMN} | _PROTECTED_COLUMNS)
    )
    if forbidden:
        raise DataQualityValidationError(
            "feature_columns may not contain targets, split fields, protected attributes, "
            f"or the audit-only sampling weight: {forbidden}"
        )


def _is_missing(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError) as exc:
        raise DataQualityValidationError("DataFrame cells must contain scalar values") from exc
    if not isinstance(missing, bool):
        try:
            return bool(missing)
        except (TypeError, ValueError) as exc:
            raise DataQualityValidationError("DataFrame cells must contain scalar values") from exc
    return missing


def _normalize_for_audit(
    frame: pd.DataFrame,
    columns: Sequence[str],
    missing_markers: Sequence[str],
    *,
    normalize_adult_target: bool,
) -> pd.DataFrame:
    """Return an audit-only normalized copy; never mutate or impute the source."""
    result = frame.copy(deep=True)
    marker_set = set(missing_markers)

    def normalize(value: Any) -> Any:
        if _is_missing(value):
            return pd.NA
        if isinstance(value, str):
            stripped = value.strip()
            if stripped in marker_set:
                return pd.NA
            return stripped
        return value

    for column in columns:
        result[column] = result[column].map(normalize)

    if normalize_adult_target:
        result[_TARGET_COLUMN] = result[_TARGET_COLUMN].replace(
            {">50K.": ">50K", "<=50K.": "<=50K"}
        )
    return result


def _validate_adult_values(frame: pd.DataFrame, name: str, *, require_split: bool) -> None:
    for column in ("sex", "race", _TARGET_COLUMN):
        invalid_types = [
            value
            for value in frame[column].dropna().tolist()
            if not isinstance(value, str) or not value
        ]
        if invalid_types:
            raise DataQualityValidationError(
                f"{name}.{column} must contain non-empty strings or missing values"
            )

    invalid_sex = sorted(set(frame["sex"].dropna()) - VALID_SEX)
    if invalid_sex:
        raise DataQualityValidationError(f"{name}.sex contains invalid values: {invalid_sex}")
    invalid_income = sorted(set(frame[_TARGET_COLUMN].dropna()) - VALID_INCOME)
    if invalid_income:
        raise DataQualityValidationError(
            f"{name}.{_TARGET_COLUMN} contains invalid values: {invalid_income}"
        )

    if require_split:
        if frame[_SPLIT_COLUMN].isna().any():
            raise DataQualityValidationError(f"{name}.split must not contain missing values")
        invalid_split_types = [
            value for value in frame[_SPLIT_COLUMN].tolist() if not isinstance(value, str)
        ]
        if invalid_split_types:
            raise DataQualityValidationError(f"{name}.split must contain strings")
        invalid_splits = sorted(set(frame[_SPLIT_COLUMN]) - VALID_SPLIT)
        if invalid_splits:
            raise DataQualityValidationError(
                f"{name}.split contains invalid values: {invalid_splits}"
            )


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _attrition_counts(input_rows: int, complete_case_rows: int) -> dict[str, int | float | None]:
    deleted_rows = input_rows - complete_case_rows
    return {
        "input_rows": int(input_rows),
        "complete_case_rows": int(complete_case_rows),
        "deleted_rows": int(deleted_rows),
        "deletion_rate": _rate(deleted_rows, input_rows),
    }


GroupKey = tuple[str | None, ...]


def _group_sort_key(key: GroupKey) -> tuple[tuple[int, str], ...]:
    return tuple((0, "") if value is None else (1, value) for value in key)


def _group_positions(frame: pd.DataFrame, dimensions: Sequence[str]) -> dict[GroupKey, list[int]]:
    positions: dict[GroupKey, list[int]] = {}
    for position, row in enumerate(frame[list(dimensions)].itertuples(index=False, name=None)):
        key = tuple(None if _is_missing(value) else str(value) for value in row)
        positions.setdefault(key, []).append(position)
    return positions


def _attributes(dimensions: Sequence[str], key: GroupKey) -> dict[str, str | None]:
    return dict(zip(dimensions, key, strict=True))


def _group_attrition(
    before: pd.DataFrame,
    after: pd.DataFrame,
    dimensions: Sequence[str],
) -> list[dict[str, Any]]:
    before_positions = _group_positions(before, dimensions)
    after_positions = _group_positions(after, dimensions)
    keys = sorted(set(before_positions) | set(after_positions), key=_group_sort_key)
    records: list[dict[str, Any]] = []
    for key in keys:
        record: dict[str, Any] = {"attributes": _attributes(dimensions, key)}
        record.update(
            _attrition_counts(
                len(before_positions.get(key, [])),
                len(after_positions.get(key, [])),
            )
        )
        records.append(record)
    return records


def _composition(
    attrition: Sequence[dict[str, Any]],
    input_total: int,
    complete_case_total: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in attrition:
        input_rows = int(row["input_rows"])
        complete_case_rows = int(row["complete_case_rows"])
        before_share = _rate(input_rows, input_total)
        after_share = _rate(complete_case_rows, complete_case_total)
        result.append(
            {
                "attributes": dict(row["attributes"]),
                "before": {"count": input_rows, "share": before_share},
                "after": {"count": complete_case_rows, "share": after_share},
                "share_change": (
                    None
                    if before_share is None or after_share is None
                    else after_share - before_share
                ),
            }
        )
    return result


def _missingness(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    row_count = len(frame)
    return [
        {
            "column": column,
            "missing_count": int(frame[column].isna().sum()),
            "missing_rate": _rate(int(frame[column].isna().sum()), row_count),
        }
        for column in columns
    ]


def _group_missingness(
    frame: pd.DataFrame,
    dimensions: Sequence[str],
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    groups = _group_positions(frame, dimensions)
    records: list[dict[str, Any]] = []
    for key in sorted(groups, key=_group_sort_key):
        subset = frame.iloc[groups[key]]
        records.append(
            {
                "attributes": _attributes(dimensions, key),
                "row_count": len(subset),
                "columns": _missingness(subset, columns),
            }
        )
    return records


def _duplicate_summary(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    group_sizes = frame.groupby(list(columns), dropna=False, sort=False).size()
    duplicate_sizes = group_sizes[group_sizes > 1]
    return {
        "identity_columns": list(columns),
        "duplicate_groups": int(len(duplicate_sizes)),
        "rows_in_duplicate_groups": int(duplicate_sizes.sum()),
        "duplicate_rows_beyond_first": int((duplicate_sizes - 1).sum()),
    }


def _conflicting_labels(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> dict[str, Any]:
    missing_identity = frame[list(feature_columns)].isna().any(axis=1)
    missing_target = frame[target_column].isna()
    eligible = frame.loc[~missing_identity & ~missing_target]
    grouped = eligible.groupby(list(feature_columns), dropna=False, sort=False)[target_column]
    sizes = grouped.size()
    label_counts = grouped.nunique(dropna=True)
    conflicting = label_counts[label_counts > 1].index
    conflicting_sizes = sizes.loc[conflicting] if len(conflicting) else sizes.iloc[0:0]
    return {
        "feature_identity_columns": list(feature_columns),
        "conflicting_feature_groups": int(len(conflicting_sizes)),
        "rows_in_conflicting_groups": int(conflicting_sizes.sum()),
        "rows_excluded_for_missing_feature_identity": int(missing_identity.sum()),
        "rows_excluded_for_missing_target": int((~missing_identity & missing_target).sum()),
    }


def _cross_split_duplicates(
    frame: pd.DataFrame,
    split_column: str,
    feature_columns: Sequence[str],
    target_column: str,
) -> dict[str, Any]:
    split_values = sorted(str(value) for value in frame[split_column].dropna().unique())
    pairs: list[dict[str, Any]] = []
    total_overlaps = 0
    for left_name, right_name in combinations(split_values, 2):
        left_all = frame.loc[frame[split_column] == left_name]
        right_all = frame.loc[frame[split_column] == right_name]
        left_missing = left_all[list(feature_columns)].isna().any(axis=1)
        right_missing = right_all[list(feature_columns)].isna().any(axis=1)
        left = left_all.loc[~left_missing]
        right = right_all.loc[~right_missing]
        left_keys = left[list(feature_columns)].drop_duplicates()
        right_keys = right[list(feature_columns)].drop_duplicates()
        overlap = left_keys.merge(right_keys, on=list(feature_columns), how="inner")
        overlap_count = len(overlap)
        total_overlaps += overlap_count

        if overlap_count:
            left_overlap = left.merge(overlap, on=list(feature_columns), how="inner")
            right_overlap = right.merge(overlap, on=list(feature_columns), how="inner")
            labels = pd.concat(
                [
                    left_overlap[[*feature_columns, target_column]],
                    right_overlap[[*feature_columns, target_column]],
                ],
                ignore_index=True,
            ).dropna(subset=[target_column])
            conflict_count = int(
                (
                    labels.groupby(list(feature_columns), dropna=False, sort=False)[target_column]
                    .nunique(dropna=True)
                    .gt(1)
                ).sum()
            )
        else:
            left_overlap = left.iloc[0:0]
            right_overlap = right.iloc[0:0]
            conflict_count = 0

        pairs.append(
            {
                "left_split": left_name,
                "right_split": right_name,
                "overlap_feature_groups": int(overlap_count),
                "left_rows_in_overlap": int(len(left_overlap)),
                "right_rows_in_overlap": int(len(right_overlap)),
                "conflicting_label_feature_groups": conflict_count,
                "left_rows_excluded_for_missing_feature_identity": int(left_missing.sum()),
                "right_rows_excluded_for_missing_feature_identity": int(right_missing.sum()),
            }
        )
    return {
        "feature_identity_columns": list(feature_columns),
        "pairwise_overlap_feature_groups": int(total_overlaps),
        "pairs": pairs,
    }


def _weight_semantics(frame: pd.DataFrame, feature_columns: Sequence[str]) -> dict[str, Any]:
    raw_weight = frame[SAMPLE_WEIGHT_COLUMN]
    numeric = pd.to_numeric(raw_weight, errors="coerce")
    non_numeric = raw_weight.notna() & numeric.isna()
    finite = numeric.map(lambda value: False if _is_missing(value) else math.isfinite(float(value)))
    non_finite = numeric.notna() & ~finite
    usable = numeric.notna() & finite
    usable_values = numeric.loc[usable].astype(float)
    return {
        "column": SAMPLE_WEIGHT_COLUMN,
        "contract_role": "audit_only_sampling_weight",
        "semantics": (
            "Census final sampling weight retained only for descriptive audit and weighted "
            "sensitivity analysis. It is excluded from predictive features."
        ),
        "included_in_predictive_features": SAMPLE_WEIGHT_COLUMN in feature_columns,
        "non_missing_count": int(raw_weight.notna().sum()),
        "missing_count": int(raw_weight.isna().sum()),
        "non_numeric_count": int(non_numeric.sum()),
        "non_finite_count": int(non_finite.sum()),
        "non_positive_count": int((usable_values <= 0).sum()),
        "total_usable_weight": float(usable_values.sum()) if len(usable_values) else None,
    }


def _small_group_flags(
    attrition_by_dimension: dict[str, list[dict[str, Any]]],
    dimensions: dict[str, Sequence[str]],
    threshold: int,
    *,
    scope: str,
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for label, dimension in dimensions.items():
        for row in attrition_by_dimension[label]:
            for stage, count_field in (
                ("input", "input_rows"),
                ("complete_case", "complete_case_rows"),
            ):
                count = int(row[count_field])
                if count < threshold:
                    flags.append(
                        {
                            "code": "SMALL_GROUP_SUPPORT",
                            "evidence_status": "limited",
                            "scope": scope,
                            "stage": stage,
                            "dimensions": list(dimension),
                            "attributes": dict(row["attributes"]),
                            "count": count,
                            "threshold": threshold,
                        }
                    )
    return flags


def _split_small_group_flags(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    split_column: str,
    dimensions: dict[str, Sequence[str]],
    threshold: int,
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    split_values = sorted(str(value) for value in before[split_column].dropna().unique())
    for split_value in split_values:
        split_before = before.loc[before[split_column] == split_value]
        split_after = after.loc[after[split_column] == split_value]
        attrition = {
            label: _group_attrition(split_before, split_after, group_dimensions)
            for label, group_dimensions in dimensions.items()
        }
        split_flags = _small_group_flags(
            attrition,
            dimensions,
            threshold,
            scope="split",
        )
        for flag in split_flags:
            flag["split"] = split_value
        flags.extend(split_flags)
    return flags


def _quality_sections(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    complete_case_columns: Sequence[str],
    feature_columns: Sequence[str],
    split_dimension: tuple[str, ...],
    small_group_threshold: int,
) -> dict[str, Any]:
    dimension_map: dict[str, Sequence[str]] = {
        "by_sex": ("sex",),
        "by_race": ("race",),
        "by_sex_and_race": ("sex", "race"),
    }
    attrition_by_dimension = {
        label: _group_attrition(before, after, dimensions)
        for label, dimensions in dimension_map.items()
    }
    split_attrition = _group_attrition(before, after, split_dimension)
    overall = _attrition_counts(len(before), len(after))
    composition = {
        label: _composition(rows, len(before), len(after))
        for label, rows in attrition_by_dimension.items()
    }
    missingness = {
        "overall": _missingness(before, complete_case_columns),
        "by_split": _group_missingness(before, split_dimension, complete_case_columns),
        **{
            label: _group_missingness(before, dimensions, complete_case_columns)
            for label, dimensions in dimension_map.items()
        },
    }
    flags = _small_group_flags(
        attrition_by_dimension,
        dimension_map,
        small_group_threshold,
        scope="overall",
    )
    flags.extend(
        _split_small_group_flags(
            before,
            after,
            split_column=split_dimension[0],
            dimensions=dimension_map,
            threshold=small_group_threshold,
        )
    )
    return {
        "attrition": {
            "overall": overall,
            "by_split": split_attrition,
            **attrition_by_dimension,
        },
        "missingness": missingness,
        "group_composition": composition,
        "duplicates": {
            "exact_rows": _duplicate_summary(before, complete_case_columns),
            "predictive_feature_rows": _duplicate_summary(before, feature_columns),
            "conflicting_labels": _conflicting_labels(before, feature_columns, _TARGET_COLUMN),
            "cross_split_predictive_features": _cross_split_duplicates(
                before,
                split_dimension[0],
                feature_columns,
                _TARGET_COLUMN,
            ),
        },
        "fnlwgt": _weight_semantics(before, feature_columns),
        "evidence_flags": flags,
    }


def _assert_json_ready(result: dict[str, Any]) -> dict[str, Any]:
    try:
        json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:  # pragma: no cover - internal invariant
        raise RuntimeError("Data-quality audit produced non-JSON-ready output") from exc
    return result


def audit_raw_attrition(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    *,
    missing_markers: Sequence[str] = DEFAULT_MISSING_MARKERS,
    columns: Sequence[str] | None = None,
    feature_columns: Sequence[str] | None = None,
    small_group_threshold: int = 30,
) -> dict[str, Any]:
    """Audit missingness and complete-case deletion before Adult preprocessing.

    The function returns counts only and never returns or mutates input rows. A
    marker-normalized copy is used for analysis; protected values are neither
    filled nor derived. Missing protected values therefore appear as JSON
    ``null`` group attributes and are removed only if the configured
    complete-case contract requires that column.
    """
    complete_case_columns = _column_list(COLUMN_NAMES if columns is None else columns, "columns")
    predictive_features = _column_list(
        FEATURE_COLUMNS if feature_columns is None else feature_columns,
        "feature_columns",
    )
    _validate_feature_contract(predictive_features)
    markers = _markers(missing_markers)
    threshold = _small_group_threshold(small_group_threshold)
    required = (
        set(complete_case_columns)
        | set(predictive_features)
        | {
            "sex",
            "race",
            _TARGET_COLUMN,
            SAMPLE_WEIGHT_COLUMN,
        }
    )
    _validate_frame(raw_train, "raw_train", required)
    _validate_frame(raw_test, "raw_test", required)
    if _RAW_SPLIT_COLUMN in raw_train.columns or _RAW_SPLIT_COLUMN in raw_test.columns:
        raise DataQualityValidationError(
            f"Input frames may not contain reserved column {_RAW_SPLIT_COLUMN!r}"
        )

    normalized_columns = sorted(required)
    train = _normalize_for_audit(
        raw_train,
        normalized_columns,
        markers,
        normalize_adult_target=True,
    )
    test = _normalize_for_audit(
        raw_test,
        normalized_columns,
        markers,
        normalize_adult_target=True,
    )
    _validate_adult_values(train, "raw_train", require_split=False)
    _validate_adult_values(test, "raw_test", require_split=False)
    train[_RAW_SPLIT_COLUMN] = "train"
    test[_RAW_SPLIT_COLUMN] = "test"
    before = pd.concat([train, test], ignore_index=True)
    complete_mask = ~before[complete_case_columns].isna().any(axis=1)
    after = before.loc[complete_mask].copy()

    sections = _quality_sections(
        before,
        after,
        complete_case_columns=complete_case_columns,
        feature_columns=predictive_features,
        split_dimension=(_RAW_SPLIT_COLUMN,),
        small_group_threshold=threshold,
    )
    for row in sections["attrition"]["by_split"]:
        row["attributes"] = {"input_split": row["attributes"].pop(_RAW_SPLIT_COLUMN)}
    for row in sections["missingness"]["by_split"]:
        row["attributes"] = {"input_split": row["attributes"].pop(_RAW_SPLIT_COLUMN)}

    result = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "audit_type": "raw_complete_case_attrition",
        "configuration": {
            "complete_case_columns": complete_case_columns,
            "feature_identity_columns": predictive_features,
            "missing_markers": markers,
            "protected_dimensions": [list(value) for value in PROTECTED_DIMENSIONS],
            "small_group_threshold": threshold,
            "protected_value_policy": "observe_or_null_never_infer",
            "target_normalization": "trim_whitespace_and_known_adult_test_period",
        },
        **sections,
    }
    return _assert_json_ready(result)


def audit_processed_quality(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    feature_columns: Sequence[str] | None = None,
    small_group_threshold: int = 30,
) -> dict[str, Any]:
    """Audit model-ready Adult data without altering its values or splits.

    The audit verifies the structural contract, reports any residual
    incomplete-case attrition, and detects feature duplication and overlap
    across the observed train, validation, and test partitions.
    """
    complete_case_columns = _column_list(
        REQUIRED_COLUMNS if columns is None else columns, "columns"
    )
    predictive_features = _column_list(
        FEATURE_COLUMNS if feature_columns is None else feature_columns,
        "feature_columns",
    )
    _validate_feature_contract(predictive_features)
    threshold = _small_group_threshold(small_group_threshold)
    required = (
        set(complete_case_columns)
        | set(predictive_features)
        | {
            "sex",
            "race",
            _TARGET_COLUMN,
            _SPLIT_COLUMN,
            SAMPLE_WEIGHT_COLUMN,
        }
    )
    _validate_frame(frame, "frame", required)
    before = _normalize_for_audit(
        frame,
        sorted(required),
        (),
        normalize_adult_target=False,
    )
    _validate_adult_values(before, "frame", require_split=True)
    complete_mask = ~before[complete_case_columns].isna().any(axis=1)
    after = before.loc[complete_mask].copy()

    sections = _quality_sections(
        before,
        after,
        complete_case_columns=complete_case_columns,
        feature_columns=predictive_features,
        split_dimension=(_SPLIT_COLUMN,),
        small_group_threshold=threshold,
    )
    result = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "audit_type": "processed_data_quality",
        "configuration": {
            "complete_case_columns": complete_case_columns,
            "feature_identity_columns": predictive_features,
            "protected_dimensions": [list(value) for value in PROTECTED_DIMENSIONS],
            "small_group_threshold": threshold,
            "protected_value_policy": "observe_or_null_never_infer",
        },
        **sections,
    }
    return _assert_json_ready(result)

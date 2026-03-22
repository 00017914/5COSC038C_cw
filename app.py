import io
import json
import os
from copy import deepcopy
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="AI-Assisted Data Wrangler & Visualizer", layout="wide")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_state() -> None:
    defaults = {
        "original_df": None,
        "working_df": None,
        "history": [],
        "log": [],
        "source_name": None,
        "ai_chart_suggestion": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    for key in ["original_df", "working_df", "history", "log", "source_name", "ai_chart_suggestion"]:
        st.session_state[key] = None if key in {"original_df", "working_df", "source_name"} else []


def push_history() -> None:
    if st.session_state.working_df is not None:
        st.session_state.history.append(st.session_state.working_df.copy())


def record_step(step: str, params: dict, affected_columns) -> None:
    if isinstance(affected_columns, list):
        column_text = ", ".join(map(str, affected_columns)) if affected_columns else "ALL"
    else:
        column_text = str(affected_columns)
    st.session_state.log.append(
        {
            "step": step,
            "parameters": params,
            "affected_columns": column_text,
            "timestamp": now_iso(),
        }
    )


def set_loaded_dataframe(df: pd.DataFrame, source_name: str, step: str, params: dict) -> None:
    st.session_state.original_df = df.copy()
    st.session_state.working_df = df.copy()
    st.session_state.source_name = source_name
    st.session_state.history = []
    st.session_state.log = [
        {
            "step": step,
            "parameters": params,
            "affected_columns": "ALL",
            "timestamp": now_iso(),
        }
    ]


@st.cache_data(show_spinner=False)
def load_uploaded_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    suffix = file_name.lower().split(".")[-1]
    buffer = io.BytesIO(file_bytes)
    if suffix == "csv":
        df = pd.read_csv(buffer)
        return infer_better_dtypes(df)
    if suffix == "xlsx":
        df = pd.read_excel(buffer)
        return infer_better_dtypes(df)
    if suffix == "json":
        df = pd.read_json(buffer)
        return infer_better_dtypes(df)
    raise ValueError("Unsupported file type. Please upload CSV, XLSX, or JSON.")


def google_sheet_csv_url(sheet_url: str) -> str:
    parsed = urlparse(sheet_url.strip())
    if "docs.google.com" not in parsed.netloc or "/spreadsheets/" not in parsed.path:
        raise ValueError("Please enter a valid Google Sheets URL.")

    path_parts = [part for part in parsed.path.split("/") if part]
    try:
        spreadsheet_id = path_parts[path_parts.index("d") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("Could not find the spreadsheet ID in the URL.") from exc

    query = parse_qs(parsed.query)
    fragment = parse_qs(parsed.fragment)
    gid = query.get("gid", fragment.get("gid", ["0"]))[0]
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"


@st.cache_data(show_spinner=False)
def load_google_sheet(sheet_url: str) -> pd.DataFrame:
    csv_url = google_sheet_csv_url(sheet_url)
    df = pd.read_csv(csv_url)
    return infer_better_dtypes(df)


def infer_better_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    inferred = df.copy()
    object_columns = inferred.select_dtypes(include=["object", "string"]).columns.tolist()

    for column in object_columns:
        series = inferred[column]
        non_null = series.dropna()
        if non_null.empty:
            continue

        # Try numeric inference for columns that mostly contain numeric-looking values.
        text_values = non_null.astype(str).str.strip()
        numeric_pattern = r"^[+-]?((\d+(\.\d*)?)|(\.\d+))$"
        numeric_like_ratio = text_values.str.fullmatch(numeric_pattern).mean()
        if numeric_like_ratio >= 0.85:
            inferred[column] = pd.to_numeric(series, errors="coerce")
            continue

        # Try datetime inference for columns with mostly parseable date values.
        datetime_candidate = pd.to_datetime(non_null, errors="coerce", format="mixed")
        if datetime_candidate.notna().mean() >= 0.85:
            inferred[column] = pd.to_datetime(inferred[column], errors="coerce", format="mixed")

    return inferred


@st.cache_data(show_spinner=False)
def profile_dataframe(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": pd.DataFrame(
            {
                "missing_count": df.isna().sum(),
                "missing_pct": (df.isna().mean() * 100).round(2),
            }
        ),
        "duplicates": int(df.duplicated().sum()),
        "numeric_summary": df[numeric_cols].describe().T if numeric_cols else pd.DataFrame(),
        "categorical_summary": df[categorical_cols].astype(str).describe().T if categorical_cols else pd.DataFrame(),
        "datetime_summary": df[datetime_cols].describe().T if datetime_cols else pd.DataFrame(),
    }


def require_dataframe() -> pd.DataFrame | None:
    df = st.session_state.working_df
    if df is None:
        st.info("Upload a dataset on Page A to begin.")
        return None
    return df


def apply_update(df: pd.DataFrame, step: str, params: dict, columns) -> None:
    push_history()
    st.session_state.working_df = df
    record_step(step, params, columns)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def non_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude="number").columns.tolist()


def outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        rows.append({"column": column, "lower_bound": lower, "upper_bound": upper, "outlier_count": int(outliers)})
    return pd.DataFrame(rows)


def mapping_from_editor(mapping_df: pd.DataFrame) -> dict:
    if mapping_df.empty:
        return {}
    cleaned = mapping_df.fillna("").copy()
    cleaned["source"] = cleaned["source"].astype(str).str.strip()
    cleaned["target"] = cleaned["target"].astype(str).str.strip()
    cleaned = cleaned[(cleaned["source"] != "") & (cleaned["target"] != "")]
    return dict(zip(cleaned["source"], cleaned["target"]))


def evaluate_formula_expression(df: pd.DataFrame, expression: str) -> pd.Series:
    local_scope = {column: df[column] for column in df.columns}
    local_scope.update(
        {
            "pd": pd,
            "np": np,
            "log": np.log,
            "log10": np.log10,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "round": np.round,
        }
    )
    result = eval(expression, {"__builtins__": {}}, local_scope)
    if isinstance(result, pd.Series):
        return result
    if np.isscalar(result):
        return pd.Series([result] * len(df), index=df.index)
    return pd.Series(result, index=df.index)


def build_chart_suggestion_prompt(df: pd.DataFrame, user_goal: str) -> str:
    numeric_cols = numeric_columns(df)
    categorical_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    preview = df.head(5).to_dict(orient="records")
    schema = {
        "row_count": len(df),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "sample_rows": preview,
    }
    return (
        "You are helping inside a Streamlit data visualization builder. "
        "Suggest the best single chart configuration for the user's goal. "
        "Return only valid JSON with keys: "
        "plot_type, x_col, y_col, group_col, aggregation, title, rationale. "
        "Allowed plot_type values: histogram, box_plot, scatter_plot, line_chart, bar_chart, heatmap_correlation. "
        "Allowed aggregation values: none, sum, mean, count, median. "
        "Use group_col as 'None' if grouping is not useful. Use y_col as 'None' when not needed. "
        "Choose only columns that exist in the schema.\n\n"
        f"User goal: {user_goal or 'Recommend the most informative chart for this dataset.'}\n\n"
        f"Dataset schema: {json.dumps(schema, default=str)}"
    )


def get_openai_api_key() -> str | None:
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def get_chart_suggestion_from_openai(df: pd.DataFrame, user_goal: str, model_name: str, temperature: float) -> dict:
    from openai import OpenAI

    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("No OPENAI_API_KEY found. Add it as an environment variable or Streamlit secret.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model_name,
        temperature=temperature,
        instructions="You are an optional AI assistant for a Streamlit coursework app. Outputs may be imperfect.",
        input=build_chart_suggestion_prompt(df, user_goal),
    )
    content = response.output_text.strip()
    cleaned = content
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()
    suggestion = json.loads(cleaned)
    return suggestion


def normalize_chart_suggestion(df: pd.DataFrame, suggestion: dict) -> dict:
    valid_plot_types = {"histogram", "box_plot", "scatter_plot", "line_chart", "bar_chart", "heatmap_correlation"}
    valid_aggs = {"none", "sum", "mean", "count", "median"}
    columns = df.columns.tolist()

    normalized = {
        "plot_type": suggestion.get("plot_type", "bar_chart"),
        "x_col": suggestion.get("x_col", columns[0] if columns else "None"),
        "y_col": suggestion.get("y_col", "None"),
        "group_col": suggestion.get("group_col", "None"),
        "aggregation": suggestion.get("aggregation", "none"),
        "title": suggestion.get("title", ""),
        "rationale": suggestion.get("rationale", ""),
    }

    if normalized["plot_type"] not in valid_plot_types:
        normalized["plot_type"] = "bar_chart"
    if normalized["aggregation"] not in valid_aggs:
        normalized["aggregation"] = "none"
    if normalized["x_col"] not in columns:
        normalized["x_col"] = columns[0] if columns else "None"
    if normalized["y_col"] != "None" and normalized["y_col"] not in columns:
        normalized["y_col"] = "None"
    if normalized["group_col"] != "None" and normalized["group_col"] not in columns:
        normalized["group_col"] = "None"
    return normalized


def render_upload_overview() -> None:
    st.header("Upload & Overview")
    left, middle, right = st.columns([2, 2, 1])
    with left:
        uploaded = st.file_uploader("Upload CSV, XLSX, or JSON", type=["csv", "xlsx", "json"])
    with middle:
        sheet_url = st.text_input(
            "Optional Google Sheets URL",
            placeholder="https://docs.google.com/spreadsheets/d/.../edit#gid=0",
            help="Use a public or shared Google Sheet tab URL. The app converts it to a CSV export link.",
        )
        load_sheet = st.button("Load Google Sheet", use_container_width=True)
    with right:
        if st.button("Reset session", use_container_width=True):
            reset_session()
            st.rerun()

    if uploaded is not None:
        try:
            df = load_uploaded_file(uploaded.getvalue(), uploaded.name)
            set_loaded_dataframe(df, uploaded.name, "load_file", {"file_name": uploaded.name, "source_type": "file"})
        except Exception as exc:
            st.error(f"Could not load file: {exc}")
            return

    if load_sheet and sheet_url.strip():
        try:
            df = load_google_sheet(sheet_url.strip())
            set_loaded_dataframe(
                df,
                "google_sheet_import",
                "load_google_sheet",
                {"sheet_url": sheet_url.strip(), "source_type": "google_sheets"},
            )
            st.success("Google Sheet loaded successfully.")
        except Exception as exc:
            st.error(f"Could not load Google Sheet: {exc}")
            return
    elif load_sheet:
        st.warning("Paste a Google Sheets URL before clicking load.")

    df = require_dataframe()
    if df is None:
        return

    profile = profile_dataframe(df)
    box1, box2, box3, box4 = st.columns(4)
    box1.metric("Rows", profile["shape"][0])
    box2.metric("Columns", profile["shape"][1])
    box3.metric("Missing cells", int(df.isna().sum().sum()))
    box4.metric("Duplicate rows", profile["duplicates"])

    st.subheader("Columns and inferred types")
    dtype_df = pd.DataFrame({"column": profile["columns"], "dtype": [profile["dtypes"][c] for c in profile["columns"]]})
    st.dataframe(dtype_df, use_container_width=True)

    stats_left, stats_right = st.columns(2)
    with stats_left:
        st.subheader("Numeric summary")
        st.dataframe(profile["numeric_summary"], use_container_width=True)
    with stats_right:
        st.subheader("Categorical summary")
        st.dataframe(profile["categorical_summary"], use_container_width=True)
    if not profile["datetime_summary"].empty:
        st.subheader("Datetime summary")
        st.dataframe(profile["datetime_summary"], use_container_width=True)

    st.subheader("Missing values by column")
    st.dataframe(profile["missing"], use_container_width=True)
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)


def render_cleaning_preparation() -> None:
    st.header("Cleaning & Preparation Studio")
    df = require_dataframe()
    if df is None:
        return

    st.subheader("Transformation log")
    st.dataframe(pd.DataFrame(st.session_state.log), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Undo last step", use_container_width=True, disabled=not st.session_state.history):
            st.session_state.working_df = st.session_state.history.pop()
            if st.session_state.log:
                st.session_state.log.pop()
            st.rerun()
    with col2:
        if st.button("Reset to original dataset", use_container_width=True, disabled=st.session_state.original_df is None):
            st.session_state.working_df = st.session_state.original_df.copy()
            st.session_state.history = []
            st.session_state.log = st.session_state.log[:1]
            st.rerun()

    st.divider()

    with st.expander("4.1 Missing Values", expanded=True):
        missing_df = pd.DataFrame(
            {"missing_count": df.isna().sum(), "missing_pct": (df.isna().mean() * 100).round(2)}
        ).sort_values("missing_count", ascending=False)
        st.dataframe(missing_df, use_container_width=True)

        action = st.selectbox(
            "Choose missing value action",
            ["Drop rows with missing values", "Drop columns above threshold", "Fill selected column"],
        )
        if action == "Drop rows with missing values":
            cols = st.multiselect("Columns to check", df.columns.tolist())
            if st.button("Apply row drop") and cols:
                before = len(df)
                updated = df.dropna(subset=cols)
                st.write(f"Before: {before} rows. After: {len(updated)} rows.")
                apply_update(updated, "drop_rows_missing", {"columns": cols}, cols)
                st.rerun()
        elif action == "Drop columns above threshold":
            threshold = st.slider("Missing percentage threshold", 0, 100, 40)
            if st.button("Drop columns above threshold"):
                cols_to_drop = missing_df[missing_df["missing_pct"] > threshold].index.tolist()
                updated = df.drop(columns=cols_to_drop) if cols_to_drop else df.copy()
                st.write(f"Columns removed: {cols_to_drop or 'None'}")
                apply_update(updated, "drop_columns_missing_threshold", {"threshold_pct": threshold}, cols_to_drop)
                st.rerun()
        else:
            column = st.selectbox("Column", df.columns.tolist(), key="fill_column")
            fill_method = st.selectbox(
                "Fill method",
                ["constant", "mean", "median", "mode", "most_frequent", "forward_fill", "backward_fill"],
            )
            constant_value = st.text_input("Constant value", value="", disabled=fill_method != "constant")
            if st.button("Fill missing values"):
                updated = df.copy()
                try:
                    if fill_method == "constant":
                        updated[column] = updated[column].fillna(constant_value)
                    elif fill_method == "mean":
                        updated[column] = pd.to_numeric(updated[column], errors="coerce").fillna(
                            pd.to_numeric(updated[column], errors="coerce").mean()
                        )
                    elif fill_method == "median":
                        updated[column] = pd.to_numeric(updated[column], errors="coerce").fillna(
                            pd.to_numeric(updated[column], errors="coerce").median()
                        )
                    elif fill_method in {"mode", "most_frequent"}:
                        updated[column] = updated[column].fillna(updated[column].mode(dropna=True).iloc[0])
                    elif fill_method == "forward_fill":
                        updated[column] = updated[column].ffill()
                    elif fill_method == "backward_fill":
                        updated[column] = updated[column].bfill()
                    st.write("Before / after missing count:", int(df[column].isna().sum()), "->", int(updated[column].isna().sum()))
                    apply_update(updated, "fill_missing", {"column": column, "method": fill_method}, [column])
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not fill missing values: {exc}")

    with st.expander("4.2 Duplicates"):
        dup_mode = st.radio("Duplicate detection mode", ["Full row duplicates", "Subset duplicates"], horizontal=True)
        subset = []
        if dup_mode == "Subset duplicates":
            subset = st.multiselect("Select key columns", df.columns.tolist())
        duplicate_mask = df.duplicated(subset=subset or None, keep=False)
        duplicate_rows = df[duplicate_mask]
        st.write(f"Duplicate groups found: {len(duplicate_rows)} rows")
        st.dataframe(duplicate_rows.head(100), use_container_width=True)
        keep = st.selectbox("Keep rule", ["first", "last"])
        if st.button("Remove duplicates"):
            updated = df.drop_duplicates(subset=subset or None, keep=keep)
            apply_update(updated, "remove_duplicates", {"subset": subset, "keep": keep}, subset or "ALL")
            st.rerun()

    with st.expander("4.3 Data Types & Parsing"):
        column = st.selectbox("Column to convert", df.columns.tolist(), key="convert_column")
        target = st.selectbox("Target type", ["numeric", "category", "datetime"])
        dt_format = st.text_input("Datetime format (optional)", value="%Y-%m-%d", disabled=target != "datetime")
        if st.button("Convert data type"):
            updated = df.copy()
            try:
                if target == "numeric":
                    cleaned = updated[column].astype(str).str.replace(r"[^0-9.\\-]", "", regex=True)
                    updated[column] = pd.to_numeric(cleaned, errors="coerce")
                elif target == "category":
                    updated[column] = updated[column].astype("category")
                else:
                    updated[column] = pd.to_datetime(updated[column], format=dt_format or None, errors="coerce")
                apply_update(updated, "convert_dtype", {"column": column, "target": target}, [column])
                st.rerun()
            except Exception as exc:
                st.error(f"Could not convert column: {exc}")

    with st.expander("4.4 Categorical Data Tools"):
        cat_cols = non_numeric_columns(df)
        if not cat_cols:
            st.info("No categorical columns available.")
        else:
            column = st.selectbox("Categorical column", cat_cols, key="cat_column")
            case_action = st.selectbox("Standardization", ["trim_whitespace", "lower", "title"])
            if st.button("Apply standardization"):
                updated = df.copy()
                series = updated[column].astype(str)
                if case_action == "trim_whitespace":
                    updated[column] = series.str.strip()
                elif case_action == "lower":
                    updated[column] = series.str.strip().str.lower()
                else:
                    updated[column] = series.str.strip().str.title()
                apply_update(updated, "standardize_category", {"column": column, "action": case_action}, [column])
                st.rerun()

            unique_preview = pd.DataFrame({"source": sorted(df[column].dropna().astype(str).unique().tolist())[:8], "target": [""] * min(8, df[column].dropna().nunique())})
            mapping_editor = st.data_editor(
                unique_preview if not unique_preview.empty else pd.DataFrame({"source": [""], "target": [""]}),
                num_rows="dynamic",
                use_container_width=True,
                key=f"mapping_editor_{column}",
                column_config={
                    "source": st.column_config.TextColumn("Source value"),
                    "target": st.column_config.TextColumn("Replacement value"),
                },
            )
            map_to_other = st.checkbox("Set unmatched values to Other", value=False)
            if st.button("Apply mapping"):
                mapping = mapping_from_editor(mapping_editor)
                updated = df.copy()
                mapped = updated[column].map(mapping)
                updated[column] = mapped.fillna("Other") if map_to_other else updated[column].replace(mapping)
                apply_update(updated, "map_categories", {"column": column, "mapping": mapping}, [column])
                st.rerun()

            threshold = st.number_input("Rare category minimum frequency", min_value=1, value=10)
            if st.button("Group rare categories"):
                counts = df[column].value_counts(dropna=False)
                rare_values = counts[counts < threshold].index
                updated = df.copy()
                updated[column] = updated[column].where(~updated[column].isin(rare_values), "Other")
                apply_update(updated, "group_rare_categories", {"column": column, "threshold": threshold}, [column])
                st.rerun()

            if st.button("One-hot encode selected column"):
                updated = pd.get_dummies(df, columns=[column], dummy_na=False)
                apply_update(updated, "one_hot_encode", {"column": column}, [column])
                st.rerun()

    with st.expander("4.5 Numeric Cleaning"):
        num_cols = numeric_columns(df)
        if not num_cols:
            st.info("No numeric columns available.")
        else:
            cols = st.multiselect("Numeric columns", num_cols, default=num_cols[: min(3, len(num_cols))])
            summary = outlier_summary(df, cols)
            st.dataframe(summary, use_container_width=True)
            action = st.selectbox("Outlier action", ["cap_winsorize", "remove_rows", "do_nothing"])
            lower_q, upper_q = st.slider("Quantile caps", 0.0, 1.0, (0.05, 0.95))
            if st.button("Apply numeric cleaning") and cols:
                updated = df.copy()
                impacted = {"rows_removed": 0, "values_capped": 0}
                if action == "cap_winsorize":
                    for col in cols:
                        lower = updated[col].quantile(lower_q)
                        upper = updated[col].quantile(upper_q)
                        before = updated[col].copy()
                        updated[col] = updated[col].clip(lower=lower, upper=upper)
                        impacted["values_capped"] += int((before != updated[col]).sum())
                elif action == "remove_rows":
                    mask = pd.Series(False, index=updated.index)
                    for col in cols:
                        q1 = updated[col].quantile(0.25)
                        q3 = updated[col].quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        mask = mask | ((updated[col] < lower) | (updated[col] > upper))
                    impacted["rows_removed"] = int(mask.sum())
                    updated = updated.loc[~mask]
                st.write(impacted)
                apply_update(updated, "numeric_cleaning", {"columns": cols, "action": action}, cols)
                st.rerun()

    with st.expander("4.6 Normalization / Scaling"):
        num_cols = numeric_columns(df)
        cols = st.multiselect("Columns to scale", num_cols, key="scale_columns")
        method = st.selectbox("Scaling method", ["min_max", "z_score"])
        if st.button("Apply scaling") and cols:
            updated = df.copy()
            before_stats = updated[cols].describe().T[["mean", "std", "min", "max"]]
            for col in cols:
                series = updated[col]
                if method == "min_max":
                    denom = series.max() - series.min()
                    updated[col] = 0 if denom == 0 else (series - series.min()) / denom
                else:
                    std = series.std()
                    updated[col] = 0 if std == 0 else (series - series.mean()) / std
            after_stats = updated[cols].describe().T[["mean", "std", "min", "max"]]
            st.write("Before")
            st.dataframe(before_stats, use_container_width=True)
            st.write("After")
            st.dataframe(after_stats, use_container_width=True)
            apply_update(updated, "scale_numeric", {"columns": cols, "method": method}, cols)
            st.rerun()

    with st.expander("4.7 Column Operations"):
        rename_col = st.selectbox("Rename column", df.columns.tolist(), key="rename_col")
        new_name = st.text_input("New name")
        if st.button("Rename selected column") and new_name:
            updated = df.rename(columns={rename_col: new_name})
            apply_update(updated, "rename_column", {"old": rename_col, "new": new_name}, [rename_col])
            st.rerun()

        drop_cols = st.multiselect("Drop columns", df.columns.tolist(), key="drop_cols")
        if st.button("Drop selected columns") and drop_cols:
            updated = df.drop(columns=drop_cols)
            apply_update(updated, "drop_columns", {"columns": drop_cols}, drop_cols)
            st.rerun()

        st.caption("Formula examples: revenue / quantity, log(sales_amount), profit - profit.mean(), sales_amount.fillna(0)")
        formula_name = st.text_input("New formula column name")
        formula_expr = st.text_input("Formula expression")
        if st.button("Create formula column") and formula_name and formula_expr:
            try:
                updated = df.copy()
                updated[formula_name] = evaluate_formula_expression(updated, formula_expr)
                apply_update(updated, "create_formula_column", {"new_column": formula_name, "expression": formula_expr}, [formula_name])
                st.rerun()
            except Exception as exc:
                st.error(f"Formula could not be applied: {exc}")

        bin_column = st.selectbox("Column to bin", numeric_columns(df), key="bin_column")
        bin_count = st.slider("Number of bins", 2, 10, 4)
        bin_method = st.selectbox("Binning method", ["equal_width", "quantile"])
        bin_name = st.text_input("New binned column name", value="binned_column")
        if st.button("Create binned column"):
            try:
                updated = df.copy()
                if bin_method == "equal_width":
                    updated[bin_name] = pd.cut(updated[bin_column], bins=bin_count)
                else:
                    updated[bin_name] = pd.qcut(updated[bin_column], q=bin_count, duplicates="drop")
                apply_update(updated, "bin_numeric_column", {"column": bin_column, "bins": bin_count, "method": bin_method}, [bin_column, bin_name])
                st.rerun()
            except Exception as exc:
                st.error(f"Could not bin column: {exc}")

    with st.expander("4.8 Data Validation Rules"):
        numeric_col = st.selectbox("Numeric range column", ["None"] + numeric_columns(df))
        if numeric_col != "None":
            min_val = st.number_input("Minimum allowed", value=float(df[numeric_col].min()))
            max_val = st.number_input("Maximum allowed", value=float(df[numeric_col].max()))
        else:
            min_val = max_val = None

        category_col = st.selectbox("Allowed categories column", ["None"] + non_numeric_columns(df))
        if category_col != "None":
            allowed = st.text_input("Comma-separated allowed values", key="validation_allowed_values")
        else:
            allowed = ""

        non_null_cols = st.multiselect("Columns that must not be null", df.columns.tolist())
        combination_mode = st.radio(
            "Violation logic",
            ["Any selected rule", "All selected rules"],
            horizontal=True,
            help="Choose whether the violations table shows rows that fail at least one rule or only rows that fail every selected rule.",
        )

        if category_col != "None":
            st.caption(
                f"Selected allowed values will be compared against `{category_col}` after trimming spaces and ignoring case."
            )

        run_validation = st.button("Run validation checks", use_container_width=True)

        if run_validation:
            summary_rows = []
            rule_masks = []
            rule_labels = []

            if numeric_col != "None":
                numeric_mask = (df[numeric_col] < min_val) | (df[numeric_col] > max_val)
                numeric_violations = df[numeric_mask].copy()
                summary_rows.append({"rule": f"{numeric_col} range", "violations": len(numeric_violations)})
                rule_masks.append(numeric_mask)
                rule_labels.append(f"{numeric_col} outside range")

            if category_col != "None" and allowed.strip():
                allowed_values = {item.strip().casefold() for item in allowed.split(",") if item.strip()}
                normalized_series = df[category_col].astype(str).str.strip().str.casefold()
                category_mask = ~normalized_series.isin(allowed_values)
                category_violations = df[category_mask].copy()
                summary_rows.append({"rule": f"{category_col} allowed list", "violations": len(category_violations)})
                st.write("Allowed values used:", ", ".join(sorted(allowed_values)))
                rule_masks.append(category_mask)
                rule_labels.append(f"{category_col} not in allowed list")

            if non_null_cols:
                null_mask = df[non_null_cols].isna().any(axis=1)
                null_violations = df[null_mask].copy()
                summary_rows.append({"rule": "Non-null constraint", "violations": len(null_violations)})
                rule_masks.append(null_mask)
                rule_labels.append("Non-null constraint failed")

            if summary_rows:
                st.subheader("Validation summary")
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            if rule_masks:
                if combination_mode == "All selected rules":
                    combined_mask = pd.concat(rule_masks, axis=1).all(axis=1)
                else:
                    combined_mask = pd.concat(rule_masks, axis=1).any(axis=1)

                violations_df = df[combined_mask].copy()
                matched_rules = []
                for idx in violations_df.index:
                    row_matches = [label for label, mask in zip(rule_labels, rule_masks) if bool(mask.loc[idx])]
                    matched_rules.append(" | ".join(row_matches))
                violations_df["violation_type"] = matched_rules
                st.caption(
                    f"Showing rows that fail {'all' if combination_mode == 'All selected rules' else 'at least one'} selected rule."
                )
                st.dataframe(violations_df, use_container_width=True)
                csv_bytes = violations_df.to_csv(index=False).encode("utf-8")
                st.download_button("Export violations table", csv_bytes, file_name="validation_violations.csv", mime="text/csv")
            else:
                st.info("No violations found for the selected rules.")


def render_visualization_builder() -> None:
    st.header("Visualization Builder")
    df = require_dataframe()
    if df is None:
        return

    with st.expander("AI Assistant"):
        enable_ai = st.checkbox("Enable AI assistant")
        st.caption("AI suggestions are optional and may be imperfect. The app works fully without AI.")
        fixed_ai_model = "gpt-4.1-nano"
        fixed_temperature = 0.1
        ai_goal = st.text_area(
            "What kind of chart do you want?",
            placeholder="Example: Show the relationship between sales and profit over time, grouped by region.",
            disabled=not enable_ai,
        )
        st.text_input("OpenAI model", value=fixed_ai_model, disabled=True)
        st.text_input("Temperature", value="0.1", disabled=True)
        if st.button("Suggest chart with AI", disabled=not enable_ai, use_container_width=True):
            try:
                suggestion = get_chart_suggestion_from_openai(df, ai_goal, fixed_ai_model, fixed_temperature)
                st.session_state.ai_chart_suggestion = normalize_chart_suggestion(df, suggestion)
            except Exception as exc:
                st.error(f"AI suggestion failed: {exc}")

        suggestion = st.session_state.get("ai_chart_suggestion")
        if suggestion:
            st.subheader("AI suggestion")
            st.json(suggestion)
            if st.button("Apply AI suggestion", use_container_width=True):
                st.session_state.viz_plot_type = suggestion["plot_type"]
                st.session_state.viz_x = suggestion["x_col"]
                st.session_state.viz_y = suggestion["y_col"]
                st.session_state.viz_group = suggestion["group_col"]
                st.session_state.viz_agg = suggestion["aggregation"]
                st.session_state.viz_title = suggestion["title"]
                st.rerun()

    filtered_df = df.copy()
    with st.sidebar:
        st.subheader("Visualization filters")
        cat_cols = non_numeric_columns(df)
        if cat_cols:
            filter_col = st.selectbox("Category filter column", ["None"] + cat_cols)
            if filter_col != "None":
                options = sorted(filtered_df[filter_col].dropna().astype(str).unique().tolist())
                selected = st.multiselect("Category values", options)
                if selected:
                    filtered_df = filtered_df[filtered_df[filter_col].astype(str).isin(selected)]

        num_cols = numeric_columns(filtered_df)
        if num_cols:
            range_col = st.selectbox("Numeric range column", ["None"] + num_cols)
            if range_col != "None":
                min_val = float(filtered_df[range_col].min())
                max_val = float(filtered_df[range_col].max())
                chosen = st.slider("Range", min_val, max_val, (min_val, max_val))
                filtered_df = filtered_df[filtered_df[range_col].between(chosen[0], chosen[1])]

    plot_type = st.selectbox(
        "Plot type",
        ["histogram", "box_plot", "scatter_plot", "line_chart", "bar_chart", "heatmap_correlation"],
        key="viz_plot_type",
    )
    cols1, cols2, cols3, cols4 = st.columns(4)
    with cols1:
        x_col = st.selectbox("X column", filtered_df.columns.tolist(), key="viz_x")
    with cols2:
        y_options = ["None"] + filtered_df.columns.tolist()
        y_col = st.selectbox("Y column", y_options, key="viz_y")
    with cols3:
        group_col = st.selectbox("Color / Group", ["None"] + filtered_df.columns.tolist(), key="viz_group")
    with cols4:
        agg = st.selectbox("Aggregation", ["none", "sum", "mean", "count", "median"], key="viz_agg")
    custom_title = st.text_input("Plot title (optional)", placeholder="Enter a custom chart title", key="viz_title")

    top_n = st.slider("Top N categories for bar charts", 3, 30, 10)
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = ["#264653", "#2a9d8f", "#e76f51", "#f4a261", "#457b9d", "#8d99ae", "#e63946"]

    try:
        chart_df = filtered_df.copy()
        if agg != "none" and plot_type in {"bar_chart", "line_chart"} and y_col != "None":
            groupers = [x_col]
            if group_col != "None":
                groupers.append(group_col)
            chart_df = getattr(chart_df.groupby(groupers)[y_col], agg)().reset_index()

        default_titles = {
            "histogram": f"Histogram of {x_col}",
            "box_plot": f"Box Plot of {x_col}",
            "scatter_plot": f"{y_col} vs {x_col}",
            "line_chart": f"{y_col} over {x_col}",
            "bar_chart": f"Bar Chart of {x_col}",
            "heatmap_correlation": "Correlation Matrix",
        }
        plot_title = custom_title.strip() or default_titles[plot_type]

        if plot_type == "histogram":
            ax.hist(pd.to_numeric(chart_df[x_col], errors="coerce").dropna(), bins=30, color="#457b9d", edgecolor="white")
            ax.set_title(plot_title)
        elif plot_type == "box_plot":
            ax.boxplot(pd.to_numeric(chart_df[x_col], errors="coerce").dropna())
            ax.set_title(plot_title)
            ax.set_xticklabels([x_col])
        elif plot_type == "scatter_plot":
            if y_col == "None":
                st.warning("Scatter plot requires both X and Y columns.")
                return
            if group_col != "None":
                for i, (group_value, group_data) in enumerate(chart_df.groupby(group_col)):
                    ax.scatter(
                        group_data[x_col],
                        group_data[y_col],
                        alpha=0.6,
                        color=palette[i % len(palette)],
                        label=str(group_value),
                    )
                ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")
            else:
                ax.scatter(chart_df[x_col], chart_df[y_col], alpha=0.6, color="#e76f51")
            ax.set_title(plot_title)
        elif plot_type == "line_chart":
            if y_col == "None":
                st.warning("Line chart requires both X and Y columns.")
                return
            if group_col != "None":
                for i, (group_value, group_data) in enumerate(chart_df.groupby(group_col)):
                    group_data = group_data.sort_values(x_col)
                    ax.plot(
                        group_data[x_col],
                        group_data[y_col],
                        color=palette[i % len(palette)],
                        label=str(group_value),
                    )
                ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")
            else:
                chart_df = chart_df.sort_values(x_col)
                ax.plot(chart_df[x_col], chart_df[y_col], color="#2a9d8f")
            ax.set_title(plot_title)
            ax.tick_params(axis="x", rotation=45)
        elif plot_type == "bar_chart":
            if y_col != "None" and agg != "none":
                if group_col != "None":
                    pivot_df = chart_df.pivot(index=x_col, columns=group_col, values=y_col).fillna(0)
                    pivot_df["__total__"] = pivot_df.sum(axis=1)
                    pivot_df = pivot_df.sort_values("__total__", ascending=False).drop(columns="__total__").head(top_n)
                    pivot_df.plot(kind="bar", ax=ax, color=palette[: len(pivot_df.columns)])
                    ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")
                else:
                    plotted = chart_df.sort_values(y_col, ascending=False).head(top_n)
                    ax.bar(plotted[x_col].astype(str), plotted[y_col], color="#264653")
            else:
                if group_col != "None":
                    grouped_counts = (
                        chart_df.groupby([x_col, group_col]).size().reset_index(name="count")
                    )
                    pivot_counts = grouped_counts.pivot(index=x_col, columns=group_col, values="count").fillna(0)
                    pivot_counts["__total__"] = pivot_counts.sum(axis=1)
                    pivot_counts = pivot_counts.sort_values("__total__", ascending=False).drop(columns="__total__").head(top_n)
                    pivot_counts.plot(kind="bar", ax=ax, color=palette[: len(pivot_counts.columns)])
                    ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")
                else:
                    value_counts = chart_df[x_col].astype(str).value_counts().head(top_n)
                    ax.bar(value_counts.index, value_counts.values, color="#264653")
            ax.set_title(plot_title)
            ax.tick_params(axis="x", rotation=45)
        else:
            corr = chart_df.select_dtypes(include="number").corr(numeric_only=True)
            if corr.empty:
                st.warning("Heatmap requires numeric columns.")
                return
            heat = ax.imshow(corr, cmap="Blues")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)
            ax.set_title(plot_title)
            fig.colorbar(heat)

        ax.set_xlabel(x_col)
        if y_col != "None":
            ax.set_ylabel(y_col)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        image_buffer = io.BytesIO()
        fig.savefig(image_buffer, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            "Download plot as PNG",
            data=image_buffer.getvalue(),
            file_name=f"{plot_type}_chart.png",
            mime="image/png",
        )
        st.caption(f"Rows available after filters: {len(filtered_df)}")
    except Exception as exc:
        st.error(f"Could not build chart: {exc}")


def render_export_report() -> None:
    st.header("Export & Report")
    df = require_dataframe()
    if df is None:
        return

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Export cleaned dataset as CSV", csv_bytes, file_name="cleaned_dataset.csv", mime="text/csv")

    excel_buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="cleaned_data")
        st.download_button(
            "Export cleaned dataset as Excel",
            excel_buffer.getvalue(),
            file_name="cleaned_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.info("Excel export requires openpyxl in the environment.")

    report_df = pd.DataFrame(deepcopy(st.session_state.log))
    st.subheader("Transformation report")
    st.dataframe(report_df, use_container_width=True)
    st.download_button(
        "Export transformation report CSV",
        report_df.to_csv(index=False).encode("utf-8"),
        file_name="transformation_report.csv",
        mime="text/csv",
    )

    recipe = {
        "source_name": st.session_state.source_name,
        "generated_at": now_iso(),
        "steps": st.session_state.log,
    }
    st.download_button(
        "Export JSON recipe",
        json.dumps(recipe, indent=2).encode("utf-8"),
        file_name="transformation_recipe.json",
        mime="application/json",
    )


def main() -> None:
    init_state()
    st.title("AI-Assisted Data Wrangler & Visualizer")
    st.caption("Coursework app for 5COSC038C: upload, clean, validate, visualize, and export reproducible data workflows.")

    page = st.sidebar.radio(
        "Navigate",
        ["Upload & Overview", "Cleaning & Preparation Studio", "Visualization Builder", "Export & Report"],
    )

    if page == "Upload & Overview":
        render_upload_overview()
    elif page == "Cleaning & Preparation Studio":
        render_cleaning_preparation()
    elif page == "Visualization Builder":
        render_visualization_builder()
    else:
        render_export_report()


if __name__ == "__main__":
    main()


 


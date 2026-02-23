import logging
import os
from typing import Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)
WORKFLOW_NAME = "eda_workflow"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    with open(prompt_path, "r") as f:
        return f.read()


class EDAWorkflow:
    """
    Exploratory Data Analysis workflow that performs consistent, first-pass analysis of datasets.
    
    Uses a fixed set of predefined analysis tools to produce structured, tabular outputs.
    Operates sequentially and deterministically through baseline EDA steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Attributes
    ----------
    response : dict or None
        Stores the full response after invoke_workflow() is called.
    """
    
    def __init__(
        self,
        model=None,
        log=False,
        log_path=None,
        checkpointer: Optional[object] = None
    ):
        self.model = model
        self.log = log
        self.log_path = log_path
        self.checkpointer = checkpointer
        self.response = None
        self._compiled_graph = make_eda_baseline_workflow(
            model=model,
            log=log,
            log_path=log_path,
            checkpointer=checkpointer
        )
    
    def invoke_workflow(self, filepath: str, **kwargs):
        """
        Run EDA analysis on the provided dataset.
        
        Parameters
        ----------
        filepath : str
            Path to the dataset file.
        **kwargs
            Additional arguments passed to the underlying graph invoke method.
        
        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        df = pd.read_csv(filepath)
        
        response = self._compiled_graph.invoke({
            "dataframe": df.to_dict(),
            "results": {},
            "observations": {},
            "current_step": "",
            "summary": "",
            "recommendations": [],
        }, **kwargs)
        
        self.response = response
        return None
    
    def get_summary(self):
        """Retrieves the analysis summary."""
        if self.response:
            return self.response.get("summary")
    
    def get_recommendations(self):
        """Retrieves the recommendations."""
        if self.response:
            return self.response.get("recommendations")
    
    def get_results(self):
        """Retrieves the full analysis results."""
        if self.response:
            return self.response.get("results")
    
    def get_observations(self):
        """Retrieves all observations from analysis steps."""
        if self.response:
            return self.response.get("observations")


def make_eda_baseline_workflow(
    model=None,
    log=False,
    log_path=None,
    checkpointer: Optional[object] = None
):
    """
    Factory function that creates a compiled LangGraph workflow for baseline EDA.
    
    Performs automated first-pass analysis with fixed analysis steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process EDA requests.
    """
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    class EDAState(TypedDict):
        dataframe: dict
        results: dict
        observations: dict[str, list[str]]
        current_step: str
        summary: str
        recommendations: list[str]
    
    def profile_dataset_node(state: EDAState):
        """Generate dataset profile with basic statistics."""
        logger.info("Profiling dataset")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_summary": (
                df[numeric_cols].describe().to_dict() if numeric_cols else {}
            ),
            "categorical_summary": {
                col: df[col].value_counts().head(10).to_dict()
                for col in categorical_cols
            },
        }
        
        results["profile_dataset"] = profile
        
        return {
            "current_step": "profile_dataset",
            "results": results,
        }
    
    def analyze_missingness_node(state: EDAState):
        """Analyze missing values in the dataset."""
        logger.info("Analyzing missingness")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        missing_count = df.isnull().sum().to_dict()
        missing_pct = (
            (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        )
        
        high_missing = {col: pct for col, pct in missing_pct.items() if pct > 20}
        
        missingness = {
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "high_missing_columns": high_missing,
            "complete_rows": int(df.dropna().shape[0]),
            "complete_rows_pct": (
                round(df.dropna().shape[0] / len(df) * 100, 2)
                if len(df) > 0 else 0
            ),
        }
        
        results["analyze_missingness"] = missingness
        
        return {
            "current_step": "analyze_missingness",
            "results": results,
        }
    
    def compute_aggregates_node(state: EDAState):
        """Compute group-by aggregates on key columns.

        Heuristics:
        - Identify candidate grouping columns from categorical-like fields (object/category/bool)
        and low-cardinality integers.
        - For each grouping column, compute group sizes and summary aggregates for numeric columns.

        Results are stored in results["compute_aggregates"].
        """
        logger.info("Computing aggregates")
        import numpy as np

        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        # Identify column types
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_like_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # Treat low-cardinality integers as categorical-like (common in encoded categories)
        int_cols = df.select_dtypes(include=["int", "int32", "int64", "Int64"]).columns.tolist()
        low_card_ints = []
        for c in int_cols:
            try:
                nun = df[c].nunique(dropna=True)
                if 2 <= nun <= 20:
                    low_card_ints.append(c)
            except Exception:
                continue

        candidate_group_cols = list(dict.fromkeys(cat_like_cols + low_card_ints))

        # Limit to a small number of the most promising group-by columns
        # Prefer columns with moderate cardinality (2..20) and low missingness.
        scored = []
        n_rows = len(df)
        for c in candidate_group_cols:
            try:
                nun = df[c].nunique(dropna=True)
                miss = float(df[c].isna().mean()) if n_rows else 0.0
                if 2 <= nun <= 20:
                    # score: fewer missing, moderate nunique
                    score = (1 - miss) * (1 - abs(nun - 8) / 20)
                    scored.append((score, c, nun, miss))
            except Exception:
                continue
        scored.sort(reverse=True)

        group_cols = [c for _, c, _, _ in scored[:5]]  # keep output compact

        # Overall numeric aggregates
        overall_numeric = {}
        for c in numeric_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            overall_numeric[c] = {
                "count": int(s.count()),
                "missing": int(s.isna().sum()),
                "mean": float(s.mean()) if s.count() else None,
                "median": float(s.median()) if s.count() else None,
                "std": float(s.std(ddof=1)) if s.count() > 1 else None,
                "min": float(s.min()) if s.count() else None,
                "max": float(s.max()) if s.count() else None,
            }

        groupby_results = {}
        for g in group_cols:
            g_series = df[g]
            # Group sizes
            top_sizes = (
                g_series.fillna(np.nan)
                .value_counts(dropna=False)
                .head(20)
                .to_dict()
            )

            numeric_aggs = {}
            for nc in numeric_cols:
                # Use numeric coercion to be safe
                tmp = df[[g, nc]].copy()
                tmp[nc] = pd.to_numeric(tmp[nc], errors="coerce")

                if tmp[nc].notna().sum() == 0:
                    continue

                gb = tmp.groupby(g, dropna=False)[nc].agg(["count", "mean", "median", "std", "min", "max"])
                gb = gb.replace({np.nan: None})

                # Keep only top groups by sample size to avoid huge payloads
                gb = gb.sort_values("count", ascending=False).head(20)

                numeric_aggs[nc] = gb.to_dict(orient="index")

            groupby_results[g] = {
                "n_groups": int(df[g].nunique(dropna=False)),
                "missing_pct": round(float(df[g].isna().mean() * 100), 2) if n_rows else 0.0,
                "top_group_sizes": top_sizes,
                "numeric_aggregates_top_groups": numeric_aggs,
            }

        aggregates = {
            "overall_numeric": overall_numeric,
            "groupby": groupby_results,
            "selected_groupby_columns": group_cols,
            "notes": {
                "selection_heuristic": "Categorical-like columns (object/category/bool) + low-cardinality integers; choose up to 5 with 2–20 unique values and low missingness."
            },
        }

        results["compute_aggregates"] = aggregates

        return {
            "current_step": "compute_aggregates",
            "results": results,
        }
    
    def analyze_relationships_node(state: EDAState):
        """Analyze relationships between variables.

        Includes:
        - Numeric↔Numeric correlations (Pearson + Spearman where possible)
        - Categorical↔Numeric association (correlation ratio / eta squared heuristic)
        - Categorical↔Categorical association (Cramér's V for low-cardinality pairs)

        Results are stored in results["analyze_relationships"].
        """
        logger.info("Analyzing relationships")
        import numpy as np

        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # Helper: pick compact set of categorical columns (low cardinality)
        cat_candidates = []
        for c in cat_cols:
            nun = df[c].nunique(dropna=True)
            if 2 <= nun <= 20:
                cat_candidates.append(c)
        cat_candidates = cat_candidates[:10]

        rel = {
            "numeric_numeric": {"pearson_top": [], "spearman_top": []},
            "categorical_numeric": {"eta_squared_top": []},
            "categorical_categorical": {"cramers_v_top": []},
            "notes": {},
        }

        # Numeric-Numeric correlations
        if len(numeric_cols) >= 2:
            num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            pear = num_df.corr(method="pearson", min_periods=20)
            spear = num_df.corr(method="spearman", min_periods=20)

            def top_pairs(corr_df, top_k=10, min_abs=0.3):
                pairs = []
                cols = corr_df.columns.tolist()
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        v = corr_df.iloc[i, j]
                        if pd.isna(v):
                            continue
                        av = abs(float(v))
                        if av >= min_abs:
                            pairs.append({
                                "var_1": cols[i],
                                "var_2": cols[j],
                                "corr": float(v),
                                "abs_corr": av,
                            })
                pairs.sort(key=lambda x: x["abs_corr"], reverse=True)
                return pairs[:top_k]

            rel["numeric_numeric"]["pearson_top"] = top_pairs(pear, top_k=10, min_abs=0.3)
            rel["numeric_numeric"]["spearman_top"] = top_pairs(spear, top_k=10, min_abs=0.3)

        # Categorical-Numeric association: eta^2 (between-group variance / total variance)
        eta_scores = []
        for cat in cat_candidates:
            groups = df[cat]
            for num in numeric_cols:
                y = pd.to_numeric(df[num], errors="coerce")
                valid = (~groups.isna()) & (~y.isna())
                if valid.sum() < 30:
                    continue

                g = groups[valid]
                yv = y[valid]

                # compute eta squared
                overall_mean = float(yv.mean())
                ss_total = float(((yv - overall_mean) ** 2).sum())
                if ss_total <= 0:
                    continue

                gb = pd.DataFrame({"g": g, "y": yv}).groupby("g")["y"]
                means = gb.mean()
                counts = gb.size()

                ss_between = float(((means - overall_mean) ** 2 * counts).sum())
                eta2 = ss_between / ss_total

                if eta2 >= 0.02:  # small but non-trivial
                    eta_scores.append({
                        "categorical": cat,
                        "numeric": num,
                        "eta_squared": float(eta2),
                        "n": int(valid.sum()),
                        "n_groups": int(counts.shape[0]),
                    })

        eta_scores.sort(key=lambda x: x["eta_squared"], reverse=True)
        rel["categorical_numeric"]["eta_squared_top"] = eta_scores[:10]

        # Categorical-Categorical association: Cramér's V (low-cardinality only)
        def cramers_v(x, y):
            ct = pd.crosstab(x, y, dropna=False)
            n = ct.to_numpy().sum()
            if n == 0:
                return None
            observed = ct.to_numpy(dtype=float)
            row_sums = observed.sum(axis=1, keepdims=True)
            col_sums = observed.sum(axis=0, keepdims=True)
            expected = row_sums @ col_sums / n
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2 = np.nansum((observed - expected) ** 2 / expected)
            r, k = observed.shape
            denom = n * (min(r, k) - 1)
            if denom <= 0:
                return None
            return float(np.sqrt(chi2 / denom))

        cc_scores = []
        for i in range(len(cat_candidates)):
            for j in range(i + 1, len(cat_candidates)):
                c1, c2 = cat_candidates[i], cat_candidates[j]
                nun1 = df[c1].nunique(dropna=True)
                nun2 = df[c2].nunique(dropna=True)
                if nun1 > 20 or nun2 > 20:
                    continue
                v = cramers_v(df[c1], df[c2])
                if v is None:
                    continue
                if v >= 0.1:
                    cc_scores.append({
                        "var_1": c1,
                        "var_2": c2,
                        "cramers_v": float(v),
                        "n": int(pd.crosstab(df[c1], df[c2], dropna=False).to_numpy().sum()),
                        "levels_1": int(nun1),
                        "levels_2": int(nun2),
                    })

        cc_scores.sort(key=lambda x: x["cramers_v"], reverse=True)
        rel["categorical_categorical"]["cramers_v_top"] = cc_scores[:10]

        rel["notes"] = {
            "thresholds": {
                "min_abs_corr": 0.3,
                "min_eta_squared": 0.02,
                "min_cramers_v": 0.1
            },
            "categorical_candidates": cat_candidates,
        }

        results["analyze_relationships"] = rel

        return {
            "current_step": "analyze_relationships",
            "results": results,
        }
    
    def extract_observations_node(state: EDAState):
        """Extract observations from the latest analysis results using LLM."""
        logger.info("Extracting observations")
        
        current_step = state.get("current_step", "")
        results = state.get("results", {})
        observations = state.get("observations", {})
        
        if model is None or not current_step or current_step not in results:
            return {"observations": observations}
        
        step_results = results.get(current_step, {})
        
        class ObservationOutput(BaseModel):
            observations: list[str] = Field(description="1-2 concise, actionable observations")
        
        observation_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("extract_observations_system.txt")),
            ("human", load_prompt("extract_observations_human.txt")),
        ])
        
        chain = observation_prompt | model.with_structured_output(ObservationOutput)
        response = chain.invoke({
            "step_name": current_step.replace("_", " ").title(),
            "results": str(step_results)
        })
        
        observations[current_step] = response.observations
        
        return {
            "observations": observations,
        }
    
    def synthesize_findings_node(state: EDAState):
        """Synthesize accumulated findings into summary and recommendations."""
        logger.info("Synthesizing findings")
        
        observations = state.get("observations", {})
        
        if model is None:
            return {
                "summary": "No LLM provided for synthesis",
                "recommendations": [],
            }
        
        class SynthesisOutput(BaseModel):
            summary: str = Field(description="A concise 2-3 sentence summary of key findings")
            recommendations: list[str] = Field(description="3-5 actionable recommendations")
        
        all_observations = []
        for step_name, step_obs in observations.items():
            all_observations.append(f"\n{step_name.replace('_', ' ').title()}:")
            for obs in step_obs:
                all_observations.append(f"  - {obs}")
        
        observations_text = "\n".join(all_observations)
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("synthesize_findings_system.txt")),
            ("human", load_prompt("synthesize_findings_human.txt")),
        ])
        
        chain = synthesis_prompt | model.with_structured_output(SynthesisOutput)
        response = chain.invoke({"observations": observations_text})
        
        return {
            "summary": response.summary,
            "recommendations": response.recommendations,
        }
    
    workflow = StateGraph(EDAState)
    
    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("extract_observations_1", extract_observations_node)
    workflow.add_node("analyze_missingness", analyze_missingness_node)
    workflow.add_node("extract_observations_2", extract_observations_node)
    workflow.add_node("compute_aggregates", compute_aggregates_node)
    workflow.add_node("extract_observations_3", extract_observations_node)
    workflow.add_node("analyze_relationships", analyze_relationships_node)
    workflow.add_node("extract_observations_4", extract_observations_node)
    workflow.add_node("synthesize_findings", synthesize_findings_node)
    
    workflow.set_entry_point("profile_dataset")
    
    workflow.add_edge("profile_dataset", "extract_observations_1")
    workflow.add_edge("extract_observations_1", "analyze_missingness")
    workflow.add_edge("analyze_missingness", "extract_observations_2")
    workflow.add_edge("extract_observations_2", "compute_aggregates")
    workflow.add_edge("compute_aggregates", "extract_observations_3")
    workflow.add_edge("extract_observations_3", "analyze_relationships")
    workflow.add_edge("analyze_relationships", "extract_observations_4")
    workflow.add_edge("extract_observations_4", "synthesize_findings")
    workflow.add_edge("synthesize_findings", END)
    
    app = workflow.compile(checkpointer=checkpointer, name=WORKFLOW_NAME)
    
    return app

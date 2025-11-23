import copy
import csv
import dataclasses
import glob
import json
import os
import re
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List

import rich
import yaml
from jsonargparse import CLI
from rich.progress import Progress

from wxo_agentic_evaluation.arg_configs import ProviderConfig, TestConfig
from wxo_agentic_evaluation.evaluation_package import EvaluationPackage
from wxo_agentic_evaluation.inference_backend import (
    EvaluationController,
    WXOInferenceBackend,
)
from wxo_agentic_evaluation.llm_user import LLMUser
from wxo_agentic_evaluation.metrics.evaluations import Extractor
from wxo_agentic_evaluation.metrics.metrics import (
    CustomEvalMetrics,
    KnowledgeBaseMetricSummary,
    TextMatchType,
    ToolCallAndRoutingMetrics,
)
from wxo_agentic_evaluation.prompt.template_render import (
    LlamaUserTemplateRenderer,
)
from wxo_agentic_evaluation.resource_map import ResourceMap
from wxo_agentic_evaluation.service_provider import (
    LOGGING_ENABLED,
    get_provider,
)
from wxo_agentic_evaluation.service_provider.provider import Provider
from wxo_agentic_evaluation.type import EvaluationData
from wxo_agentic_evaluation.utils import json_dump
from wxo_agentic_evaluation.utils.evaluation_discovery import (
    find_evaluation_subclasses,
)
from wxo_agentic_evaluation.utils.utils import (
    SummaryPanel,
    create_table,
    safe_divide,
)
from wxo_agentic_evaluation.wxo_client import get_wxo_client


def process_test_case(
    task_n: int,
    test_case: str,
    config: TestConfig,
    inference_backend: WXOInferenceBackend,
    resource_map: ResourceMap,
    llm_user: LLMUser,
    llmaaj_provider: Provider,
    run_idx: int = 0,
):
    summary_results_for_path = []
    test_case_name = os.path.basename(test_case).replace(".json", "")
    run_tag = f".run{run_idx+1}" if config.n_runs > 1 else ""

    with open(test_case, "r") as f:
        evaluation_data = EvaluationData.model_validate(json.load(f))

    evaluation_controller = EvaluationController(
        wxo_inference_backend=inference_backend,
        llm_user=llm_user,
        config=config,
    )

    rich.print(
        f"[bold magenta]Running test case: {test_case_name}[/bold magenta]"
    )

    (
        history,
        call_tracker,
        conversational_search_data,
    ) = evaluation_controller.run(
        task_n,
        evaluation_data.story,
        agent_name=evaluation_data.agent,
        starting_user_input=evaluation_data.starting_sentence,
        max_user_turns=evaluation_data.max_user_turns,
    )
    result = list()
    for message in history:
        result.append(message.model_dump())

    json_dump(
        os.path.join(
            config.output_dir,
            "messages",
            f"{test_case_name}{run_tag}.messages.json",
        ),
        result,
    )

    if len(conversational_search_data) > 0:
        fn = f"{test_case_name}{run_tag}.retrieval_context.json"
        out_folder = Path(config.output_dir) / "knowledge_base_metrics"
        out_folder.mkdir(exist_ok=True)
        rc = [context.model_dump() for context in conversational_search_data]
        json_dump(out_folder / fn, rc)

    # If data annotation run, skip summary generation
    if config.data_annotation_run:
        return summary_results_for_path  # empty result set, skip summary

    # Handle custom extractions
    all_extractors = []
    if config.extrators_config.paths is not None:
        for path in config.extrators_config.paths:
            extractors = find_evaluation_subclasses(
                directory=path, base_class_name="Extractor"
            )
            for extractor_class in extractors:
                extractor: Extractor = extractor_class()
                all_extractors.append(extractor)

    # Handle custom evaluations
    all_custom_evals = []
    if config.custom_metrics_config.paths is not None:
        for path in config.custom_metrics_config.paths:
            custom_eval_classes = find_evaluation_subclasses(path)
            for _class in custom_eval_classes:
                custom_eval = _class(llm_client=llmaaj_provider)
                all_custom_evals.append(custom_eval)

    evaluation_package = EvaluationPackage(
        test_case_name=test_case_name,
        messages=history,
        ground_truth=evaluation_data,
        conversational_search_data=conversational_search_data,
        resource_map=resource_map,
        config=config,
        custom_evals=all_custom_evals,
        extractors=all_extractors,
        similarity_threshold=config.similarity_threshold,
        enable_fuzzy_matching=config.enable_fuzzy_matching,
    )
    (
        keyword_semantic_matches,
        knowledge_base_metrics,
        messages_with_reason,
        metrics,
        custom_metrics,
    ) = evaluation_package.generate_summary()
    temp = []
    for message in messages_with_reason:
        temp.append(message.model_dump())
    expected_tools = [
        gd.tool_name
        for gd in evaluation_data.goal_details
        if getattr(gd, "type", None) == "tool_call"
    ]

    raw_actual = []
    for m in history:
        try:
            if getattr(m, "type", None) == "tool_call":
                payload = (
                    json.loads(m.content)
                    if isinstance(m.content, str)
                    else m.content
                )
                name = (payload or {}).get("name")
                if name:
                    raw_actual.append(str(name).strip())
        except Exception:
            pass

    expected_set = set(expected_tools)
    agent_names = (
        set(getattr(resource_map, "agent2tools", {}).keys())
        if resource_map
        else set()
    )

    filtered_actual_tool_calls = [n for n in raw_actual if n not in agent_names]

    missed_tool_calls = sorted(expected_set - set(filtered_actual_tool_calls))

    temp.append(
        {
            "meta": {
                "expected_tool_calls": expected_tools,
                "actual_tool_calls": filtered_actual_tool_calls,
                "missed_tool_calls": missed_tool_calls,
            }
        }
    )
    json_dump(
        os.path.join(
            config.output_dir,
            "messages",
            f"{test_case_name}{run_tag}.messages.analyze.json",
        ),
        temp,
    )

    json_dump(
        os.path.join(
            config.output_dir,
            "messages",
            f"{test_case_name}{run_tag}.metrics.json",
        ),
        metrics.model_dump(),
    )

    metrics.dataset_name = test_case_name
    metrics.avg_resp_time = (
        sum(call_tracker.generic) + sum(call_tracker.tool_call)
    ) / (len(call_tracker.generic) + len(call_tracker.tool_call))
    metrics.avg_resp_time = round(metrics.avg_resp_time, 2)

    summary_results_for_path.append(
        (metrics, knowledge_base_metrics, custom_metrics)
    )

    return summary_results_for_path


def main(config: TestConfig):
    executor = ThreadPoolExecutor(max_workers=config.num_workers)
    if not getattr(config, "skip_available_results", False):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config.output_dir = os.path.join(config.output_dir, ts)
    if config.num_workers > 1 and config.enable_manual_user_input:
        rich.print(
            "[bold yellow]Warning ⚠️: Manual user input is disabled for parallel execution.[/bold yellow]"
        )
        config.enable_manual_user_input = (
            False  # disable manual user input for parallel execution
        )
        # reason: threads continue to stream messages while waiting for user input, which is not desired
        # and the manual input prompt is not labelled properly in the UI
    wxo_client = get_wxo_client(
        config.auth_config.url,
        config.auth_config.tenant_name,
        config.auth_config.token,
    )

    resource_map = ResourceMap(wxo_client)
    inference_backend = WXOInferenceBackend(wxo_client=wxo_client)
    original_provider_config = config.provider_config
    provider_config_dict = asdict(original_provider_config)

    provider_kwargs = {
        "config": ProviderConfig(**provider_config_dict),
        "model_id": config.llm_user_config.model_id,
    }

    if provider_config_dict.get("provider", "gateway") == "gateway":
        provider_kwargs.update(
            token=config.auth_config.token or wxo_client.api_key,
            instance_url=wxo_client.service_url,
        )
        config.auth_config.token = (
            config.auth_config.token or wxo_client.api_key
        )
        config.auth_config.url = (
            config.auth_config.url or wxo_client.service_url
        )

    llm_user = LLMUser(
        wai_client=get_provider(**provider_kwargs),
        template=LlamaUserTemplateRenderer(
            config.llm_user_config.prompt_config
        ),
        user_response_style=config.llm_user_config.user_response_style,
    )

    llamaj_provider_kwargs = copy.deepcopy(provider_kwargs)
    llamaj_config_dict = asdict(llamaj_provider_kwargs["config"])

    llamaj_config_dict["model_id"] = (
        config.custom_metrics_config.llmaaj_config.model_id
    )
    llamaj_config_dict["embedding_model_id"] = (
        config.custom_metrics_config.llmaaj_config.embedding_model_id
    )
    llamaj_provider_kwargs["config"] = ProviderConfig(**llamaj_config_dict)
    llmaaj_provider = get_provider(**llamaj_provider_kwargs)

    print(f"Running evaluation with tenant {config.auth_config.tenant_name}")

    results_list = []

    knowledge_base_output_folder = (
        Path(config.output_dir) / "knowledge_base_metrics"
    )
    knowledge_base_output_folder.mkdir(exist_ok=True, parents=True)
    detailed_rag_output_file = (
        knowledge_base_output_folder / "knowledge_base_detailed_metrics.json"
    )
    summary_rag_output_file = (
        Path(config.output_dir) / "knowledge_base_summary_metrics.json"
    )

    os.makedirs(os.path.join(config.output_dir, "messages"), exist_ok=True)

    def _removesuffix(s, suf):  # py<3.9 safety
        return s[: -len(suf)] if s.endswith(suf) else s

    available_runs = defaultdict(set)
    if config.skip_available_results:
        for f in glob.glob(
            os.path.join(config.output_dir, "messages", "*.messages.json")
        ):
            # strip the fixed tail
            name = _removesuffix(os.path.basename(f), ".messages.json")
            # match either "<stem>" (single run) OR "<stem>.runN" (multi-run)
            m = re.match(r"^(?P<stem>.+?)(?:\.run(?P<run>\d+))?$", name)
            if not m:
                continue
            stem = m.group("stem")
            run_num = int(m.group("run") or 1)  # no suffix ⇒ run 1
            available_runs[stem].add(run_num)

    test_cases: list[str] = []
    for test_path in config.test_paths:
        if os.path.isdir(test_path):
            test_path = os.path.join(test_path, "*.json")
        test_cases.extend(sorted(glob.glob(test_path)))

    futures = []
    task_n = 0
    n_runs = getattr(config, "n_runs", 1)

    for test_case in test_cases:
        if not test_case.endswith(".json") or test_case.endswith("agent.json"):
            continue

        stem = Path(test_case).stem

        for run_idx in range(n_runs):
            run_number = run_idx + 1

            # Skip precisely this (test, run) if results exist
            if config.skip_available_results and (
                run_number in available_runs.get(stem, set())
            ):
                print(
                    f"Skipping {stem} run {run_number} as results already exist."
                )
                continue

            future = executor.submit(
                process_test_case,
                task_n,
                test_case,
                config,
                inference_backend,
                resource_map,
                llm_user,
                llmaaj_provider,
                run_idx,  # 👈 pass run index
            )
            futures.append(((test_case, run_idx), future))
            task_n += 1

    if futures:

        if LOGGING_ENABLED:
            # No progress bar when logging - just process tasks
            for (test_case, run_idx), future in futures:
                try:
                    results_list.extend(future.result())
                except Exception as e:
                    rich.print(f"test case {test_case} fails with {e}")
                    traceback.print_exc()
        else:
            with Progress() as progress:
                task1 = progress.add_task(
                    f"[purple]Evaluating {len(futures)} tasks...",
                    total=len(futures),
                )
                for (test_case, run_idx), future in futures:
                    try:
                        results_list.extend(future.result())
                    except Exception as e:
                        rich.print(f"test case {test_case} fails with {e}")
                        traceback.print_exc()
                    finally:
                        progress.update(task1, advance=1)

    tool_call_metrics = [metric[0] for metric in results_list]
    knowledge_base_metrics = [metric[1] for metric in results_list]
    custom_metrics: List[CustomEvalMetrics] = [
        metric[2] for metric in results_list
    ]

    rag_metric_summary = KnowledgeBaseMetricSummary(
        knowledge_base_metrics=knowledge_base_metrics
    )
    SummaryPanel(rag_metric_summary).print()

    with open(detailed_rag_output_file, "w+", encoding="utf-8") as f:
        json.dump(
            rag_metric_summary.model_dump(by_alias=True)["detailed"],
            f,
            indent=4,
        )

    with open(summary_rag_output_file, "w+", encoding="utf-8") as f:
        json.dump(
            rag_metric_summary.model_dump(by_alias=True)["summary"], f, indent=4
        )

    if len(tool_call_metrics) > 0:
        # remove the average row if exist
        tool_call_metrics = [
            row
            for row in tool_call_metrics
            if row.dataset_name != "Summary (Average)"
        ]

        def filter_display_only_values(
            tool_call_metric: ToolCallAndRoutingMetrics,
        ):
            row = {
                "Dataset": tool_call_metric.dataset_name,
                "Total Steps": tool_call_metric.total_steps,
                "LLM Steps": tool_call_metric.llm_step,
                "Total Tool Calls": tool_call_metric.total_tool_calls,
                "Tool Call Precision": tool_call_metric.tool_call_precision,
                "Tool Call Recall": tool_call_metric.tool_call_recall,
                "Agent Routing Accuracy": tool_call_metric.agent_routing_accuracy,
                "Text Match": tool_call_metric.text_match,
                "Journey Success": tool_call_metric.is_success,
                "Avg Resp Time (sec)": tool_call_metric.avg_resp_time,
            }
            return row

        def create_avg_row(metrics: List[dict]):
            avg_row = {
                "Dataset": "Summary (Average)",
                "Runs": 0,
                "Total Steps": 0,
                "LLM Steps": 0,
                "Total Tool Calls": 0,
                "Tool Call Precision": 0,
                "Tool Call Recall": 0,
                "Agent Routing Accuracy": 0,
                "Text Match": 0,
                "Journey Success": 0,
                "Avg Resp Time (sec)": 0,
            }
            if metrics:
                for row in metrics:
                    avg_row["Runs"] += row.get("Runs", 0)
                    avg_row["Total Steps"] += row["Total Steps"]
                    avg_row["LLM Steps"] += row["LLM Steps"]
                    avg_row["Total Tool Calls"] += row["Total Tool Calls"]
                    avg_row["Tool Call Precision"] += row["Tool Call Precision"]
                    avg_row["Tool Call Recall"] += row["Tool Call Recall"]
                    avg_row["Agent Routing Accuracy"] += row[
                        "Agent Routing Accuracy"
                    ]
                    avg_row["Text Match"] += row["Text Match"]
                    avg_row["Journey Success"] += row["Journey Success"]
                    avg_row["Avg Resp Time (sec)"] += row["Avg Resp Time (sec)"]

                n = len(metrics)
                # Average over datasets
                avg_row["Runs"] = round(safe_divide(avg_row["Runs"], n), 2)
                avg_row["Total Steps"] = round(
                    safe_divide(avg_row["Total Steps"], n), 2
                )
                avg_row["LLM Steps"] = round(
                    safe_divide(avg_row["LLM Steps"], n), 2
                )
                avg_row["Total Tool Calls"] = round(
                    safe_divide(avg_row["Total Tool Calls"], n), 2
                )
                avg_row["Tool Call Precision"] = round(
                    safe_divide(avg_row["Tool Call Precision"], n), 2
                )
                avg_row["Tool Call Recall"] = round(
                    safe_divide(avg_row["Tool Call Recall"], n), 2
                )
                avg_row["Agent Routing Accuracy"] = round(
                    safe_divide(avg_row["Agent Routing Accuracy"], n), 2
                )
                avg_row["Text Match"] = round(
                    safe_divide(avg_row["Text Match"], n), 2
                )
                avg_row["Journey Success"] = round(
                    safe_divide(avg_row["Journey Success"], n), 2
                )
                avg_row["Avg Resp Time (sec)"] = round(
                    safe_divide(avg_row["Avg Resp Time (sec)"], n), 2
                )

            return avg_row

        grouped = defaultdict(list)
        for m in tool_call_metrics:
            grouped[m.dataset_name].append(filter_display_only_values(m))

        numeric_keys = [
            "Total Steps",
            "LLM Steps",
            "Total Tool Calls",
            "Tool Call Precision",
            "Tool Call Recall",
            "Agent Routing Accuracy",
            "Avg Resp Time (sec)",
        ]

        def mean(vals):
            return round(sum(vals) / len(vals), 2) if vals else None

        def _to_pct(value, decimals=0):
            if value is None:
                return "NA"
            try:
                return f"{round(float(value) * 100, decimals)}%"
            except Exception:
                return "NA"

        per_test_rows = []
        for ds, rows in grouped.items():
            out = {"Dataset": ds}
            # Average numeric columns over runs
            for k in numeric_keys:
                out[k] = mean(
                    [r[k] for r in rows if isinstance(r.get(k), (int, float))]
                )

            # Add total runs per dataset
            out["Runs"] = round(float(len(rows)), 2)

            # Journey Success -> numeric fraction in [0,1]
            js_vals = [1 if bool(r.get("Journey Success")) else 0 for r in rows]
            out["Journey Success"] = round(
                safe_divide(sum(js_vals), len(js_vals)), 2
            )

            # Text Match -> numeric fraction in [0,1]
            tm_hits = 0
            tm_den = len(rows)
            for r in rows:
                val = r.get("Text Match")
                if str(val).strip() == TextMatchType.text_match.value:
                    tm_hits += 1
            out["Text Match"] = round(safe_divide(tm_hits, tm_den), 2)

            per_test_rows.append(out)

        # Keep the old overall-avg logic: apply it over the per-test rows (each test counted once)
        overall_row = create_avg_row(per_test_rows)
        tool_call_metrics_for_display = per_test_rows + [overall_row]

        column_order = [
            "Dataset",
            "Runs",
            "Total Steps",
            "LLM Steps",
            "Total Tool Calls",
            "Tool Call Precision",
            "Tool Call Recall",
            "Agent Routing Accuracy",
            "Text Match",
            "Journey Success",
            "Avg Resp Time (sec)",
        ]
        for row in tool_call_metrics_for_display:
            row["Text Match"] = _to_pct(row.get("Text Match"), decimals=0)
            row["Journey Success"] = _to_pct(
                row.get("Journey Success"), decimals=0
            )

        tool_call_metrics_for_display = [
            {col: row.get(col, "") for col in column_order}
            for row in tool_call_metrics_for_display
        ]
        tool_call_table_for_display = create_table(
            tool_call_metrics_for_display
        )

        if tool_call_table_for_display:
            tool_call_table_for_display.print()

    if len(tool_call_metrics) > 0:
        tool_call_metrics = [
            metric.model_dump() for metric in tool_call_metrics
        ]
        output_file = os.path.join(config.output_dir, "summary_metrics.csv")
        header = list(tool_call_metrics[0].keys())

        with open(output_file, "w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(header)
            for entry in tool_call_metrics:
                csv_writer.writerow([entry[name] for name in header])
    # Check if any custom metrics have been calculated
    if any([m.custom_metrics for m in custom_metrics]):
        custom_metrics_display_data = []
        for metric in custom_metrics:
            row = {}
            row["dataset_name"] = metric.dataset_name
            for metric in metric.custom_metrics:
                row[metric.eval_name] = metric.value
            custom_metrics_display_data.append(row)
        create_table(
            custom_metrics_display_data, title="Custom Metrics"
        ).print()

    with open(
        os.path.join(config.output_dir, "config.yml"), "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(dataclasses.asdict(config), f)

    print(f"Results saved to {config.output_dir}")


if __name__ == "__main__":
    main(CLI(TestConfig, as_positional=False))

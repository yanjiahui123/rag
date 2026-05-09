from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional, TypeVar

T = TypeVar("T")
SearchTask = Callable[[], Optional[Iterable[T]]]

DEFAULT_MAX_PARALLEL_SEARCH_TASKS = 4


def run_ordered_parallel_tasks(
    tasks: Iterable[SearchTask[T]],
    max_workers: int = DEFAULT_MAX_PARALLEL_SEARCH_TASKS,
) -> List[T]:
    task_list = [task for task in tasks if task is not None]
    if not task_list:
        return []
    if len(task_list) == 1:
        return _normalize_task_result(task_list[0]())

    worker_count = min(len(task_list), max(int(max_workers or 1), 1))
    results_by_order = {}
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_to_order = {pool.submit(task): order for order, task in enumerate(task_list)}
        for future in as_completed(future_to_order):
            results_by_order[future_to_order[future]] = _normalize_task_result(future.result())

    results: List[T] = []
    for order in range(len(task_list)):
        results.extend(results_by_order.get(order, []))
    return results


def _normalize_task_result(result: Optional[Iterable[T]]) -> List[T]:
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return list(result)


def query_strategy_requires_embedding(query_strategy) -> bool:
    return getattr(query_strategy, "name", str(query_strategy)) != "FULL_TEXT_QUERY"

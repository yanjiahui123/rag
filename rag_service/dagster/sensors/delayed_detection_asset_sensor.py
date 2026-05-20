from datetime import datetime, time
from typing import List

import pytz
from dagster import DefaultSensorStatus, RunRequest, SensorResult, SkipReason, sensor
from sqlmodel import Session, select

from rag_service.constants import DELAYED_JOB_START_TIME, DELAYED_JOB_END_TIME
from rag_service.dagster import delayed_detection_job
from rag_service.dagster.assets.delayed_document_detection_asset import delayed_detection_asset
from rag_service.dagster.dagster_common_op import (
    dummy_delay_consumer,
    load_document_pairs,
)
from rag_service.dagster.partitions.knowledge_base_asset_partition import knowledge_base_asset_partitions_def
from rag_service.database import engine
from rag_service.models.database.models import AutoJobInstances, ServiceConfig
from rag_service.models.enums import JobStatus, VectorizationJobType
from rag_service.utils.dagster_util import generate_asset_partition_key, get_max_op_concurrency
from rag_service.utils.db_util import change_vectorization_job_status, get_pending_jobs_by_type


@sensor(job=delayed_detection_job, default_status=DefaultSensorStatus.RUNNING)
def delayed_detection_asset_sensor():
    """监视延迟至半夜执行的检测任务，若未到达执行时间窗口，则不下发任务"""
    delayed_job_start_time = _get_delay_job_time_config("delayed_job_start_time", DELAYED_JOB_START_TIME)
    delayed_job_end_time = _get_delay_job_time_config("delayed_job_end_time", DELAYED_JOB_END_TIME)
    allowed_start_time = time(delayed_job_start_time, 0)
    allowed_end_time = time(delayed_job_end_time, 0)

    # 获取中国时区
    shanghai_tz = pytz.timezone("Asia/Shanghai")
    current_shanghai_time = datetime.now(shanghai_tz)
    with Session(engine) as session:
        pending_jobs: List[AutoJobInstances] = get_pending_jobs_by_type(
            VectorizationJobType.DELAYED_DOCUMENT_DETECTION, session
        )

        if not pending_jobs:
            return SkipReason("No pending vectorization jobs.")
        if not _is_within_time_window(current_shanghai_time, allowed_start_time, allowed_end_time):
            return SkipReason("当前时间不在允许的时间窗口内")

        for job in pending_jobs:
            change_vectorization_job_status(session, job, JobStatus.STARTED)

        return SensorResult(
            run_requests=[
                RunRequest(
                    partition_key=generate_asset_partition_key(job.knowledge_base_asset),
                    run_config={
                        "ops": {
                            delayed_detection_asset.node_def.name: {
                                "ops": {
                                    load_document_pairs.name: {"config": {"job_id": str(job.id)}},
                                    dummy_delay_consumer.name: {"config": {"job_id": str(job.id)}},
                                }
                            }
                        },
                        "execution": {
                            "config": {
                                "multiprocess": {
                                    "max_concurrent": get_max_op_concurrency(),
                                },
                            }
                        },
                    },
                )
                for job in pending_jobs
            ],
            dynamic_partitions_requests=[
                knowledge_base_asset_partitions_def.build_add_request([
                    generate_asset_partition_key(job.knowledge_base_asset) for job in pending_jobs
                ])
            ],
        )


def _is_within_time_window(current_time: datetime, start_time: time, end_time: time) -> bool:
    """
    检查当前时间是否在指定的时间窗口内

    Args:
        current_time: 当前时间（带时区信息）
        start_time: 窗口开始时间
        end_time: 窗口结束时间

    Returns:
        bool: 是否在时间窗口内
    """
    current_time_only = current_time.time()

    # 处理跨午夜的情况（如 22:00 - 06:00）
    if start_time <= end_time:
        # 正常情况：如 09:00 - 17:00
        return start_time <= current_time_only <= end_time
    # 跨午夜情况：如 22:00 - 06:00
    return current_time_only >= start_time or current_time_only <= end_time


def _get_delay_job_time_config(config_name: str, default_value: int) -> int:
    with Session(engine) as session:
        result = session.exec(
            select(ServiceConfig.value).where(ServiceConfig.name == config_name)
        ).one_or_none()
        return int(result) if result is not None else default_value
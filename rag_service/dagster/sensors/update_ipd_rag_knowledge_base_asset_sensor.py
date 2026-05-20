from typing import List

from dagster import DefaultSensorStatus, RunRequest, SensorResult, SkipReason, sensor
from sqlmodel import Session

from rag_service.dagster.assets.updated_ipd_rag_knowledge_base_asset import (
    update_ipd_rag_knowledge_base_asset,
)
from rag_service.dagster.dagster_common_op import (
    change_vectorization_job_status_to_started,
    change_vectorization_job_status_to_success,
)
from rag_service.dagster.jobs.update_ipd_rag_knowledge_base_asset_job import update_ipd_rag_knowledge_base_asset_job
from rag_service.dagster.partitions.knowledge_base_asset_partition import knowledge_base_asset_partitions_def
from rag_service.database import engine
from rag_service.models.database.models import AutoJobInstances
from rag_service.models.enums import JobStatus, VectorizationJobType
from rag_service.utils.dagster_util import generate_asset_partition_key, get_max_op_concurrency
from rag_service.utils.db_util import change_vectorization_job_status, get_pending_jobs_by_type


@sensor(job=update_ipd_rag_knowledge_base_asset_job, default_status=DefaultSensorStatus.RUNNING)
def update_ipd_rag_knowledge_base_asset_sensor():
    with Session(engine) as session:
        pending_jobs: List[AutoJobInstances] = get_pending_jobs_by_type(
            VectorizationJobType.INCREMENTAL_IN_IPD_RAG, session
        )
        if not pending_jobs:
            return SkipReason("No pending vectorization jobs.")

        for job in pending_jobs:
            change_vectorization_job_status(session, job, JobStatus.STARTING)

        return SensorResult(
            run_requests=[
                RunRequest(
                    partition_key=generate_asset_partition_key(job.knowledge_base_asset),
                    run_config={
                        "ops": {
                            update_ipd_rag_knowledge_base_asset.node_def.name: {
                                "ops": {
                                    change_vectorization_job_status_to_started.name: {
                                        "config": {"job_id": str(job.id)}
                                    },
                                    change_vectorization_job_status_to_success.name: {
                                        "config": {"job_id": str(job.id)}
                                    },
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
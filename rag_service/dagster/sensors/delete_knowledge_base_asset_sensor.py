from typing import List

from dagster import DefaultSensorStatus, RunRequest, SensorResult, SkipReason, sensor
from sqlmodel import Session

from rag_service.dagster.dagster_common_op import (
    change_vectorization_job_status_to_started,
)
from rag_service.dagster.jobs.delete_knowledge_base_asset_job import (
    delete_knowledge_base_asset_job, delete_knowledge_base_asset_database,
)
from rag_service.database import engine
from rag_service.models.database.models import AutoJobInstances
from rag_service.models.enums import JobStatus, VectorizationJobType
from rag_service.utils.dagster_util import generate_asset_partition_key
from rag_service.utils.db_util import change_vectorization_job_status, get_pending_jobs_by_type


@sensor(job=delete_knowledge_base_asset_job, default_status=DefaultSensorStatus.RUNNING)
def delete_knowledge_base_asset_sensor():
    with Session(engine) as session:
        pending_jobs: List[AutoJobInstances] = get_pending_jobs_by_type(VectorizationJobType.DELETE, session)

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
                            change_vectorization_job_status_to_started.name: {
                                "config": {"job_id": str(job.id)}
                            },
                            delete_knowledge_base_asset_database.name: {
                                "config": {"job_id": str(job.id)}
                            }
                        }
                    },
                )
                for job in pending_jobs
            ]
        )
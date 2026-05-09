import uuid

from dagster import In, Nothing, OpExecutionContext, job, op
from sqlmodel import Session, select

from rag_service.dagster.dagster_common_op import (
    change_vectorization_job_status_to_started,
    delete_knowledge_base_asset_resources,
    delete_knowledge_base_asset_vector_store,
)
from rag_service.dagster.partitions.knowledge_base_asset_partition import (
    knowledge_base_asset_partitions_def,
    knowledge_base_asset_partitions_def_config,
)
from rag_service.database import engine
from rag_service.models.database.models import AutoJobInstances
from rag_service.models.enums import VectorizationJobType, JobStatus
from rag_service.rag_app.dao.knowledge_base_asset_dao import delete_documents_by_kb_sn_and_kba_name
from rag_service.utils.dagster_util import parse_asset_partition_key
from rag_service.utils.db_util import get_knowledge_base_asset, get_knowledge_base_by_kb_sn
from rag_service.utils.time_util import now_with_time_zone


@op(ins={"no_input": In(Nothing)})
def change_knowledge_base_update_at(context: OpExecutionContext):
    """
    更新知识库变更时间
    """
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base = get_knowledge_base_by_kb_sn(session, knowledge_base_serial_number)
        knowledge_base.updated_at = now_with_time_zone()
        session.commit()


@op(ins={"no_input": In(Nothing)})
def delete_knowledge_base_asset_database(context: OpExecutionContext):
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
        vectorization_job = session.exec(
            select(AutoJobInstances).where(AutoJobInstances.id == uuid.UUID(context.op_config["job_id"]))
        ).one_or_none()

        if not vectorization_job:
            return

        kb_allow_synchronous_update = knowledge_base_asset.knowledge_base.allow_synchronous_update
        if kb_allow_synchronous_update and vectorization_job.job_type == VectorizationJobType.DELETE:
            job_type = VectorizationJobType.DELETE_IN_IPD_RAG
            ipd_rag_job = AutoJobInstances(status=JobStatus.PENDING, job_type=job_type)
            knowledge_base_asset.auto_job_instances.append(ipd_rag_job)
            session.add(knowledge_base_asset)
            session.commit()
        else:
            delete_documents_by_kb_sn_and_kba_name(session, knowledge_base_serial_number, knowledge_base_asset_name)
            session.commit()


@op(ins={"no_input": In(Nothing)})
def delete_knowledge_base_asset_partition(context: OpExecutionContext):
    context.instance.delete_dynamic_partition(
        knowledge_base_asset_partitions_def_config.partitions_def.name, context.partition_key
    )


@job(partitions_def=knowledge_base_asset_partitions_def)
def delete_knowledge_base_asset_job():
    delete_knowledge_base_asset_partition(
        change_knowledge_base_update_at(
            delete_knowledge_base_asset_database(
                delete_knowledge_base_asset_resources(
                    delete_knowledge_base_asset_vector_store(change_vectorization_job_status_to_started())
                )
            )
        )
    )

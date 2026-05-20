from dagster import define_asset_job, RetryPolicy

from rag_service.dagster.assets.delayed_document_detection_asset import delayed_detection_asset

DELAYED_DETECTION_JOB = "delayed_detection_job"
delayed_detection_job = define_asset_job(
    DELAYED_DETECTION_JOB,
    [delayed_detection_asset],
    tags={
        "dagster/max_retries": "0"  # 这会覆盖全局配置
    }
)
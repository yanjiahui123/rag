import re
from pathlib import Path
from typing import BinaryIO, List, Union

from obs import ObsClient
from obs.model import GetResult, ListObjectsResponse, ObjectStream
from tenacity import retry, stop_after_attempt

from rag_service.config import OBS_AK, OBS_BUCKET, OBS_ENDPOINT, OBS_SK
from rag_service.exceptions import ObsException
from rag_service.logger import Module, get_logger

logger = get_logger(module=Module.OBS)

VALID_OBS_KEY_REGEX = r"\w #\-.\[\]^_`{}"


def _get_obs_client(ak=OBS_AK, sk=OBS_SK, server=OBS_ENDPOINT) -> ObsClient:
    return ObsClient(
        access_key_id=ak,
        secret_access_key=sk,
        server=server,
        path_style=True,
        signature="v2",
        is_signature_negotiation=True,
    )


def get_obs_key(knowledge_base_serial_number: str, asset_name: str, file_name: str) -> str:
    return (Path(knowledge_base_serial_number) / asset_name / file_name).as_posix()


def get_obs_dir(knowledge_base_serial_number: str, asset_name: str) -> str:
    return (Path(knowledge_base_serial_number) / asset_name).as_posix() + "/"


@retry(stop=stop_after_attempt(2), reraise=True)
def upload_file(object_key: str, file_path: str, bucket=OBS_BUCKET) -> str:
    logger.info("Uploading file <%s> to bucket <%s>.", file_path, bucket)
    resp: GetResult = _get_obs_client().putFile(bucket, object_key, file_path)
    if resp.status < 300:
        logger.info("Successfully upload file <%s> to bucket <%s>, key is <%s>.", file_path, bucket, object_key)
        return object_key
    logger.warning("Failed to upload file <%s> to bucket <%s>, resp: <%s>, retry.", file_path, bucket, resp)
    raise ObsException("文件上传失败")


def upload_file_as_bytes(object_key: str, content: Union[BinaryIO, str], bucket=OBS_BUCKET) -> str:
    logger.info("Uploading content of <%s> to bucket <%s>.", object_key, bucket)
    resp: GetResult = _get_obs_client().putObject(bucket, object_key, content)
    if resp.status < 300:
        logger.info("Successfully upload content of <%s> to bucket <%s>, key is <%s>.", object_key, bucket, object_key)
        return object_key
    logger.warning("Failed to upload content of <%s> to bucket <%s>, resp: <%s> retry.", object_key, bucket, resp)
    raise ObsException("文件上传失败")


@retry(stop=stop_after_attempt(2), reraise=True)
def download_file_as_bytes(object_key: str, bucket=OBS_BUCKET) -> bytes:
    logger.info("Downloading file with object key <%s> from bucket <%s>.", object_key, bucket)
    resp: GetResult = _get_obs_client().getObject(bucket, object_key, loadStreamInMemory=True)
    if resp.status < 300:
        logger.info("Successfully download file with object key <%s> from bucket <%s>.", object_key, bucket)
        content: ObjectStream = resp.body
        return content.buffer
    logger.warning(
        "Failed to download file with object key <%s> from bucket <%s>, resp: <%s>, retry.", object_key, bucket, resp
    )
    raise ObsException("文件下载失败")


@retry(stop=stop_after_attempt(2), reraise=True)
def download_file(object_key: str, file_path: str, bucket=OBS_BUCKET) -> None:
    logger.info("Downloading file with object key <%s> from bucket <%s> to <%s>.", object_key, bucket, file_path)
    resp: GetResult = _get_obs_client().getObject(bucket, object_key, downloadPath=file_path)
    if resp.status < 300:
        logger.info(
            "Successfully download file with object key <%s> from bucket <%s> to <%s>.", object_key, bucket, file_path
        )
    else:
        logger.warning(
            "Failed to download file with object key <%s> from bucket <%s> to <%s>, " "resp: <%s>, retry.",
            object_key,
            bucket,
            file_path,
            resp,
        )
        raise ObsException("文件下载失败")


@retry(stop=stop_after_attempt(2), reraise=True)
def get_dir_objects(obs_dir: str, bucket=OBS_BUCKET) -> List[str]:
    logger.info("Getting objects in <%s> from bucket <%s>.", obs_dir, bucket)
    keys = []
    mark = None
    while True:
        resp: GetResult = _get_obs_client().listObjects(bucket, obs_dir, marker=mark)
        if resp.status < 300:
            logger.info("Successfully get objects in <%s> from bucket <%s>.", obs_dir, bucket)
            body: ListObjectsResponse = resp.body
            keys.extend([_["key"] for _ in body.contents])
            if resp.body.is_truncated:
                mark = resp.body.next_marker
            else:
                break
        else:
            logger.warning("Failed to get objects in <%s> from bucket <%s>, resp: <%s>, retry.", obs_dir, bucket, resp)
            raise ObsException("获取目录文件失败")
    return keys


@retry(stop=stop_after_attempt(2), reraise=True)
def delete_object(object_key: str, bucket=OBS_BUCKET) -> None:
    logger.info("Deleting object <%s> from bucket <%s>.", object_key, bucket)
    resp: GetResult = _get_obs_client().deleteObject(bucket, object_key)
    if resp.status < 300:
        logger.info("Successfully delete object <%s> from bucket <%s>.", object_key, bucket)
    else:
        logger.warning("Failed to delete object <%s> from bucket <%s>, resp: <%s>, retry.", object_key, bucket, resp)
        raise ObsException("删除文件失败")


@retry(stop=stop_after_attempt(2), reraise=True)
def delete_dir(obs_dir: str, bucket=OBS_BUCKET) -> None:
    logger.info("Deleting dir in <%s> from bucket <%s>.", obs_dir, bucket)
    for object_key in get_dir_objects(obs_dir, bucket):
        delete_object(object_key, bucket)
    delete_object(obs_dir, bucket)
    logger.info("Successfully delete dir in <%s> from bucket <%s>.", obs_dir, bucket)


def create_signed_url(object_key: str, bucket=OBS_BUCKET):
    """
    获取obs桶上文件的临时访问链接
    :param object_key: 对象key
    :param bucket: 桶
    :return: 临时访问链接
    """
    logger.info("create signed url <%s> from bucket <%s>.", object_key, bucket)
    resp = _get_obs_client().createSignedUrl('GET', bucket, object_key, expires=3600)
    if resp.signedUrl:
        logger.info("Successfully create signed url <%s> from bucket <%s>.", object_key, bucket)
        return resp.signedUrl
    else:
        logger.warning("Failed to create signed url <%s> from bucket <%s>, resp: <%s>, retry.", object_key, bucket, resp)
        raise ObsException("生成obs的url失败")


def is_valid_object_key(object_key: str):
    return bool(re.match(rf"^[{VALID_OBS_KEY_REGEX}]+$", object_key))


def unify_object_key(object_key: str):
    return re.sub(rf"[^{VALID_OBS_KEY_REGEX}]", "_", object_key)


@retry(stop=stop_after_attempt(2), reraise=True)
def obs_copy_object(source_object_key: str, dest_object_key: str, bucket=OBS_BUCKET):
    logger.info(
        f"Copying source object key <{source_object_key}> to dest object key <{dest_object_key}> from bucket <bucket>.")
    resp = _get_obs_client().copyObject(bucket, source_object_key, bucket, dest_object_key)
    if resp.status < 300:
        logger.info(
            f"Successfully copy object key <{source_object_key}> to dest object key <{dest_object_key}> from bucket <{bucket}>.")
    else:
        logger.warning(
            f"Failed to copy object key <{source_object_key}> to dest object key <{dest_object_key}> from bucket <{bucket}>, resp: <{resp}>, retry.")
        raise ObsException("OBS桶内复制文件失败")
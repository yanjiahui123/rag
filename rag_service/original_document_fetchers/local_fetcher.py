import datetime
import json
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import List, Tuple

from sqlmodel import Session

from rag_service.constants import DELETE_ORIGINAL_DOCUMENT_METADATA, DELETE_ORIGINAL_DOCUMENT_METADATA_KEY, \
    DOC_IMPORT_FAIL_KIA
from rag_service.database import engine
from rag_service.document_loaders.structured_artifacts import build_document_download_key
from rag_service.models.database.models import OriginalDocument as OriginalDocumentEntity, VectorStore
from rag_service.models.enums import AssetType, DocumentLoadStatus
from rag_service.models.generic.models import OriginalDocument
from rag_service.original_document_fetchers.base import BaseFetcher
from rag_service.utils.dagster_util import parse_knowledge_base_asset_root_dir, generate_asset_partition_key
from rag_service.utils.db_util import get_knowledge_base_asset
from rag_service.utils.his_util.obs_util import download_file, get_dir_objects, get_obs_dir, upload_file
from rag_service.utils.kia_check_util import get_kia_file_path_list
from rag_service.utils.online_url_util import get_online_url_info_from_json, merge_online_url_info
from rag_service.utils.redis_util import rc, vector_store_name_keyspace
from rag_service.utils.time_util import now_with_time_zone


class LocalFetcher(BaseFetcher, asset_types={AssetType.UPLOADED_ASSET, AssetType.LARGE_ASSET, AssetType.QA_PAIR}):
    @staticmethod
    def get_knowledge_base_asset_id(knowledge_base_sn: str, knowledge_base_asset_name: str) -> str:
        with Session(engine) as session:
            knowledge_base_asset = get_knowledge_base_asset(
                session, knowledge_base_sn, knowledge_base_asset_name
            )
            return str(knowledge_base_asset.id)

    def upload_download_file(self, knowledge_base_asset_id: str, doc_id: uuid.UUID, path: Path, root_path: Path):
        if not self._save_file:
            return None
        download_key = build_document_download_key(
            knowledge_base_asset_id,
            str(doc_id),
            str(path.relative_to(root_path)),
        )
        return upload_file(download_key, str(path))

    @staticmethod
    def get_online_url_info(local_file_online_dic, path, root_path):
        online_url_info = local_file_online_dic.get(str(path.relative_to(root_path)), "")
        online_url, online_url_type, video_start_time = get_online_url_info_from_json(online_url_info)
        return online_url, online_url_type, video_start_time

    @staticmethod
    def get_vector_store(knowledge_base_asset):
        if knowledge_base_asset.vector_stores:
            return knowledge_base_asset.vector_stores[-1]
        asset_partition_key = generate_asset_partition_key(knowledge_base_asset)
        vector_store_name = rc.get(vector_store_name_keyspace.resolve(asset_partition_key))
        if not vector_store_name:
            vector_store_name = uuid.uuid4().hex
            rc.set(vector_store_name_keyspace.resolve(asset_partition_key), vector_store_name,
                   px=60 * 60 * 1000)
        vector_store = VectorStore(name=vector_store_name)
        knowledge_base_asset.vector_stores.append(vector_store)
        return vector_store

    def handle_kia_file(self, kia_file_path_list, knowledge_base_sn, knowledge_base_asset_name):
        with Session(engine) as session:
            knowledge_base_asset = get_knowledge_base_asset(
                session, knowledge_base_sn, knowledge_base_asset_name
            )
            vector_store = self.get_vector_store(knowledge_base_asset)
            root_path = Path(self._asset_root_dir)
            # kia文档直接保存到数据库，不用后续的向量化处理
            for kia_file_name in kia_file_path_list:
                path = root_path / Path(kia_file_name).name
                vector_store.original_documents.append(
                    OriginalDocumentEntity(
                        uri=str(path),
                        source=str(path.relative_to(root_path)),
                        mtime=now_with_time_zone(),
                        extended_metadata={
                            "title": str(path.relative_to(root_path)),
                            "kb_sn": knowledge_base_sn,
                            "asset_name": knowledge_base_asset_name,
                        },
                        document_load_status=DocumentLoadStatus.KIA_INTERCEPT,
                        error_info=DOC_IMPORT_FAIL_KIA

                    ))
            session.add(knowledge_base_asset)
            session.commit()

    def fetch(self, knowledge_base_sn: str, knowledge_base_asset_name: str) -> Generator[OriginalDocument, None, None]:
        root_path = Path(self._asset_root_dir)
        local_file_online_dic = self._local_file_online_dic if self._local_file_online_dic else {}
        knowledge_base_asset_id = (
            self.get_knowledge_base_asset_id(knowledge_base_sn, knowledge_base_asset_name)
            if self._save_file else ""
        )
        object_key_list = get_dir_objects(get_obs_dir(*parse_knowledge_base_asset_root_dir(root_path)))
        # 查询kia文件
        kia_file_path_list = get_kia_file_path_list(object_key_list)
        if kia_file_path_list:
            self.handle_kia_file(kia_file_path_list, knowledge_base_sn, knowledge_base_asset_name)
        for object_key in object_key_list:
            if kia_file_path_list and object_key in kia_file_path_list:
                # kia文件前面已经处理，不用后续的向量化操作，此处跳过
                continue

            path = Path(self._asset_root_dir) / Path(object_key).name
            download_file(object_key, str(path))
            doc_id = uuid.uuid4()
            download_key = self.upload_download_file(knowledge_base_asset_id, doc_id, path, root_path)
            online_url, online_url_type, video_start_time = self.get_online_url_info(local_file_online_dic, path,
                                                                                     root_path)
            yield OriginalDocument(
                doc_id=doc_id,
                uri=str(path),
                source=str(path.relative_to(root_path)),
                mtime=datetime.datetime.fromtimestamp(
                    path.lstat().st_mtime, tz=datetime.timezone(datetime.timedelta(hours=8))
                ),
                download_key=download_key,
                extended_metadata={
                    "title": str(path.relative_to(root_path)),
                    "kb_sn": knowledge_base_sn,
                    "asset_name": knowledge_base_asset_name,
                },
                online_url=merge_online_url_info(online_url_type, online_url, video_start_time)
            )

    def update_fetch(
        self, knowledge_base_sn: str, knowledge_base_asset_name: str
    ) -> Tuple[List[str], List[str], List[OriginalDocument]]:
        root_path = Path(self._asset_root_dir)
        uploaded_original_document_sources: List[str] = []
        uploaded_original_documents: List[OriginalDocument] = []
        local_file_online_dic = self._local_file_online_dic if self._local_file_online_dic else {}
        knowledge_base_asset_id = (
            self.get_knowledge_base_asset_id(knowledge_base_sn, knowledge_base_asset_name)
            if self._save_file else ""
        )
        object_key_list = get_dir_objects(get_obs_dir(*parse_knowledge_base_asset_root_dir(root_path)))
        # 查询kia文件
        kia_file_path_list = get_kia_file_path_list(object_key_list)
        if kia_file_path_list:
            self.handle_kia_file(kia_file_path_list, knowledge_base_sn, knowledge_base_asset_name)
        for object_key in object_key_list:
            if kia_file_path_list and object_key in kia_file_path_list:
                # kia文件前面已经处理，不用后续的向量化操作，此处跳过
                continue

            path = Path(self._asset_root_dir) / Path(object_key).name
            download_file(object_key, str(path))
            if Path(object_key).name == DELETE_ORIGINAL_DOCUMENT_METADATA:
                continue
            doc_id = uuid.uuid4()
            download_key = self.upload_download_file(knowledge_base_asset_id, doc_id, path, root_path)
            uploaded_original_document_sources.append(str(path.relative_to(root_path)))
            online_url, online_url_type, video_start_time = self.get_online_url_info(local_file_online_dic, path,
                                                                                     root_path)
            uploaded_original_documents.append(
                OriginalDocument(
                    doc_id=doc_id,
                    uri=str(path),
                    source=str(path.relative_to(root_path)),
                    mtime=datetime.datetime.fromtimestamp(
                        path.lstat().st_mtime, tz=datetime.timezone(datetime.timedelta(hours=8))
                    ),
                    download_key=download_key,
                    extended_metadata={
                        "title": str(path.relative_to(root_path)),
                        "kb_sn": knowledge_base_sn,
                        "asset_name": knowledge_base_asset_name,
                    },
                    online_url=merge_online_url_info(online_url_type, online_url, video_start_time)
                )
            )
        delete_original_document_metadata_path = root_path / DELETE_ORIGINAL_DOCUMENT_METADATA
        with delete_original_document_metadata_path.open("r", encoding="utf-8") as file_content:
            delete_original_document_dict = json.load(file_content)
            delete_original_document_sources = delete_original_document_dict[DELETE_ORIGINAL_DOCUMENT_METADATA_KEY]
        return delete_original_document_sources, uploaded_original_document_sources, uploaded_original_documents

import random
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain.docstore.document import Document
from sqlmodel import Session

from rag_service.constants import DEFAULT_SAMPLE_SIZE
from rag_service.database import engine
from rag_service.logger import Module, get_logger
from rag_service.models.database.models import DocumentDefect, OriginalDocument
from rag_service.models.enums import CorpusAccessControlPolicy
from rag_service.models.generic.models import DetectionMetadata
from rag_service.rag_app.dao.document_dao import query_document_by_ids

logger = get_logger(module=Module.DETECTION)


class CorpusDetector(ABC):
    def __init__(self):
        ...

    def __init_subclass__(cls, detector_name: str, description: str, **kwargs):
        super().__init_subclass__(**kwargs)

    @abstractmethod
    def detect(
        self,
        doc_infos: Optional[List[Tuple[OriginalDocument, List[Document]]]] = None,
        job_id: Optional[UUID] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[DetectionMetadata]:
        """
        检测文本内容的主要方法

        Args:
            doc_infos: 待检测的文本信息以及对应的文本切片列表
            job_id: 可选，关联任务的id
            config: 可选，额外的元数据信息

        Returns:
            List[DetectionMetadata]: 检测结果信息
        """

    def save_record(
        self,
        job_id: UUID,
        rule_code: int,
        detection_context: List[DetectionMetadata],
        kb_id: Optional[UUID] = None,
        kba_id: Optional[UUID] = None,
    ):
        if not detection_context:
            return
        doc_ids, records = [], []
        for detection_info in detection_context:
            document_defect = DocumentDefect(
                job_id=job_id,
                rule_code=rule_code,
                detection_context=detection_info.dict(),
                compare_doc_id=detection_info.segment_comparison_raw_info.doc_id
                if detection_info.segment_comparison_raw_info
                else None,
                kb_id=kb_id if kb_id else None,
                kba_id=kba_id if kba_id else None,
                operation=detection_info.operation,
            )
            info_source = (
                detection_info.duplicate_segment_info
                if rule_code == CorpusAccessControlPolicy.SEMANTIC_INTEGRITY.value
                else detection_info.segment_row_info
            )
            document_defect.doc_id = info_source.doc_id if info_source else None
            records.append(document_defect)
            if document_defect.doc_id:
                doc_ids.append(uuid.UUID(document_defect.doc_id))

        with Session(engine) as session:
            docs = query_document_by_ids(session, doc_ids)
            for doc in docs:
                doc.has_defect = True
            session.add_all(docs)
            session.add_all(records)
            session.commit()

    def send_notification(self, message: str):
        pass

    def random_sampling(
        self, doc_infos: List[Tuple[OriginalDocument, List[Document]]], default_sample_size: int = DEFAULT_SAMPLE_SIZE
    ):
        if not doc_infos:
            return []

        num_docs = len(doc_infos)

        # 情况1: doc数量 >= default_sample_size，每个doc随机选1个chunk
        if num_docs >= default_sample_size:
            result = []
            for doc, chunks in doc_infos:
                if chunks:
                    # 随机选择1个chunk
                    selected_chunk = random.choice(chunks)
                    result.append((doc, [selected_chunk]))

            logger.info(f"文档数({num_docs}) >= {default_sample_size}, 每个doc选1个chunk")
            logger.info(f"结果: {num_docs} 个doc, 最多 {sum(len(c) for _, c in result)} 个chunk")
            return result

        # 情况2: doc数量 < 50，平均分配
        # 计算每个doc平均可以选几个chunk
        chunks_per_doc = default_sample_size // num_docs
        remaining_chunks = default_sample_size % num_docs

        result = []
        for i, (doc, chunks) in enumerate(doc_infos):
            if not chunks:
                continue

            # 前remaining_chunks个doc多分配1个
            num_to_select = chunks_per_doc + (1 if i < remaining_chunks else 0)

            # 不能超过该doc实际拥有的chunk数量
            num_to_select = min(num_to_select, len(chunks))

            # 随机选择指定数量的chunks
            if num_to_select > 0:
                selected_chunks = random.sample(chunks, num_to_select)
                result.append((doc, selected_chunks))

        total_selected = sum(len(c) for _, c in result)
        logger.info(
            f"文档数({num_docs}) < {default_sample_size}, 平均每个doc选 {chunks_per_doc}-{chunks_per_doc + 1} 个chunk"
        )
        logger.info(f"结果: {num_docs} 个doc, 共 {total_selected} 个chunk")
        return result

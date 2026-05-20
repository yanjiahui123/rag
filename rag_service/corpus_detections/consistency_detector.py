import json
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain.docstore.document import Document
from sqlmodel import Session

from rag_service.config import CONSISTENCY_DETECT_PROMPT_ID
from rag_service.constants import CONSISTENCE_DOCUMENT_SIZE, CONSISTENCY_DETECT_PROMPT
from rag_service.corpus_detections.detector import CorpusDetector
from rag_service.database import engine
from rag_service.llms.llm import RagLLM
from rag_service.logger import Module, get_logger
from rag_service.models.api.models import RetrievedDocument
from rag_service.models.enums import KnowledgeBaseType, QueryStrategy
from rag_service.models.generic.models import (
    DetectionMetadata,
    OriginalDocument,
    SegmentComparisonRawInfo,
    SegmentRawInfo,
)
from rag_service.utils.db_util import get_embedding_model_and_vector_stores, validate_knowledge_base
from rag_service.utils.prompt_util import get_prompt
from rag_service.vectorstore import get_vector_store_manager

logger = get_logger(module=Module.DETECTION)


class ConsistencyDetector(
    CorpusDetector,
    detector_name="文档一致性检测器",
    description="检测知识库中是否存在语义表达矛盾的文档片段",
):
    @staticmethod
    def get_consistence_prompt():
        return get_prompt(default_prompt=CONSISTENCY_DETECT_PROMPT, prompt_id=CONSISTENCY_DETECT_PROMPT_ID)

    @staticmethod
    def parse_consistence_result(result: str) -> Tuple[int, str]:
        try:
            ans = json.loads(result)
            return ans.get("verdict", 0), ans.get("reason", "")
        except Exception as e:
            logger.exception(e)
            return 0, ""

    def detect(
        self,
        doc_infos: Optional[List[Tuple[OriginalDocument, List[Document]]]] = None,
        job_id: Optional[UUID] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[DetectionMetadata]:
        # 验证输入
        if not doc_infos:
            return []

        kb_sn = config.get("kb_sn")
        operation_details = []
        vs_manager = get_vector_store_manager()
        llm_model = RagLLM()

        with Session(engine) as session:
            knowledge_base = validate_knowledge_base(session, kb_sn)
            embed_model_and_vector_stores = get_embedding_model_and_vector_stores(session, kb_sn)
            analyzer = knowledge_base.analyzer

        # 只对分层知识库进行检测
        if knowledge_base.kb_type != KnowledgeBaseType.LAYER:
            return []

        sample_docs = self.random_sampling(doc_infos)
        for doc, chunks in sample_docs:
            non_consistent_details = self.detect_chunk_consistency(
                doc, chunks, vs_manager, analyzer, embed_model_and_vector_stores, llm_model, job_id
            )
            operation_details.extend(non_consistent_details)
        return operation_details

    def _filter_first_different_doc(
        self, chunk: Document, related_chunks: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        for related_chunk in related_chunks:
            if related_chunk.text != chunk.page_content:
                return [related_chunk]
        return []

    def _get_non_consistent_details(
        self,
        job_id: UUID,
        ori_chunk: Document,
        related_chunks: List[RetrievedDocument],
        llm_model: RagLLM,
        doc: OriginalDocument,
    ):
        operation_details = []
        ori_doc_source = ori_chunk.metadata.get("source", "")
        default_prompt = self.get_consistence_prompt()
        for related_chunk in related_chunks:
            doc_source = related_chunk.metadata.source if related_chunk.metadata.source else ""

            # 同一文档进行过滤
            if ori_doc_source == doc_source:
                continue

            prompt = default_prompt.replace("text_a", ori_chunk.page_content).replace("text_b", related_chunk.text)
            try:
                llm_result = llm_model.invoke(prompt)
            except Exception as e:
                logger.exception(f"任务id为: {job_id} 进行一致性检测调用大模型发生错误", e)
                continue
            consistence_flag, reason = self.parse_consistence_result(llm_result)
            if consistence_flag == 1 and reason:
                logger.info(
                    f"任务id：{job_id}，当前片段doc_id：{doc.doc_id}，矛盾片段来源：{related_chunk.metadata.source}\n"
                    f"原始片段内容：{ori_chunk.page_content}\n矛盾片段内容：{related_chunk.text}"
                )
                segment_row_info = SegmentRawInfo(
                    content=ori_chunk.page_content,
                    kb_sn=ori_chunk.metadata.get("extended_metadata").get("kb_sn", ""),
                    asset_name=ori_chunk.metadata.get("extended_metadata").get("asset_name", ""),
                    source=ori_doc_source,
                    doc_id=str(doc.doc_id),
                )
                segment_comparison_raw_info = SegmentComparisonRawInfo(
                    content=related_chunk.text,
                    kb_sn=related_chunk.metadata.extended_metadata.get("kb_sn", ""),
                    asset_name=related_chunk.metadata.extended_metadata.get("asset_name", ""),
                    source=doc_source,
                    judge_reason=reason,
                    es_doc_id=related_chunk.es_doc_id,
                    doc_id=str(related_chunk.metadata.extended_metadata.get("doc_id", None)),
                )
                operation_details.append(
                    DetectionMetadata(
                        operation=f"片段存在逻辑性问题，判断原因如下：\n{reason}",
                        segment_row_info=segment_row_info,
                        segment_comparison_raw_info=segment_comparison_raw_info,
                    )
                )
        return operation_details

    def detect_chunk_consistency(
        self, doc, chunks, vs_manager, analyzer, embed_model_and_vector_stores, llm_model, job_id
    ):
        result = []
        for chunk in chunks:
            related_chunks = vs_manager.retrieve(
                chunk.page_content,
                CONSISTENCE_DOCUMENT_SIZE,
                embed_model_and_vector_stores,
                collect_info=False,
                analyzer=analyzer,
                query_strategy=QueryStrategy.FULL_TEXT_QUERY,
            )
            if not related_chunks:
                continue
            related_chunks = self._filter_first_different_doc(chunk, related_chunks)
            non_consistent_details = self._get_non_consistent_details(job_id, chunk, related_chunks, llm_model, doc)
            result.extend(non_consistent_details)
        return result
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
import numpy as np
from trulens_eval import ( Feedback, TruLlama, OpenAI )
from trulens_eval.feedback.provider.langchain import Langchain
# trulens_eval==0.31.0，对应langchain-core-0.1.52~0.2.5
# from trulens_eval.feedback.v2.feedback import Groundedness
# trulens_eval==0.18.3
# from trulens_eval.feedback import Groundedness
from groundedness import Groundedness
import nest_asyncio

nest_asyncio.apply()

_ = load_dotenv(find_dotenv())

# 评测llm，这里可以更换为效果更好的模型
llm = ChatOpenAI(
    temperature=0.01,
    model="glm-3-turbo",
    openai_api_key=os.environ["ZHIPUAI_API_KEY"],
    openai_api_base="https://open.bigmodel.cn/api/paas/v4"
)

openai = Langchain(chain=llm)

# 答案相关度的反馈函数最容易设置，因为它仅依赖输入/输出。可以使用新的 TruLens 帮助函数  .on_input_output()
qa_relevance = (
    Feedback(openai.relevance_with_cot_reasons,
             name="Answer Relevance")
    .on_input_output()
)

# 将每个上下文相关度整合为一个分数值。本例中，我们使用最大值衡量相关度，在实现过程中也可使用其他指标，如平均值或最小值。
qs_relevance = (
    Feedback(openai.relevance_with_cot_reasons,
             name="Context Relevance")
    .on_input()
    .on(TruLlama.select_source_nodes().node.text)
    .aggregate(np.mean)
)

# 准确性设置类似，整合过程略有不同。这里，我们取每个语句准确性的最高分和各语句准确性的的平均分。
grounded = Groundedness(groundedness_provider=openai)
groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
)

feedbacks = [qa_relevance, qs_relevance, groundedness]


def get_trulens_recorder(query_engine, feedbacks, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder


def get_prebuilt_trulens_recorder(query_engine, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
        )
    return tru_recorder


from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader, Document, StorageContext, load_index_from_storage, Settings
from llama_index.core.node_parser.text.sentence_window import SentenceWindowNodeParser
from llama_index.legacy.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
import os


def build_sentence_window_index(
    document, llm, embed_model="local:./m3e-base", save_dir="sentence_index"
):
    # create the sentence window node parser default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            [document], service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="./bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


def build_automerging_index(
    documents,
    llm,
    embed_model="local:./m3e-base",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    service_context,
    similarity_top_k=12,
    rerank_top_n=2,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="./bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        service_context=service_context, retriever=retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine
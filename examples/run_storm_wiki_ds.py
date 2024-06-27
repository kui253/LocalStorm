"""
STORM Wiki pipeline powered by GPT-3.5/4 and You.com search engine.
You need to set up the following environment variables to run this script:
    - OPENAI_API_KEY: OpenAI API key
    - OPENAI_API_TYPE: OpenAI API type (e.g., 'openai' or 'azure')
    - AZURE_API_BASE: Azure API base URL if using Azure API
    - AZURE_API_VERSION: Azure API version if using Azure API
    - YDC_API_KEY: You.com API key; or, BING_SEARCH_API_KEY: Bing Search API key

Output will be structured as below
args.output_dir/
    topic_name/  # topic_name will follow convention of underscore-connected topic name w/o space and slash
        conversation_log.json           # Log of information-seeking conversation
        raw_search_results.json         # Raw search results from search engine
        direct_gen_outline.txt          # Outline directly generated with LLM's parametric knowledge
        storm_gen_outline.txt           # Outline refined with collected information
        url_to_info.json                # Sources that are used in the final article
        storm_gen_article.txt           # Final article generated
        storm_gen_article_polished.txt  # Polished final article (if args.do_polish_article is True)
"""

import os
import sys
from argparse import ArgumentParser
from langchain.schema import Document
from BCEmbedding import EmbeddingModel, RerankerModel


sys.path.append("./src")
from lm import OpenAIModel
from file_man import FileManager
from rm_local import LocalRM, get_collection, MyEmbeddingFunction
from lm_ds import DSClient
from rm import YouRM, BingSearch
from storm_wiki.engine_local import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from utils import load_api_key


def main(args):
    load_api_key(toml_file_path="secrets.toml")
    lm_configs = STORMWikiLMConfigs()
    openai_kwargs = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_provider": os.getenv("OPENAI_API_TYPE"),
        "temperature": 1.0,
        "top_p": 0.9,
        "api_base": os.getenv("AZURE_API_BASE"),
        "api_version": os.getenv("AZURE_API_VERSION"),
    }

    # STORM is a LM system so different components can be powered by different models.
    # For a good balance between cost and quality, you can choose a cheaper/faster model for conv_simulator_lm
    # which is used to split queries, synthesize answers in the conversation. We recommend using stronger models
    # for outline_gen_lm which is responsible for organizing the collected information, and article_gen_lm
    # which is responsible for generating sections with citations.

    conv_simulator_lm = DSClient(
        model="deepseek-chat", max_tokens=500, model_type="chat", **openai_kwargs
    )

    question_asker_lm = DSClient(
        model="deepseek-chat", max_tokens=500, model_type="chat", **openai_kwargs
    )
    outline_gen_lm = DSClient(
        model="deepseek-chat", max_tokens=400, model_type="chat", **openai_kwargs
    )
    article_gen_lm = DSClient(
        model="deepseek-chat", max_tokens=700, model_type="chat", **openai_kwargs
    )
    article_polish_lm = DSClient(
        model="deepseek-chat", max_tokens=4000, model_type="chat", **openai_kwargs
    )

    lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )

    # STORM is a knowledge curation system which consumes information from the retrieval module.
    # Currently, the information source is the Internet and we use search engine API as the retrieval module.
    if args.retriever == "bing":
        rm = BingSearch(
            bing_search_api=os.getenv("BING_SEARCH_API_KEY"), k=engine_args.search_top_k
        )
    elif args.retriever == "you":
        rm = YouRM(ydc_api_key=os.getenv("YDC_API_KEY"), k=engine_args.search_top_k)
    elif args.retriever == "local":
        access_token = os.getenv("ACCESS_TOKEN")
        local_embedding_model_weights_dir = os.getenv(
            "LOCAL_EMBEDDING_MODEL_WEIGHTS_DIR"
        )
        local_reranker_weights_dir = os.getenv("LOCAL_RERANKER_WEIGHTS_DIR")
        doc_dir = os.getenv("DOC_DIR")
        reranker = RerankerModel(
            model_name_or_path=local_reranker_weights_dir,
            token=access_token,
            device=f"cuda:{args.reranker_device}",
        )
        embedding_model = EmbeddingModel(
            model_name_or_path=local_embedding_model_weights_dir,
            token=access_token,
            device=f"cuda:{args.embedding_model_device}",
        )
        file_manager = FileManager(doc_dir)
        method_statements = file_manager.get_all_files()

        documents = [i["content"] for i in method_statements]

        # summaries = [i["summary"] for i in method_statements]

        # 支持多文档操作
        # with open(doc_dir, "r") as file:
        #     text_content = file.read()
        #     documents.append(Document(page_content=text_content))

        collection = get_collection(documents, adding=True, embed_model=embedding_model)
        # 本地的retrieve模块完成
        rm = LocalRM(
            reranker=reranker,
            embed_func=MyEmbeddingFunction(embedding_model),
            collection=collection,
            k=engine_args.search_top_k,
        )
        ## test rm part (done)
        # import pdb

        # pdb.set_trace()
        # query = "A report about how to manager the water?"
        # rm_results = rm(query)

    runner = STORMWikiRunner(engine_args, lm_configs, rm)
    if args.debug:
        topic = "A report about how to manage water"
    else:
        topic = input("Topic: ")
    runner.run(
        topic=topic,
        do_research=args.do_research,
        do_generate_outline=args.do_generate_outline,
        do_generate_article=args.do_generate_article,
        do_polish_article=args.do_polish_article,
    )
    runner.post_run()
    runner.summary()


if __name__ == "__main__":
    parser = ArgumentParser()
    # global arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/gpt",
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--max-thread-num",
        type=int,
        default=3,
        help="Maximum number of threads to use. The information seeking part and the article generation"
        "part can speed up by using multiple threads. Consider reducing it if keep getting "
        '"Exceed rate limit" error when calling LM API.',
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["bing", "you", "local"],
        help="The search engine API to use for retrieving information.",
    )
    # stage of the pipeline
    parser.add_argument(
        "--do-research",
        action="store_true",
        help="If True, simulate conversation to research the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-generate-outline",
        action="store_true",
        help="If True, generate an outline for the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-generate-article",
        action="store_true",
        help="If True, generate an article for the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-polish-article",
        action="store_true",
        help="If True, polish the article by adding a summarization section and (optionally) removing "
        "duplicate content.",
    )
    # hyperparameters for the pre-writing stage
    parser.add_argument(
        "--max-conv-turn",
        type=int,
        default=3,
        help="Maximum number of questions in conversational question asking.",
    )
    parser.add_argument(
        "--max-perspective",
        type=int,
        default=3,
        help="Maximum number of perspectives to consider in perspective-guided question asking.",
    )
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=3,
        help="Top k search results to consider for each search query.",
    )
    # hyperparameters for the writing stage
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=3,
        help="Top k collected references for each section title.",
    )
    parser.add_argument(
        "--remove-duplicate",
        action="store_true",
        help="If True, remove duplicate content from the article.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If True, remove duplicate content from the article.",
    )
    parser.add_argument(
        "--embedding_model_device",
        default=0,
        type=int,
        help="If True, remove duplicate content from the article.",
    )
    parser.add_argument(
        "--reranker_device",
        default=1,
        type=int,
        help="If True, remove duplicate content from the article.",
    )
    main(parser.parse_args())

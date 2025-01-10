import logging
from typing import Dict, List

import numpy as np
from tqdm.auto import tqdm

from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider.langchain import Langchain
from trulens_eval.feedback.provider import Provider
from trulens_eval.feedback.provider.bedrock import Bedrock
from trulens_eval.feedback.provider.hugs import Huggingface
from trulens_eval.feedback.provider.litellm import LiteLLM
from trulens_eval.feedback.provider.openai import AzureOpenAI
from trulens_eval.feedback.provider.openai import OpenAI
from trulens_eval.utils.generated import re_0_10_rating
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.serial import SerialModel

logger = logging.getLogger(__name__)

LLM_GROUNDEDNESS_FULL_SYSTEM = """You are a INFORMATION OVERLAP classifier providing the overlap of information between a SOURCE and STATEMENT.
                For every sentence in the statement, please answer with this template:

                TEMPLATE: 
                Statement Sentence: <Sentence>, 
                Supporting Evidence: <Choose the exact unchanged sentences in the source that can answer the statement, if nothing matches, say NOTHING FOUND>
                Score: <Output a number between 0-10 where 0 is no information overlap and 10 is all information is overlapping>
                """

class Groundedness(SerialModel, WithClassInfo):
    """Measures Groundedness.
    """
    groundedness_provider: Provider

    def __init__(self, groundedness_provider: Provider = None):
        """Instantiates the groundedness providers. Currently the groundedness functions work well with a summarizer.
        This class will use an LLM to find the relevant strings in a text. The groundedness_provider can 
        either be an LLM provider (such as OpenAI) or NLI with huggingface.

        Usage 1:
        ```
        from trulens_eval.feedback import Groundedness
        from trulens_eval.feedback.provider.openai import OpenAI
        openai_provider = OpenAI()
        groundedness_imp = Groundedness(groundedness_provider=openai_provider)
        ```

        Usage 2:
        ```
        from trulens_eval.feedback import Groundedness
        from trulens_eval.feedback.provider.hugs import Huggingface
        huggingface_provider = Huggingface()
        groundedness_imp = Groundedness(groundedness_provider=huggingface_provider)
        ```

        Args:
            groundedness_provider (Provider, optional): groundedness provider options: OpenAI LLM or HuggingFace NLI. Defaults to OpenAI().
            summarize_provider (Provider, optional): Internal Usage for DB serialization.
        """

        if groundedness_provider is None:
            groundedness_provider = OpenAI()

        super().__init__(
            groundedness_provider=groundedness_provider,
            obj=self  # for WithClassInfo
        )

    def groundedness_measure(self, source: str, statement: str) -> float:
        """A measure to track if the source material supports each sentence in the statement. 
        This groundedness measure is faster; but less accurate than `groundedness_measure_with_summarize_step` 

        Usage on RAG Contexts:
        ```
        from trulens_eval import Feedback
        from trulens_eval.feedback import Groundedness
        from trulens_eval.feedback.provider.openai import OpenAI
        grounded = feedback.Groundedness(groundedness_provider=OpenAI())


        f_groundedness = feedback.Feedback(grounded.groundedness_measure).on(
            Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content # See note below
        ).on_output().aggregate(grounded.grounded_statements_aggregator)
        ```
        The `on(...)` selector can be changed. See [Feedback Function Guide : Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)


        Args:
            source (str): The source that should support the statement
            statement (str): The statement to check groundedness

        Returns:
            float: A measure between 0 and 1, where 1 means each sentence is grounded in the source.
        """
        logger.warning(
            "Feedback function `groundedness_measure` was renamed to `groundedness_measure_with_cot_reasons`. The new functionality of `groundedness_measure` function will no longer emit reasons as a lower cost option. It may have reduced accuracy due to not using Chain of Thought reasoning in the scoring."
        )

        groundedness_scores = {}
        if isinstance(self.groundedness_provider,
                      (AzureOpenAI, OpenAI, LiteLLM, Bedrock)):
            groundedness_scores[f"full_doc_score"] = re_0_10_rating(
                self.groundedness_provider.
                _groundedness_doc_in_out(source, statement)
            ) / 10
            reason = "Reasons not supplied for non chain of thought function"
        elif isinstance(self.groundedness_provider, Huggingface):
            reason = ""
            for i, hypothesis in enumerate(
                    tqdm(statement.split("."),
                         desc="Groundendess per statement in source")):
                plausible_junk_char_min = 4  # very likely "sentences" under 4 characters are punctuation, spaces, etc
                if len(hypothesis) > plausible_junk_char_min:
                    score = self.groundedness_provider._doc_groundedness(
                        premise=source, hypothesis=hypothesis
                    )
                    reason = reason + str.format(
                        prompts.GROUNDEDNESS_REASON_TEMPLATE,
                        statement_sentence=hypothesis,
                        supporting_evidence="[Doc NLI Used full source]",
                        score=score * 10,
                    )
                    groundedness_scores[f"statement_{i}"] = score

        return groundedness_scores, {"reason": reason}

    def groundedness_measure_with_cot_reasons(
        self, source: str, statement: str
    ) -> float:
        """A measure to track if the source material supports each sentence in the statement. 
        This groundedness measure is faster; but less accurate than `groundedness_measure_with_summarize_step`.
        Also uses chain of thought methodology and emits the reasons.

        Usage on RAG Contexts:
        ```
        from trulens_eval import Feedback
        from trulens_eval.feedback import Groundedness
        from trulens_eval.feedback.provider.openai import OpenAI
        grounded = feedback.Groundedness(groundedness_provider=OpenAI())


        f_groundedness = feedback.Feedback(grounded.groundedness_measure_with_cot_reasons).on(
            Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content # See note below
        ).on_output().aggregate(grounded.grounded_statements_aggregator)
        ```
        The `on(...)` selector can be changed. See [Feedback Function Guide : Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)


        Args:
            source (str): The source that should support the statement
            statement (str): The statement to check groundedness

        Returns:
            float: A measure between 0 and 1, where 1 means each sentence is grounded in the source.
        """
        groundedness_scores = {}

        if isinstance(self.groundedness_provider, (AzureOpenAI, OpenAI, LiteLLM, Bedrock, Langchain)):
            plausible_junk_char_min = 4  # very likely "sentences" under 4 characters are punctuation, spaces, etc

            reason = ""
            if len(statement) > plausible_junk_char_min:
                reason = self._groundedness_doc_in_out(
                    source, statement
                )
            i = 0
            for line in reason.split('\n'):
                if "Score" in line:
                    groundedness_scores[f"statement_{i}"] = re_0_10_rating(line) / 10
                    i += 1
            return groundedness_scores, {"reason": reason}
        elif isinstance(self.groundedness_provider, Huggingface):
            raise Exception(
                "Chain of Thought reasoning is only applicable to OpenAI groundedness providers. Instantiate `Groundedness(groundedness_provider=OpenAI())` or use `groundedness_measure` feedback function."
            )

    def groundedness_measure_with_summarize_step(
        self, source: str, statement: str
    ) -> float:
        """A measure to track if the source material supports each sentence in the statement. 
        This groundedness measure is more accurate; but slower using a two step process.
        - First find supporting evidence with an LLM
        - Then for each statement sentence, check groundendness
        
        Usage on RAG Contexts:
        ```
        from trulens_eval import Feedback
        from trulens_eval.feedback import Groundedness
        from trulens_eval.feedback.provider.openai import OpenAI
        grounded = feedback.Groundedness(groundedness_provider=OpenAI())


        f_groundedness = feedback.Feedback(grounded.groundedness_measure_with_summarize_step).on(
            Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content # See note below
        ).on_output().aggregate(grounded.grounded_statements_aggregator)
        ```
        The `on(...)` selector can be changed. See [Feedback Function Guide : Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)


        Args:
            source (str): The source that should support the statement
            statement (str): The statement to check groundedness

        Returns:
            float: A measure between 0 and 1, where 1 means each sentence is grounded in the source.
        """
        groundedness_scores = {}
        reason = ""
        for i, hypothesis in enumerate(
                tqdm(statement.split("."),
                     desc="Groundendess per statement in source")):
            plausible_junk_char_min = 4  # very likely "sentences" under 4 characters are punctuation, spaces, etc
            if len(hypothesis) > plausible_junk_char_min:
                supporting_premise = self.groundedness_provider._find_relevant_string(
                    source, hypothesis
                )
                score = self.groundedness_provider._summarized_groundedness(
                    premise=supporting_premise, hypothesis=hypothesis
                )
                reason = reason + str.format(
                    prompts.GROUNDEDNESS_REASON_TEMPLATE,
                    statement_sentence=hypothesis,
                    supporting_evidence=supporting_premise,
                    score=score * 10,
                )
                groundedness_scores[f"statement_{i}"] = score
        return groundedness_scores, {"reason": reason}

    def grounded_statements_aggregator(
        self, source_statements_multi_output: List[Dict]
    ) -> float:
        """Aggregates multi-input, mulit-output information from the groundedness_measure methods.


        Args:
            source_statements_multi_output (List[Dict]): A list of scores. Each list index is a context. The Dict is a per statement score.

        Returns:
            float: for each statement, gets the max groundedness, then averages over that.
        """
        all_results = []

        statements_to_scores = {}

        # Ensure source_statements_multi_output is a list
        if not isinstance(source_statements_multi_output, list):
            source_statements_multi_output = [source_statements_multi_output]

        for multi_output in source_statements_multi_output:
            for k in multi_output:
                if k not in statements_to_scores:
                    statements_to_scores[k] = []
                statements_to_scores[k].append(multi_output[k])

        for k in statements_to_scores:
            all_results.append(np.max(statements_to_scores[k]))

        return np.mean(all_results)


    def _groundedness_doc_in_out(self, premise: str, hypothesis: str) -> str:
        """
        An LLM prompt using the entire document for premise and entire statement
        document for hypothesis.

        Args:
            premise (str): A source document
            hypothesis (str): A statement to check

        Returns:
            str: An LLM response using a scorecard template
        """

        return self.groundedness_provider.endpoint.run_in_pace(
            lambda: self.groundedness_provider._create_chat_completion(
                prompt=str.format(LLM_GROUNDEDNESS_FULL_SYSTEM, ) + str.
                    format(
                    prompts.LLM_GROUNDEDNESS_FULL_PROMPT,
                    premise=premise,
                    hypothesis=hypothesis
                )
            )
        )
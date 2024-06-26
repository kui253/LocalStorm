import logging
import re
from typing import Union, List
import os
import dspy
import requests
from bs4 import BeautifulSoup


def get_outline_from_dir(url):
    """Get the main title and table of contents from an url of a Wikipedia page."""
    p_dir = os.path.dirname(url)
    p_dir = os.path.dirname(p_dir)
    file_name = url.split("/")[-1]
    with open(os.path.join(p_dir, "structure", file_name), "r") as f:
        toc = f.read()
    main_title = file_name.replace(".txt", "")
    return main_title, toc.strip()


class LocalFindRelatedTopic(dspy.Signature):
    """I'm writing a method Statements for a topic mentioned below. Please identify and recommend some method statements on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in Method Statements for similar topics.
    Please list the topics in separate lines."""

    topic = dspy.InputField(prefix="Topic of interest:", format=str)
    related_topics = dspy.OutputField(format=str)


class LocalGenPersona(dspy.Signature):
    """You need to select a group of Method Statements editors who will work together to create a comprehensive article on the topic. Each of them represents a different perspective, role, or affiliation related to this topic. You can use other Method Statements of related topics for inspiration. For each editor, add a description of what they will focus on.
    Give your answer in the following format: 1. short summary of editor 1: description\n2. short summary of editor 2: description\n...
    """

    topic = dspy.InputField(prefix="Topic of interest:", format=str)
    examples = dspy.InputField(
        prefix="Wiki page outlines of related topics for inspiration:\n", format=str
    )
    personas = dspy.OutputField(format=str)


class LocalCreateWriterWithPersona(dspy.Module):
    """Discover different perspectives of researching the topic by reading Wikipedia pages of related topics."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel], rm):
        super().__init__()
        self.find_related_topic = dspy.ChainOfThought(LocalFindRelatedTopic)
        self.gen_persona = dspy.ChainOfThought(LocalGenPersona)
        self.engine = engine
        self.retriever = rm

    def forward(self, topic: str, draft=None):
        with dspy.settings.context(lm=self.engine):
            # Get section names from wiki pages of relevant topics for inspiration.

            related_topics = self.find_related_topic(topic=topic).related_topics
            dirs = set()

            for it in self.retriever(related_topics.split("\n")):
                dirs.add(it["url"])

            examples = []
            for d in dirs:

                try:
                    title, toc = get_outline_from_dir(d)
                    examples.append(f"Title: {title}\nTable of Contents: {toc}")
                except Exception as e:
                    logging.error(f"Error occurs when processing {d}: {e}")
                    continue
            if len(examples) == 0:
                examples.append("N/A")
            gen_persona_output = self.gen_persona(
                topic=topic, examples="\n----------\n".join(examples)
            ).personas

        personas = []
        for s in gen_persona_output.split("\n"):
            match = re.search(r"\d+\.\s*(.*)", s)
            if match:
                personas.append(match.group(1))

        sorted_personas = personas

        return dspy.Prediction(
            personas=personas,
            raw_personas_output=sorted_personas,
            related_topics=related_topics,
        )


class LocalStormPersonaGenerator:
    """
    A generator class for creating personas based on a given topic.

    This class uses an underlying engine to generate personas tailored to the specified topic.
    The generator integrates with a `CreateWriterWithPersona` instance to create diverse personas,
    including a default 'Basic fact writer' persona.

    Attributes:
        create_writer_with_persona (CreateWriterWithPersona): An instance responsible for
            generating personas based on the provided engine and topic.

    Args:
        engine (Union[dspy.dsp.LM, dspy.dsp.HFModel]): The underlying engine used for generating
            personas. It must be an instance of either `dspy.dsp.LM` or `dspy.dsp.HFModel`.
    """

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel], rm):
        self.create_writer_with_persona = LocalCreateWriterWithPersona(
            engine=engine, rm=rm
        )

    def generate_persona(self, topic: str, max_num_persona: int = 3) -> List[str]:
        """
        Generates a list of personas based on the provided topic, up to a maximum number specified.

        This method first creates personas using the underlying `create_writer_with_persona` instance
        and then prepends a default 'Basic fact writer' persona to the list before returning it.
        The number of personas returned is limited to `max_num_persona`, excluding the default persona.

        Args:
            topic (str): The topic for which personas are to be generated.
            max_num_persona (int): The maximum number of personas to generate, excluding the
                default 'Basic fact writer' persona.

        Returns:
            List[str]: A list of persona descriptions, including the default 'Basic fact writer' persona
                and up to `max_num_persona` additional personas generated based on the topic.
        """
        personas = self.create_writer_with_persona(topic=topic)
        default_persona = "Basic fact writer: Basic fact writer focusing on broadly covering the basic facts about the topic."
        considered_personas = [default_persona] + personas.personas[:max_num_persona]
        return considered_personas

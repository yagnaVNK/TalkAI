import os
import json
import getpass
from typing import List, Annotated
from typing_extensions import TypedDict
import operator
from IPython.display import Image, display
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from StreamingTTS import StreamingTTS
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question: str  # User question
    generation: str  # LLM generation
    web_search: str  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # List of retrieved documents

class RAGSystem:
    def __init__(self):
        # Set environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        if not os.environ.get("TAVILY_API_KEY"):
            os.environ["TAVILY_API_KEY"] = getpass.getpass("TAVILY_API_KEY: ")

        # Initialize LLM
        self.local_llm = "mistral"
        self.llm = ChatOllama(model=self.local_llm, temperature=0)
        self.llm_json_mode = ChatOllama(model=self.local_llm, temperature=0, format="json")
        
        search = GoogleSearchAPIWrapper(k=3)

        
        # Initialize vector store
        self._initialize_vectorstore()
        
        # Initialize web search
        #self.web_search_tool = TavilySearchResults(k=3)
        self.web_search_tool = Tool(
            name="google_search",
            description="Search Google for recent results.",
            func=search.run,
        )
        # Initialize workflow
        self._setup_workflow()

    def _initialize_vectorstore(self):
        # URLs for document loading
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]

        # Load and process documents
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Create vector store
        self.vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
        )
        self.retriever = self.vectorstore.as_retriever(k=3)

    def _setup_workflow(self):
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("websearch", self._web_search)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("generate", self._generate)

        # Build graph
        workflow.set_conditional_entry_point(
            self._route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        workflow.add_conditional_edges(
            "generate",
            self._grade_generation,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
                "max retries": END,
            },
        )

        self.graph = workflow.compile()

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _retrieve(self, state):
        print("---RETRIEVE---")
        documents = self.retriever.invoke(state["question"])
        return {"documents": documents}

    def _generate(self, state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        loop_step = state.get("loop_step", 0)

        rag_prompt = """You are an assistant for question-answering tasks. 
        Here is the context to use to answer the question:
        {context} 
        Think carefully about the above context. 
        Now, review the user question:
        {question}
        Provide an answer to this questions using only the above context. 
        Use two sentences maximum and keep the answer concise and 
        also do not use unnecessary punctuation like () or [ ]. Do not Itemize anyting in the answer.
        Answer:"""

        docs_txt = self._format_docs(documents)
        rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
        generation = self.llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        return {"generation": generation, "loop_step": loop_step + 1}

    def _grade_documents(self, state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

        doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 
        This carefully and objectively assess whether the document contains at least some information that is relevant to the question.
        Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

        filtered_docs = []
        web_search = "No"
        for d in documents:
            doc_grader_prompt_formatted = doc_grader_prompt.format(
                document=d.page_content, question=question
            )
            result = self.llm_json_mode.invoke(
                [SystemMessage(content=doc_grader_instructions)]
                + [HumanMessage(content=doc_grader_prompt_formatted)]
            )
            grade = json.loads(result.content)["binary_score"]
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
        return {"documents": filtered_docs, "web_search": web_search}

    def _web_search(self, state):
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state.get("documents", [])

        web_results = self.web_search_tool.run(question)
        #web_results = "\n".join([d["content"] for d in docs]) # Use this for TAVILY
        web_results = Document(page_content=web_results) 
        documents.append(web_results)
        return {"documents": documents}

    def _route_question(self, state):
        print("---ROUTE QUESTION---")
        router_instructions = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.
        Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

        route_question = self.llm_json_mode.invoke(
            [SystemMessage(content=router_instructions)]
            + [HumanMessage(content=state["question"])]
        )
        source = json.loads(route_question.content)["datasource"]
        if source == "websearch":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        else:
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"

    def _decide_to_generate(self, state):
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]

        if web_search == "Yes":
            print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "websearch"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def _grade_generation(self, state):
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        max_retries = state.get("max_retries", 3)

        hallucination_grader_instructions = """You are a teacher grading a quiz. 
        You will be given FACTS and a STUDENT ANSWER. 
        Here is the grade criteria to follow:
        (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
        (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.
        Score:
        A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
        A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give."""

        hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 
        Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

        hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
            documents=self._format_docs(documents), generation=generation.content
        )
        result = self.llm_json_mode.invoke(
            [SystemMessage(content=hallucination_grader_instructions)]
            + [HumanMessage(content=hallucination_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]

        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            print("---GRADE GENERATION vs QUESTION---")
            
            answer_grader_instructions = """You are a teacher grading a quiz. 
            You will be given a QUESTION and a STUDENT ANSWER. 
            Here is the grade criteria to follow:
            (1) The STUDENT ANSWER helps to answer the QUESTION
            Score:
            A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
            The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.
            A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give."""

            answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 
            Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

            answer_grader_prompt_formatted = answer_grader_prompt.format(
                question=question, generation=generation.content
            )
            result = self.llm_json_mode.invoke(
                [SystemMessage(content=answer_grader_instructions)]
                + [HumanMessage(content=answer_grader_prompt_formatted)]
            )
            grade = json.loads(result.content)["binary_score"]
            
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            elif state["loop_step"] <= max_retries:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
            else:
                print("---DECISION: MAX RETRIES REACHED---")
                return "max retries"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"

    def run(self):
        print("\nWelcome to the RAG System!")
        print("Type 'exit' to quit the system.\n")
        
        while True:
            question = input("\nEnter your question: ")
            
            if question.lower() == 'exit':
                print("\nThank you for using the RAG System. Goodbye!")
                break
            
            inputs = {"question": question, "max_retries": 3}
            
            print("\nProcessing your question...\n")
            
            for event in self.graph.stream(inputs, stream_mode="values"):
                if "generation" in event:
                    print("\nAnswer:", event["generation"].content)

            try:
                streaming_tts.generate_and_stream(
                    text=event["generation"].content,
                    speaker_wav="myvoices/Sample1.wav",  # Optional for voice cloning
                    language="en"
                )
            except KeyboardInterrupt:
                print("\nStopping TTS stream...")
                streaming_tts.stop_stream()

if __name__ == "__main__":
    rag_system = RAGSystem()
    streaming_tts = StreamingTTS()
    
    final_answer = rag_system.run()
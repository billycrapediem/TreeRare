from dependency import DependencyTree, TreeNode
from consituency_tree import ConstituencyTreeNode, ConstituencyTree
import stanza
from utils import chat_with_gpt4, rerank_document
from BM25 import BM_find_document
#from dpr import dpr_find_document
from typing import List, Set, Tuple
from collections import Counter

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse,constituency')
IGNORE_DEPS = {"case", "det", "aux", "cop", "mark", "cc", "punct", "expl", "clf", "aux:pass"}
SKIP_WORD = {"who","when","where","what","which","When","Where","Who","What","Which"}
class Hotpot_traverse():
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        
        
    def create_answer_prompt(self, questions: str, context: str) -> str:
        """Create the appropriate prompt for disambiguation."""
        return f"""
Answer the {questions} based on on the document info.(If document does not contain relevant information, do the research on your own). For each question find as many answers as possible. Response all the answers in a short paragraph (as specific as possible). 
# Document Info: {context}
    """
    
    def create_pure_prompt(self, phrase:str, context:str, questions:str) -> str:
        return f"""
You're a an expert analyzing "{phrase}" in main query: {self.question}.
1.You should generate at most 3 simple questions that mainly ask for information about {phrase} that helps understand the main query. Question should be single-hop, clear and search-friendly.
2.Here is question ask by other experts. 
#Questions:{questions}
3. Here is what we currently know
#Documents:{context}
pick top 5 questions that are best in decomposing the main query and leading to final answer of the main query. 
strictly FOLLOW the format: #response: question1; question2;....
        """
    def disambiguate_node(self,node) -> Tuple[str, Set[str]]:
        
        """Disambiguate a single node in the tree."""
        child_infos = {}
        documents = []
        for child in node.children:
            if( len(child.substance.split()) >= 3) and (child.deprel not in IGNORE_DEPS) and (child.substance not in SKIP_WORD):
                disambiguated_question, docs = self.disambiguate_node(child)
                child_infos[child.substance] = disambiguated_question
                documents += docs
        pure_context = str(documents)
        pure_prompt = self.create_pure_prompt(phrase=node.substance, context = pure_context, questions = str(child_infos))
        result_question = []
        response,inp,out = chat_with_gpt4(pure_prompt)
        self.input_tokens += inp
        self.output_tokens += out
        print(f"phrase: {node.substance}\n question:{response}")
        try:
            result_question += response.split(';')
            tmp_doc = []
            for r in result_question:
                tmp_doc += BM_find_document(r,10)
            response,inp,out = chat_with_gpt4(self.create_answer_prompt(str(result_question), str(tmp_doc)))
            print(f"answer: {response}")  
            self.input_tokens += inp
            self.output_tokens += out 
            documents.append(response)
            return result_question, documents
            
        except Exception as e:
            print(e)
            documents += BM_find_document(response)
            node.disambiguation = response
            return [response],  documents
    
    def disambiguate_text(self,text: str) -> Tuple[List[str], Set[str]]:
        """Main function to disambiguate text using dependency tree."""
        # Create and analyze dependency tree
        doc = nlp(text)
        self.question = text
        tree = ConstituencyTree()
        tree.build_from_sentence(doc.sentences[0])
        #tree.print_tree()
        #questions, documents = self.disambiguate_node(tree.root.children[0])
        questions, documents = self.disambiguate_node(tree.root)
        return questions, documents

class ASQA_traverse(): 
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
    def create_pure_prompt(self, phrase:str,context:str, questions:str) -> str:
        return f"""
You're a disambiguation expert analyzing "{phrase}" in: 
{self.question}  
Instruction:
1. Analyze the question by considering these potential ambiguities:
   - Temporal: Check for unclear time references, periods, or temporal scope
   - Entity: Identify names, references, or terms that could refer to multiple entities
   - Semantic: Look for words with multiple meanings (polysemy/homonymy)
   - Scope: Consider possible boundaries and levels of detail
   - Intent: Examine possible purposes and expected answer types
   - Cultural: Consider cultural-dependent interpretations
   - Quantitative: Check for unclear measurements or numerical references
   - Linguistic: Analyze syntax and referential clarity
   - Categorical: Consider possible classification schemes
   - Contextual: Examine required background knowledge and relationships
2. Analyze the question word by word. Return disambiguated question and its interperatation for each different meaning strictly

Question Ask by others:
{questions}

Here is what we currently know
Documents:{context}

pick top 5 questions that are best in disambiguating the question. (covers different meanings of the questions) and strictly FOLLOW the format: #response: question1; question2;....
        """  
    def create_disambiguation_prompt(self ,questions: str, context: str) -> str:
        """Create the appropriate prompt for disambiguation."""
        return f"""
Answer the {questions} based on on the document info.(If document does not contain relevant information, do the research on your own). For each question find as many answers as possible. Response all the answers in a short paragraph (as specific as possible). 
# Document Info: {context}
            """
    
    def disambiguate_node(self,node) -> Tuple[str, Set[str]]:
        """Disambiguate a single node in the tree."""
        child_infos = {}
        documents = []
        #process the children
        for child in node.children:
            if len(child.substance.split()) >= 3 and child.deprel not in IGNORE_DEPS and child.substance not in SKIP_WORD:
                disambiguated_question, docs = self.disambiguate_node(child)
                child_infos[child.substance] = disambiguated_question
                documents += docs
        context_info = f"\n{str(documents)}"
        
        prompt = self.create_pure_prompt(node.substance, context_info, str(child_infos))
        try:
            response,i,o = chat_with_gpt4(prompt)
            response = response.split("response:")[1].strip()
            self.input_tokens += i
            self.output_tokens += o
            questions = response.split(';')
            tmp_doc = []
            for r in questions:
                tmp_doc += BM_find_document(r,10)
            re_prompt = self.create_disambiguation_prompt(str(questions), str(tmp_doc))
            d, i, o = chat_with_gpt4(re_prompt)
            self.input_tokens += i
            self.output_tokens += o
            documents.append(d)
            return [], documents 
        except Exception as e:
            print(e)
            documents += BM_find_document(node.substance)
            node.disambiguation = response
            return [context_info], documents
    
    def disambiguate_text(self,text: str) -> Tuple[List[str], Set[str]]:
        """Main function to disambiguate text using dependency tree."""
        # Create and analyze dependency tree
        doc = nlp(text)
        self.question = text
        tree = ConstituencyTree()
        tree.build_from_sentence(doc.sentences[0])
        #tree.print_tree()
        #questions, documents = self.disambiguate_node(tree.root.children[0])
        questions, documents = self.disambiguate_node(tree.root)
        return questions, documents
    
class AmbigDoc_traverse():
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
    def create_pure_prompt(self,context:str) -> str:
        return f"""
        answer {self.question}. Look for named entities in the question. Identify and explain the entity referred to by the name in each context. Search sentences within the context as answer for ALL different entities you recognized. For an unique entity, there can be multiple answers, find ALL of them (DO NOT SUMMARIZE information). 
        #Context:{context}
        """  
    def create_disambiguation_prompt(self ,phrase: str, context: str,questions:str) -> str:
        """Create the appropriate prompt for disambiguation."""
        return f"""
You're a disambiguation expert analyzing
{self.question}  
Instruction:
1. Analyze the question by considering these potential entity ambiguity. Carefully look at the context information.   
2. Based on possible interperation and your analysis, try to disambiguate the entity in the question.
3.pick top 7 questions that are best disambiguated question from your questions and question from others. Good list of question should cover different topics but related to the main question
strictly FOLLOW the format: #response: question1; question2;....
# Context Info: {context}
Here is question ask by other experts. 
#Questions:{questions}
            """
    def disambiguate_node(self,node) -> Tuple[str, Set[str]]:
        """Disambiguate a single node in the tree."""
        child_infos = {}
        documents = []
        for child in node.children:
            if child.deprel not in IGNORE_DEPS and child.substance not in SKIP_WORD and len(child.substance.split()) >= 2:
                disambiguated_question, docs = self.disambiguate_node(child)
                child_infos[child.substance] = disambiguated_question
                documents += docs
        try:
            context_info = f""
            prompt = self.create_disambiguation_prompt(node.substance, context_info,str(child_infos))
            pure,i, o = chat_with_gpt4(prompt)
            self.input_tokens += i
            self.output_tokens += o
            response = pure.split("response:")[1].strip()
            #print(f"prompt: {prompt}\nresponse:{pure}")
            questions = response.split(';')
            tmp_doc = []
            for r in questions:
                tmp_doc += BM_find_document(r,5)
            context_info += f"\n#sub_nodes info: {str(child_infos)}\n documents{str(tmp_doc)}"
            d, i , o = chat_with_gpt4(self.create_pure_prompt( context_info))
            self.input_tokens += i
            self.output_tokens += o
            documents.append(d)
            return questions, documents
        except Exception as e:
            print(e)
            documents += BM_find_document(node.substance)
            node.disambiguation = pure
            return [pure], documents
    
    def disambiguate_text(self,text: str) -> Tuple[List[str], Set[str]]:
        doc = nlp(text)
        self.question = text
        tree = ConstituencyTree()
        tree.build_from_sentence(doc.sentences[0])
        #tree.print_tree()
        #questions, documents = self.disambiguate_node(tree.root.children[0])
        questions, documents = self.disambiguate_node(tree.root)
        #top_elements = rerank_document(documents, text, 30)
        return questions, documents
    
class retrieve_traverse():
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
    def disambiguate_node(self,node) -> Tuple[str, Set[str]]:
        documents = []
        # Process children first
        for child in node.children:
            if( len(child.substance.split())>=3) and (child.deprel not in IGNORE_DEPS) and (child.substance not in SKIP_WORD):
                docs = self.disambiguate_node(child)
                documents += docs
        documents += BM_find_document(node.substance,30)
        documents = rerank_document(documents, self.question, 15)
        return documents
        
    
    def disambiguate_text(self,text: str) -> Tuple[List[str], Set[str]]:
        """Main function to disambiguate text using dependency tree."""
        # Create and analyze dependency tree
        doc = nlp(text)
        self.question = text
        #tree = DependencyTree()
        #tree.build_from_sentence(doc.sentences[0])
        #documents = self.disambiguate_node(tree.root.children[0])
        tree = ConstituencyTree()
        tree.build_from_sentence(doc.sentences[0])
        documents = self.disambiguate_node(tree.root.children[0])
        top_elements = rerank_document(documents,text, 15)
        return top_elements
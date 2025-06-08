from typing import List, Optional, Dict
from dataclasses import dataclass

# Initialize global variables
IGNORE_DEPS = {"case", "det", "aux", "cop", "mark", "cc", "punct", "expl", "clf", "aux:pass"}

@dataclass
class TreeNode:
    """Represents a node in the dependency tree."""
    word_id: int
    text: str
    deprel: str
    pos: str
    children: List['TreeNode']
    parent: Optional['TreeNode']
    substance: str
    start_idx: int
    end_idx: int
    disambiguation: str = ""

    def __init__(self, word_id: int, text: str, deprel: str, pos: str):
        self.word_id = word_id
        self.text = text
        self.deprel = deprel
        self.pos = pos
        self.children = []
        self.parent = None
        self.substance = ""
        self.start_idx = -1
        self.end_idx = -1
        
    def add_child(self, child: 'TreeNode') -> None:
        """Add a child node to the current node."""
        child.parent = self
        self.children.append(child)
    
    def __str__(self) -> str:
        return f"{self.text} --{self.deprel}--> [{self.substance}]" if self.substance else f"{self.text} --{self.deprel}-->"

class DependencyTree:
    """Represents a dependency parse tree for a sentence."""
    def __init__(self):
        self.root = TreeNode(0, "ROOT", "root", 'null')
        self.nodes: Dict[int, TreeNode] = {0: self.root}
        self.original_sentence: str = ""
    
    def build_from_sentence(self, sentence) -> None:
        """Constructs the dependency tree from a sentence."""
        self._initialize_sentence(sentence)
        self._create_nodes(sentence)
        self._combine_flat_relationships(sentence)
        self._establish_relationships(sentence)
        self._build_substances()
    
    def _initialize_sentence(self, sentence) -> None:
        """Initialize the original sentence and calculate word indices."""
        self.original_sentence = " ".join(word.text for word in sentence.words)
        
    def _create_nodes(self, sentence) -> None:
        """Create initial nodes for each word in the sentence."""
        current_idx = 0
        for word in sentence.words:
            word_start = self.original_sentence.find(word.text, current_idx)
            word_end = word_start + len(word.text)
            
            node = TreeNode(word.id, word.text, word.deprel, word.pos)
            node.start_idx, node.end_idx = word_start, word_end
            self.nodes[word.id] = node
            
            current_idx = word_end
    
    def _combine_flat_relationships(self, sentence) -> None:
        """Combine nodes with flat relationships."""
        nodes_to_remove = set()
        for word in sentence.words:
            if word.deprel == "flat":
                parent_node = self.nodes[word.head]
                parent_node.end_idx = self.nodes[word.id].end_idx
                parent_node.text = self.original_sentence[parent_node.start_idx:parent_node.end_idx]
                nodes_to_remove.add(word.id)
        
        for node_id in nodes_to_remove:
            del self.nodes[node_id]
    
    def _establish_relationships(self, sentence) -> None:
        """Establish parent-child relationships between nodes."""
        for word in sentence.words:
            if word.id in self.nodes:
                child_node = self.nodes[word.id]
                parent_id = self._find_parent_id(word.head, sentence)
                self.nodes[parent_id].add_child(child_node)
    
    def _find_parent_id(self, current_id: int, sentence) -> int:
        """Find the actual parent ID accounting for flat relationships."""
        while current_id not in self.nodes and current_id != 0:
            for word in sentence.words:
                if word.id == current_id:
                    current_id = word.head
                    break
        return current_id
    
    def _build_substances(self) -> None:
        """Build substances for all nodes."""
        for node_id, node in self.nodes.items():
            if node_id != 0:
                self._build_node_substance(node)
    
    def _build_node_substance(self, node: TreeNode) -> None:
        """Build substance for a single node."""
        start_idx, end_idx = node.start_idx, node.end_idx
        
        def update_bounds(n: TreeNode) -> None:
            nonlocal start_idx, end_idx
            start_idx = min(start_idx, n.start_idx)
            end_idx = max(end_idx, n.end_idx)
            for child in n.children:
                update_bounds(child)
        
        for child in node.children:
            if child.deprel not in IGNORE_DEPS: 
                update_bounds(child)
        
        node.substance = self.original_sentence[start_idx:end_idx].strip()

    def print_tree(self, node: Optional[TreeNode] = None, level: int = 0, prefix: str = "") -> None:
        """Print the tree structure."""
        if node is None:
            node = self.root
            print(f"{node.text} [{node.substance}]")
            
        child_count = len(node.children)
        for i, child in enumerate(node.children):
            is_last = i == child_count - 1
            branch = "└──" if is_last and level > 0 else "├──" if level > 0 else "│  "
            
            print(f"{prefix}{branch} {child}")
            
            new_prefix = prefix + ("   " if is_last else "│  ") if level > 0 else prefix
            self.print_tree(child, level + 1, new_prefix + "│  ")
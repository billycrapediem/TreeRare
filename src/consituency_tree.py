from typing import List, Optional, Dict
from dataclasses import dataclass
import stanza

# Initialize Stanza pipeline for constituency parsing

@dataclass
class ConstituencyTreeNode:
    """Represents a node in the constituency tree."""
    label: str
    children: List['ConstituencyTreeNode']
    start_idx: int
    end_idx: int
    substance: str  # Stores the concatenated string of all children
    text: Optional[str] = None  # Only for leaf nodes

    def __init__(self, label: str, start_idx: int = -1, end_idx: int = -1, text: Optional[str] = None):
        self.label = label
        self.children = []
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.text = text
        self.substance = ""  # Initialize substance as an empty string
        self.deprel = "new"

    def add_child(self, child: 'ConstituencyTreeNode') -> None:
        """Add a child node to the current node."""
        self.children.append(child)

    def __str__(self) -> str:
        if self.text:
            return f"{self.label}: {self.text}"
        return f"{self.label} [{self.substance}]"

class ConstituencyTree:
    """Represents a constituency parse tree for a sentence."""
    def __init__(self):
        self.root: Optional[ConstituencyTreeNode] = None
        self.original_sentence: str = ""

    def build_from_sentence(self, sentence) -> None:
        """Constructs the constituency tree from a sentence."""
        self._initialize_sentence(sentence)
        self._build_tree(sentence)
        self._build_substances()  # Build substances after the tree is constructed

    def _initialize_sentence(self, sentence) -> None:
        """Initialize the original sentence."""
        self.original_sentence = " ".join(word.text for word in sentence.words)

    def _build_tree(self, sentence) -> None:
        """Build the constituency tree from the Stanza parse tree."""
        if sentence.constituency:
            print("here")
            self.root = self._build_tree_recursive(sentence.constituency)

    def _build_tree_recursive(self, node) -> ConstituencyTreeNode:
        """Recursively build the constituency tree from Stanza's parse tree."""
        # For leaf nodes (words), extract the text and indices
        if not node.children:
            start_idx = self.original_sentence.find(node.label)
            end_idx = start_idx + len(node.label)
            return ConstituencyTreeNode(label=node.label, start_idx=start_idx, end_idx=end_idx, text=node.label)

        # For non-leaf nodes, create a new tree node and recursively add children
        tree_node = ConstituencyTreeNode(label=node.label)
        for child in node.children:
            tree_node.add_child(self._build_tree_recursive(child))

        # Calculate the start and end indices for the current node based on its children
        if tree_node.children:
            tree_node.start_idx = tree_node.children[0].start_idx
            tree_node.end_idx = tree_node.children[-1].end_idx

        return tree_node

    def _build_substances(self) -> None:
        """Build substances for all nodes."""
        if self.root:
            self._build_node_substance(self.root)

    def _build_node_substance(self, node: ConstituencyTreeNode) -> None:
        """Build substance for a single node."""
        if node.children:
            # For non-leaf nodes, concatenate the substances of all children
            for child in node.children:
                self._build_node_substance(child)
            node.substance = " ".join(child.substance if child.substance else child.text for child in node.children)
            # Recursively build substances for children
            
        else:
            # For leaf nodes, the substance is the text itself
            node.substance = node.text

    def print_tree(self, node: Optional[ConstituencyTreeNode] = None, level: int = 0, prefix: str = "") -> None:
        """Print the tree structure."""
        if node is None:
            node = self.root
            if node is None:
                print("Tree is empty.")
                return
            print(f"{node.label} [{node.substance}]")

        child_count = len(node.children)
        for i, child in enumerate(node.children):
            is_last = i == child_count - 1
            branch = "└──" if is_last and level > 0 else "├──" if level > 0 else "│  "

            print(f"{prefix}{branch} {child}")

            new_prefix = prefix + ("   " if is_last else "│  ") if level > 0 else prefix
            self.print_tree(child, level + 1, new_prefix + "│  ")

"""# Example usage
text = "Who sang the theme music for the television show \"Dear John\"?"
doc = nlp(text)
tree = ConstituencyTree()
tree.build_from_sentence(doc.sentences[0])
tree.print_tree()"""
import csv
from typing import List, Dict, Optional, Set


class TreeNode:
    """A node in the pathway tree that can have multiple children."""

    def __init__(self, name: str, parent: Optional['TreeNode'] = None):
        self.name = name
        self.parent = parent
        self.children: Dict[str, 'TreeNode'] = {}
        self.genes: Set[str] = set()

    def add_child(self, child_name: str) -> 'TreeNode':
        """Add a child node and return it. If it already exists, return the existing one."""
        if child_name not in self.children:
            self.children[child_name] = TreeNode(child_name, parent=self)
        return self.children[child_name]

    def add_genes(self, genes: List[str]):
        """Add genes to this node."""
        self.genes.update(genes)

    def get_path(self) -> List[str]:
        """Get the full path from root to this node."""
        path = []
        current = self
        while current is not None:
            path.append(current.name)
            current = current.parent
        return list(reversed(path))

    def find_node(self, path: List[str]) -> Optional['TreeNode']:
        """Find a node by following the given path."""
        current = self
        for part in path:
            if part in current.children:
                current = current.children[part]
            else:
                return None
        return current

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children) == 0

    def get_all_descendants(self) -> List['TreeNode']:
        """Get all descendant nodes (children, grandchildren, etc.)."""
        descendants = []
        for child in self.children.values():
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants

    def __str__(self) -> str:
        return f"TreeNode(name='{self.name}', children={len(self.children)}, genes={len(self.genes)})"

    def __repr__(self) -> str:
        return self.__str__()


class PathwayTree:
    """A tree structure for representing pathway hierarchies."""

    def __init__(self):
        self.root = TreeNode("ROOT")

    def add_pathway(self, pathway: str, genes: List[str] = None):
        """
        Add a pathway to the tree. Pathways are separated by ' > '.

        Args:
            pathway: The pathway string (e.g., "A > B > C")
            genes: Optional list of genes associated with this pathway
        """
        if genes is None:
            genes = []

        # Split pathway and clean whitespace
        parts = [part.strip() for part in pathway.split(' > ')]

        # Navigate/create the path in the tree
        current = self.root
        for part in parts:
            current = current.add_child(part)

        # Add genes to the final node
        if genes:
            current.add_genes(genes)

    def find_pathway(self, pathway: str) -> Optional[TreeNode]:
        """Find a node by pathway string."""
        parts = [part.strip() for part in pathway.split(' > ')]
        return self.root.find_node(parts)

    def get_all_pathways(self) -> List[str]:
        """Get all pathway strings in the tree."""
        pathways = []

        def collect_paths(node: TreeNode):
            if node != self.root:  # Skip the root node
                path_parts = node.get_path()[1:]  # Remove ROOT from path
                if path_parts:
                    pathways.append(' > '.join(path_parts))

            for child in node.children.values():
                collect_paths(child)

        collect_paths(self.root)
        return sorted(pathways)

    def get_leaf_pathways(self) -> List[str]:
        """Get only the leaf pathway strings (endpoints)."""
        leaf_pathways = []

        def collect_leaf_paths(node: TreeNode):
            if node != self.root and node.is_leaf():
                path_parts = node.get_path()[1:]  # Remove ROOT from path
                if path_parts:
                    leaf_pathways.append(' > '.join(path_parts))

            for child in node.children.values():
                collect_leaf_paths(child)

        collect_leaf_paths(self.root)
        return sorted(leaf_pathways)

    def display(self, max_depth: int = None):
        """Display the tree structure with tab-based indentation."""
        def display_node(node: TreeNode, depth: int = 0):
            if node != self.root:  # Don't display the ROOT node
                indent = '\t' * depth
                gene_count = f" ({len(node.genes)} genes)" if node.genes else ""
                # print(f"{indent}{node.name}{gene_count}")
                print(f"{indent}{node.name}")

            # Only recurse if we haven't hit max_depth
            if max_depth is None or depth < max_depth:
                for child in node.children.values():
                    display_node(child, depth + 1 if node != self.root else depth)

        display_node(self.root)

    def print_tree(self, indent: int = 0, node: TreeNode = None, max_depth: int = None):
        """Print the tree structure with indentation."""
        if node is None:
            node = self.root

        if node != self.root:  # Don't print the ROOT node
            gene_count = f" ({len(node.genes)} genes)" if node.genes else ""
            print(' ' * indent + f"├── {node.name}{gene_count}")
            indent += 4

        # Only recurse if we haven't hit max_depth
        if max_depth is None or indent < max_depth * 4:
            for child in node.children.values():
                self.print_tree(indent, child, max_depth)

    def print_base_nodes(self):
        """Print only the direct children of root (base nodes)."""
        print("Base nodes:")
        for child_name, child_node in self.root.children.items():
            gene_count = f" ({len(child_node.genes)} genes)" if child_node.genes else ""
            child_count = f" [{len(child_node.children)} children]" if child_node.children else ""
            print(f"  ├── {child_name}{gene_count}{child_count}")

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the tree."""
        all_nodes = [self.root] + self.root.get_all_descendants()
        leaf_nodes = [node for node in all_nodes if node.is_leaf()]
        nodes_with_genes = [node for node in all_nodes if node.genes]

        return {
            'total_nodes': len(all_nodes) - 1,  # Exclude root
            'leaf_nodes': len(leaf_nodes),
            'nodes_with_genes': len(nodes_with_genes),
            'total_genes': sum(len(node.genes) for node in all_nodes),
            'max_depth': self._get_max_depth()
        }

    def _get_max_depth(self) -> int:
        """Calculate the maximum depth of the tree."""
        def get_depth(node: TreeNode) -> int:
            if not node.children:
                return 0
            return 1 + max(get_depth(child) for child in node.children.values())

        return get_depth(self.root)


def build_tree_from_csv(csv_path: str) -> PathwayTree:
    """
    Build a pathway tree from a CSV file.

    Expected CSV format:
    - Column 1: ID (ignored)
    - Column 2: Pathway name (ignored - we use column 3 for hierarchy)
    - Column 3: Full pathway hierarchy (separated by ' > ')
    - Column 4: Genes (comma-separated, optional)

    Args:
        csv_path: Path to the CSV file

    Returns:
        PathwayTree object
    """
    tree = PathwayTree()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)

        for row_num, row in enumerate(reader, 1):
            if not row or len(row) < 3:
                print(f"Warning: Skipping empty or incomplete row {row_num}")
                continue

            try:
                # Use the third column which contains the full hierarchy
                pathway_hierarchy = row[2]

                # Skip header-like rows or empty hierarchies
                if not pathway_hierarchy or pathway_hierarchy in ['MitoPathways Hierarchy', '']:
                    continue

                # Parse genes if they exist (fourth column)
                genes = []
                if len(row) >= 4 and row[3]:
                    genes = [gene.strip() for gene in row[3].split(',')]

                tree.add_pathway(pathway_hierarchy, genes)

            except Exception as e:
                print(f"Error processing row {row_num}: {e}")
                continue

    return tree


def main():
    """Main function to demonstrate the tree functionality."""
    # Build tree from CSV
    csv_path = 'data/human_mitocarta_pathways.csv'

    print("Building pathway tree from CSV...")
    tree = build_tree_from_csv(csv_path)

    print("\nTree Statistics:")
    stats = tree.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nBase Nodes Only:")
    tree.print_base_nodes()

    print(f"\nTotal base nodes: {len(tree.root.children)}")

    tree.display()


if __name__ == '__main__':
    main()

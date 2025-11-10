#!/usr/bin/env python3
"""
Assign yeast proteins to pathways based on their GO annotations.

This script reads yeast GO annotations from a pickle file and assigns proteins
to pathways based on a GO-to-pathway mapping CSV file. It produces output
similar to the human_mitocarta_pathways.csv format but for yeast genes.

Usage Examples:
    # Basic usage with sample data
    python assign_proteins_to_pathway.py

    # Use full dataset with reliable evidence filter
    python assign_proteins_to_pathway.py \
        --go_annotations data/yeast_all_go_annotations.pkl \
        --pathway_assignments data/go_pathway_assignments.csv \
        --uniprot_mapping data/yeast_mitocarta.csv \
        --evidence_filter reliable \
        --output yeast_pathways_reliable.csv

    # Use only physical evidence with debugging
    python assign_proteins_to_pathway.py \
        --go_annotations data/yeast_all_go_annotations.pkl \
        --pathway_assignments data/go_pathway_assignments.csv \
        --uniprot_mapping data/yeast_mitocarta.csv \
        --evidence_filter physical \
        --debug --stats \
        --output yeast_pathways_physical.csv

Input Files:
    - GO annotations pickle file: Contains GoAnnotation objects for each protein
    - Pathway assignments CSV/JSON: Maps GO IDs to pathway hierarchies
      CSV format requires columns: GO_id, GO_description, Pathway, Description
      JSON format: {"pathway_hierarchy": ["GO:ID (description)", ...]}
    - UniProt mapping CSV: Maps UniProt IDs to gene names
      Required columns: UniprotID, Gene
    - Pathway tree TXT: Hierarchical pathway structure for sorting

Output Format:
    CSV file similar to human_mitocarta_pathways.csv with columns:
    - Pathway_ID: Numeric identifier
    - Pathway: Most specific pathway name (last part of hierarchy)
    - Pathway_Hierarchy: Full pathway hierarchy string
    - Uniprot_IDs: Comma-separated list of UniProt protein IDs
    - Genes: Comma-separated list of gene names (mapped from UniProt IDs)
    - GO_Annotations: Semicolon-separated GO terms used for assignment
    - Gene_Count: Number of unique proteins in the pathway

Evidence Filters:
    - 'physical': Only experimental evidence (IDA, IMP, IPI, HDA, HMP)
    - 'reliable': Physical + traceable evidence (adds TAS, NAS, ComplexPortal)
    - 'all': All evidence codes including computational predictions

Hierarchical Propagation:
    Proteins are automatically propagated to all parent pathway levels. If a protein
    appears in "OXPHOS > Complex II > CII subunits", it will also appear in
    "OXPHOS > Complex II" and "OXPHOS".

Pathway Depth Filtering:
    By default, proteins assigned to both broad (depth 1) and specific (depth >=2)
    pathways will only be retained in the more specific pathways. This reduces noise
    from overly broad categorizations while preserving detailed functional assignments.
    Use --no_depth_filter to disable this behavior.
"""

import pickle
import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import argparse

# Add the current directory to the path to import GO annotation classes
sys.path.append(str(Path(__file__).parent))

try:
    from fetch_go_annotations import GoAnnotation, GoAnnotationCollection
except ImportError:
    print("Error: Could not import GoAnnotation classes. Make sure fetch_go_annotations.py is in the current directory.")
    sys.exit(1)


# Define evidence codes considered as "physical" or experimental evidence
PHYSICAL_EVIDENCE_CODES = {
    'IDA',  # Inferred from Direct Assay
    'IMP',  # Inferred from Mutant Phenotype
    'IPI',  # Inferred from Physical Interaction
    'IEP',  # Inferred from Expression Pattern
    'IGI',  # Inferred from Genetic Interaction
    'HDA',  # High throughput Direct Assay (SGD specific)
    'HMP',  # High throughput Mutant Phenotype (SGD specific)
}

# Additional evidence codes that could be considered reliable
RELIABLE_EVIDENCE_CODES = PHYSICAL_EVIDENCE_CODES.union({
    'TAS',  # Traceable Author Statement
    'NAS',  # Non-traceable Author Statement
    'IDA:ComplexPortal',  # ComplexPortal direct assay
    'IDA:SGD',  # SGD direct assay
    'IMP:SGD',  # SGD mutant phenotype
    'HDA:SGD',  # SGD high throughput direct assay
    'HMP:SGD',  # SGD high throughput mutant phenotype
    'TAS:Reactome',  # Reactome traceable author statement
    'NAS:ComplexPortal',  # ComplexPortal non-traceable author statement
})


def load_go_annotations(pickle_path: str) -> Dict[str, GoAnnotation]:
    """Load GO annotations from pickle file."""
    print(f"Loading GO annotations from {pickle_path}...")
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data)} protein annotations")
        return data
    except FileNotFoundError:
        print(f"Error: File {pickle_path} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)


def load_pathway_tree_order(tree_path: str) -> Dict[str, int]:
    """Load pathway tree and create ordering dictionary."""
    try:
        with open(tree_path, 'r') as f:
            lines = f.readlines()

        pathway_order = {}
        order_index = 0

        for line in lines:
            # Strip whitespace and tabs
            pathway = line.strip()
            if pathway:  # Skip empty lines
                # Remove leading tabs to get clean pathway name
                clean_pathway = pathway.lstrip('\t')
                pathway_order[clean_pathway] = order_index
                order_index += 1

        print(f"Loaded pathway tree order for {len(pathway_order)} pathways")
        return pathway_order
    except FileNotFoundError:
        print(f"Warning: Pathway tree file {tree_path} not found. Using alphabetical order.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading pathway tree: {e}. Using alphabetical order.")
        return {}


def load_uniprot_gene_mapping(csv_path: str) -> Dict[str, str]:
    """Load UniProt ID to gene name mapping from CSV file."""
    print(f"Loading UniProt to gene name mapping from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        # Filter out rows with NaN gene names and create mapping
        df_clean = df.dropna(subset=['Gene'])
        mapping = dict(zip(df_clean['UniprotID'], df_clean['Gene']))
        nan_count = len(df) - len(df_clean)
        if nan_count > 0:
            print(f"Warning: {nan_count} UniProt IDs have missing gene names")
        print(f"Loaded {len(mapping)} UniProt ID to gene name mappings")
        return mapping
    except FileNotFoundError:
        print(f"Warning: File {csv_path} not found. Gene names will not be available.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading UniProt mapping file: {e}. Gene names will not be available.")
        return {}


def load_pathway_assignments(file_path: str) -> pd.DataFrame:
    """Load GO-to-pathway assignments from CSV or JSON file."""
    print(f"Loading pathway assignments from {file_path}...")
    try:
        if file_path.endswith('.json'):
            # Load JSON format: hierarchy -> list of GO terms
            import json
            with open(file_path, 'r') as f:
                hierarchy_to_go = json.load(f)

            # Convert to DataFrame format expected by the rest of the code
            rows = []
            for pathway_hierarchy, go_terms in hierarchy_to_go.items():
                for go_term in go_terms:
                    # Extract GO ID and description from format "GO:0005739 (mitochondrion)"
                    if '(' in go_term and ')' in go_term:
                        go_id = go_term.split('(')[0].strip()
                        go_description = go_term.split('(')[1].rstrip(')')
                    else:
                        go_id = go_term
                        go_description = go_term

                    rows.append({
                        'GO_id': go_id,
                        'GO_description': go_description,
                        'Pathway': pathway_hierarchy,
                        'Description': f"Pathway assignment from {pathway_hierarchy}"
                    })

            df = pd.DataFrame(rows)
            print(f"Loaded {len(hierarchy_to_go)} pathway hierarchies with {len(df)} GO-pathway mappings from JSON")
        else:
            # Load CSV format
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} pathway assignments from CSV")

        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)


def filter_annotations_by_evidence(annotations: List[Dict],
                                 evidence_filter: str = 'reliable') -> List[Dict]:
    """
    Filter GO annotations based on evidence codes.

    Args:
        annotations: List of GO annotation dictionaries
        evidence_filter: 'physical', 'reliable', or 'all'

    Returns:
        Filtered list of annotations
    """
    if evidence_filter == 'all':
        return annotations

    evidence_codes = PHYSICAL_EVIDENCE_CODES if evidence_filter == 'physical' else RELIABLE_EVIDENCE_CODES

    filtered = []
    for ann in annotations:
        evidence = ann.get('evidence', '')
        # Check if the evidence code (or its prefix) matches our criteria
        if any(evidence.startswith(code) for code in evidence_codes):
            filtered.append(ann)

    return filtered


def assign_proteins_to_pathways(go_annotations: Dict[str, GoAnnotation],
                              pathway_assignments: pd.DataFrame,
                              evidence_filter: str = 'reliable') -> Dict[str, List[Dict]]:
    """
    Assign proteins to pathways based on their GO annotations.

    Args:
        go_annotations: Dictionary mapping protein IDs to GoAnnotation objects
        pathway_assignments: DataFrame with GO_id, GO_description, Pathway columns
        evidence_filter: Evidence quality filter ('physical', 'reliable', 'all')

    Returns:
        Dictionary mapping pathways to lists of protein assignment info
    """
    print(f"Assigning proteins to pathways using evidence filter: {evidence_filter}")

    # Create mapping from GO ID to pathway info
    go_to_pathway = {}
    for _, row in pathway_assignments.iterrows():
        go_to_pathway[row['GO_id']] = {
            'pathway': row['Pathway'],
            'go_description': row['GO_description'],
            'pathway_description': row.get('Description', '')
        }

    print(f"Found {len(go_to_pathway)} GO terms mapped to pathways")

    # Dictionary to store pathway assignments
    pathway_assignments_dict = defaultdict(list)

    # Track statistics
    proteins_with_assignments = 0
    total_assignments = 0

    # Process each protein
    for protein_id, go_annotation in go_annotations.items():
        # Get all annotations for this protein
        all_annotations = go_annotation.processed_annotations

        # Filter annotations by evidence quality
        filtered_annotations = filter_annotations_by_evidence(all_annotations, evidence_filter)

        # Track GO annotations for this protein
        protein_go_terms = []
        protein_pathways = set()

        # Check each annotation against pathway assignments
        for annotation in filtered_annotations:
            go_id = annotation.get('go_id')
            if go_id in go_to_pathway:
                pathway_info = go_to_pathway[go_id]
                pathway = pathway_info['pathway']

                # Add to pathway assignments
                assignment_info = {
                    'protein_id': protein_id,
                    'go_id': go_id,
                    'go_term': annotation.get('term', ''),
                    'go_evidence': annotation.get('evidence', ''),
                    'go_aspect': annotation.get('aspect', ''),
                    'go_description': pathway_info['go_description']
                }

                pathway_assignments_dict[pathway].append(assignment_info)
                protein_pathways.add(pathway)
                total_assignments += 1

        if protein_pathways:
            proteins_with_assignments += 1

    print(f"Results:")
    print(f"  - {proteins_with_assignments}/{len(go_annotations)} proteins assigned to pathways")
    print(f"  - {total_assignments} total pathway assignments")
    print(f"  - {len(pathway_assignments_dict)} pathways have assigned proteins")

    return dict(pathway_assignments_dict)


def filter_pathway_assignments_by_depth(pathway_assignments: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Filter pathway assignments to prioritize specific pathways over broad categories.
    Only removes broad pathways if the protein has more specific pathways within
    the same hierarchy branch.

    For example, if a protein is assigned to both:
    - "Metabolism" (depth 1)
    - "Metabolism > Carbohydrate metabolism > Glycolysis" (depth 3)

    Only the "Metabolism" assignment will be removed, but the protein should still
    appear in "Metabolism > Carbohydrate metabolism" if it's also assigned there.

    Args:
        pathway_assignments: Dictionary mapping pathways to protein assignment lists

    Returns:
        Filtered dictionary with depth-based filtering applied
    """
    # Build protein-to-pathways mapping with depths and hierarchy info
    protein_pathways = defaultdict(list)

    for pathway, assignments in pathway_assignments.items():
        pathway_parts = pathway.split(' > ')
        pathway_depth = len(pathway_parts)
        for assignment in assignments:
            protein_id = assignment['protein_id']
            protein_pathways[protein_id].append({
                'pathway': pathway,
                'pathway_parts': pathway_parts,
                'depth': pathway_depth,
                'assignment': assignment
            })

    # For each protein, determine which pathways to keep
    filtered_assignments = defaultdict(list)
    proteins_filtered = 0

    for protein_id, pathways_info in protein_pathways.items():
        kept_pathways = []
        protein_had_filtering = False

        # Group pathways by hierarchy root
        hierarchy_groups = defaultdict(list)
        for info in pathways_info:
            root = info['pathway_parts'][0]  # First part of hierarchy
            hierarchy_groups[root].append(info)

        # Process each hierarchy group separately
        for root, group_pathways in hierarchy_groups.items():
            max_depth_in_group = max(info['depth'] for info in group_pathways)

            if max_depth_in_group >= 2:
                # Only remove depth 1 pathways within this hierarchy if deeper ones exist
                group_kept = []
                for info in group_pathways:
                    if info['depth'] == 1 and max_depth_in_group > 1:
                        # Skip this broad pathway since we have more specific ones in same hierarchy
                        protein_had_filtering = True
                        continue
                    group_kept.append(info)
                kept_pathways.extend(group_kept)
            else:
                # Keep all pathways in this hierarchy (all depth 1)
                kept_pathways.extend(group_pathways)

        if protein_had_filtering:
            proteins_filtered += 1

        # Add filtered assignments back
        for info in kept_pathways:
            filtered_assignments[info['pathway']].append(info['assignment'])

    print(f"Pathway depth filtering: {proteins_filtered} proteins had broad-only assignments removed")

    # Remove empty pathways
    filtered_assignments = {pathway: assignments for pathway, assignments
                          in filtered_assignments.items() if assignments}

    return dict(filtered_assignments)


def ensure_complete_hierarchical_coverage(pathway_assignments: Dict[str, List[Dict]],
                                        pathway_tree_order: Dict[str, int]) -> Dict[str, List[Dict]]:
    """
    Ensure ALL pathway levels exist and have appropriate proteins, even if they don't
    have direct GO term assignments. This fills in missing intermediate levels.

    Args:
        pathway_assignments: Dictionary mapping pathways to protein assignment lists
        pathway_tree_order: Dictionary mapping pathway names to their tree order

    Returns:
        Dictionary with complete hierarchical coverage
    """
    complete_assignments = defaultdict(list)

    # First, copy all existing assignments
    for pathway, assignments in pathway_assignments.items():
        complete_assignments[pathway].extend(assignments)

    # Collect all unique proteins and their assignments by hierarchy level
    hierarchy_proteins = defaultdict(set)

    # For each existing pathway, collect proteins at each hierarchy level
    for pathway, assignments in pathway_assignments.items():
        pathway_parts = pathway.split(' > ')

        for assignment in assignments:
            protein_id = assignment['protein_id']

            # Add this protein to all parent levels
            for i in range(1, len(pathway_parts) + 1):
                parent_pathway = ' > '.join(pathway_parts[:i])
                hierarchy_proteins[parent_pathway].add(protein_id)

    # Now ensure every pathway level that should exist has its proteins
    missing_levels_filled = 0

    for pathway, protein_ids in hierarchy_proteins.items():
        if pathway not in complete_assignments or len(complete_assignments[pathway]) == 0:
            # This pathway level is missing - create assignments for all its proteins
            for protein_id in protein_ids:
                # Find an existing assignment for this protein to use as template
                template_assignment = None
                for existing_pathway, existing_assignments in pathway_assignments.items():
                    for assignment in existing_assignments:
                        if assignment['protein_id'] == protein_id:
                            template_assignment = assignment.copy()
                            break
                    if template_assignment:
                        break

                if template_assignment:
                    complete_assignments[pathway].append(template_assignment)

            missing_levels_filled += 1

    if missing_levels_filled > 0:
        print(f"Complete hierarchical coverage: filled {missing_levels_filled} missing pathway levels")

    return dict(complete_assignments)


def propagate_to_parent_pathways(pathway_assignments: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Propagate protein assignments to ALL parent pathway levels.
    Every protein that appears in a specific pathway will also appear in all
    parent levels of that pathway.

    Args:
        pathway_assignments: Dictionary mapping pathways to protein assignment lists

    Returns:
        Dictionary with proteins propagated to all parent pathway levels
    """
    propagated_assignments = defaultdict(list)

    # Track all unique proteins and their assignments by pathway
    protein_assignments_by_pathway = defaultdict(dict)

    # First, collect all proteins and their assignments for each pathway
    for pathway, assignments in pathway_assignments.items():
        for assignment in assignments:
            protein_id = assignment['protein_id']
            protein_assignments_by_pathway[pathway][protein_id] = assignment

    # Now propagate every protein to all parent levels
    total_propagations = 0
    proteins_affected = set()

    for pathway, protein_assignments in protein_assignments_by_pathway.items():
        pathway_parts = pathway.split(' > ')

        # For each protein in this pathway
        for protein_id, assignment in protein_assignments.items():

            # Add to the original pathway
            propagated_assignments[pathway].append(assignment)

            # Add to all parent pathways
            for i in range(len(pathway_parts)):
                parent_pathway = ' > '.join(pathway_parts[:i+1])

                # Check if protein already exists in this parent pathway
                existing_proteins = {a['protein_id'] for a in propagated_assignments[parent_pathway]}

                if protein_id not in existing_proteins:
                    # Create assignment for parent pathway
                    parent_assignment = assignment.copy()
                    propagated_assignments[parent_pathway].append(parent_assignment)

                    if parent_pathway != pathway:  # Don't count original assignment
                        total_propagations += 1
                        proteins_affected.add(protein_id)

    if total_propagations > 0:
        print(f"Hierarchical propagation: {len(proteins_affected)} proteins propagated to {total_propagations} parent pathways")

    return dict(propagated_assignments)


def create_pathway_output(pathway_assignments: Dict[str, List[Dict]],
                         uniprot_gene_mapping: Dict[str, str],
                         pathway_tree_order: Dict[str, int]) -> pd.DataFrame:
    """
    Create output DataFrame similar to human_mitocarta_pathways.csv format.

    Args:
        pathway_assignments: Dictionary mapping pathways to protein assignment lists
        uniprot_gene_mapping: Dictionary mapping UniProt IDs to gene names
        pathway_tree_order: Dictionary mapping pathway names to their order in the tree

    Returns:
        DataFrame with pathway information and assigned genes
    """
    output_rows = []

    for pathway_idx, (pathway, assignments) in enumerate(pathway_assignments.items(), 1):
        # Get unique proteins for this pathway
        unique_proteins = {}
        for assignment in assignments:
            protein_id = assignment['protein_id']
            if protein_id not in unique_proteins:
                unique_proteins[protein_id] = []
            unique_proteins[protein_id].append({
                'go_id': assignment['go_id'],
                'go_term': assignment['go_term'],
                'evidence': assignment['go_evidence'],
                'aspect': assignment['go_aspect']
            })

        # Create gene list and GO annotation summary
        uniprot_ids = sorted(unique_proteins.keys())
        uniprot_ids_str = ', '.join(uniprot_ids)

        # Map UniProt IDs to gene names
        gene_names = []
        for uniprot_id in uniprot_ids:
            gene_name = uniprot_gene_mapping.get(uniprot_id, uniprot_id)  # Use UniProt ID if no mapping
            gene_names.append(gene_name)
        gene_names_str = ', '.join(gene_names)

        # Create GO annotation summary for the pathway
        all_go_terms = set()
        for assignment in assignments:
            all_go_terms.add(f"{assignment['go_id']} ({assignment['go_term']})")

        go_annotations_str = '; '.join(sorted(all_go_terms))

        # Split pathway hierarchy to get most specific pathway name
        pathway_parts = pathway.split(' > ')
        pathway_name = pathway_parts[-1]  # Most specific part
        pathway_hierarchy = pathway  # Full hierarchy

        output_rows.append({
            'Pathway_ID': pathway_idx,
            'Pathway': pathway_name,
            'Pathway_Hierarchy': pathway_hierarchy,
            'Uniprot_IDs': uniprot_ids_str,
            'Genes': gene_names_str,
            'GO_Annotations': go_annotations_str,
            'Gene_Count': len(uniprot_ids)
        })

    df = pd.DataFrame(output_rows)

    # Sort by pathway tree order
    def get_pathway_order(pathway_hierarchy):
        # Try to find the most specific pathway name in the tree order
        pathway_parts = pathway_hierarchy.split(' > ')

        # Check from most specific to least specific
        for i in range(len(pathway_parts) - 1, -1, -1):
            pathway_name = pathway_parts[i]
            if pathway_name in pathway_tree_order:
                return pathway_tree_order[pathway_name]

        # If not found, return a large number to put it at the end
        return 999999

    if pathway_tree_order:
        df['_sort_order'] = df['Pathway_Hierarchy'].apply(get_pathway_order)
        df = df.sort_values('_sort_order').drop('_sort_order', axis=1)
        # Reset Pathway_ID to maintain sequential numbering
        df['Pathway_ID'] = range(1, len(df) + 1)

    return df


def main():
    """Main function to run the pathway assignment."""
    parser = argparse.ArgumentParser(description='Assign yeast proteins to pathways based on GO annotations')
    parser.add_argument('--go_annotations',
                       default='go_assignment/data/yeast_go_annotations_sample.pkl',
                       help='Path to GO annotations pickle file')
    parser.add_argument('--pathway_assignments',
                       default='go_assignment/data/go_pathway_assignments_test.csv',
                       help='Path to GO-pathway assignments CSV or JSON file')
    parser.add_argument('--uniprot_mapping',
                       default='go_assignment/data/yeast_mitocarta.csv',
                       help='Path to UniProt ID to gene name mapping CSV file')
    parser.add_argument('--output',
                       default='yeast_pathway_assignments.csv',
                       help='Output CSV file path')
    parser.add_argument('--evidence_filter',
                       choices=['physical', 'reliable', 'all'],
                       default='reliable',
                       help='Evidence quality filter (default: reliable)')
    parser.add_argument('--debug',
                       action='store_true',
                       help='Show detailed debugging information')
    parser.add_argument('--stats',
                       action='store_true',
                       help='Show detailed statistics')
    parser.add_argument('--no_depth_filter',
                       action='store_true',
                       help='Disable pathway depth filtering (keep all assignments)')
    parser.add_argument('--no_propagation',
                       action='store_true',
                       help='Disable hierarchical propagation to parent pathways')
    parser.add_argument('--pathway_tree',
                       default='go_assignment/data/pathway_tree.txt',
                       help='Path to pathway tree file for ordering output')
    parser.add_argument('--examples',
                       action='store_true',
                       help='Show usage examples and exit')

    args = parser.parse_args()

    # Show examples if requested
    if args.examples:
        print_usage_examples()
        sys.exit(0)

    # Load data
    go_annotations = load_go_annotations(args.go_annotations)
    pathway_df = load_pathway_assignments(args.pathway_assignments)
    uniprot_gene_mapping = load_uniprot_gene_mapping(args.uniprot_mapping)
    pathway_tree_order = load_pathway_tree_order(args.pathway_tree)

    # Show debugging information if requested
    if args.debug:
        print_debug_info(go_annotations, pathway_df, args.evidence_filter)

    # Perform assignments
    pathway_assignments = assign_proteins_to_pathways(
        go_annotations, pathway_df, args.evidence_filter
    )

    # Apply hierarchical propagation unless disabled
    if not args.no_propagation:
        pathway_assignments = propagate_to_parent_pathways(pathway_assignments)
        # Ensure complete hierarchical coverage
        pathway_assignments = ensure_complete_hierarchical_coverage(pathway_assignments, pathway_tree_order)

    # Apply pathway depth filtering unless disabled
    if not args.no_depth_filter:
        pathway_assignments = filter_pathway_assignments_by_depth(pathway_assignments)

    # Create output DataFrame
    output_df = create_pathway_output(pathway_assignments, uniprot_gene_mapping, pathway_tree_order)

    # Save results
    output_df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

    # Print summary
    print(f"\nSummary:")
    print(f"  - Total pathways with assignments: {len(output_df)}")

    if len(output_df) > 0:
        print(f"  - Total unique genes assigned: {sum(output_df['Gene_Count'])}")
        print(f"  - Average genes per pathway: {sum(output_df['Gene_Count']) / len(output_df):.1f}")

        # Show top pathways by gene count
        print(f"\nTop pathways by gene count:")
        top_pathways = output_df.nlargest(5, 'Gene_Count')[['Pathway', 'Gene_Count']]
        for _, row in top_pathways.iterrows():
            print(f"  - {row['Pathway']}: {row['Gene_Count']} genes")
    else:
        print(f"  - No genes were assigned to pathways")
        print(f"  - This may be due to:")
        print(f"    * No matching GO IDs between annotations and pathway assignments")
        print(f"    * Evidence filter too restrictive")
        print(f"    * Empty input data")

    # Show detailed statistics if requested
    if args.stats and len(output_df) > 0:
        print_detailed_stats(go_annotations, pathway_assignments, output_df)
    elif args.stats:
        print("\nNo statistics to show - no pathways were assigned.")
        print("Try using a less restrictive evidence filter or check your input data.")

    return output_df


def print_usage_examples():
    """Print usage examples for the script."""
    print("\n=== USAGE EXAMPLES ===")
    print("1. Basic usage with sample data:")
    print("   python assign_proteins_to_pathway.py")
    print("2. Full dataset with reliable evidence:")
    print("   python assign_proteins_to_pathway.py \\")
    print("       --go_annotations data/yeast_all_go_annotations.pkl \\")
    print("       --pathway_assignments data/go_pathway_assignments.csv \\")
    print("       --evidence_filter reliable")
    print("   # OR with JSON format:")
    print("   python assign_proteins_to_pathway.py \\")
    print("       --pathway_assignments data/hierarchy_to_go.json")
    print("\n3. Physical evidence only with debugging:")
    print("   python assign_proteins_to_pathway.py \\")
    print("       --evidence_filter physical --debug --stats")
    print("\n4. Custom UniProt mapping and output file:")
    print("   python assign_proteins_to_pathway.py \\")
    print("       --uniprot_mapping data/yeast_mitocarta.csv \\")
    print("       --output my_yeast_pathways.csv")
    print("\n5. Disable pathway depth filtering:")
    print("   python assign_proteins_to_pathway.py \\")
    print("       --no_depth_filter")
    print("\n6. Custom pathway tree ordering:")
    print("   python assign_proteins_to_pathway.py \\")
    print("       --pathway_tree data/pathway_tree.txt")
    print("\n7. Disable hierarchical propagation:")
    print("   python assign_proteins_to_pathway.py \\")
    print("       --no_propagation")


def print_debug_info(go_annotations: Dict[str, GoAnnotation],
                    pathway_df: pd.DataFrame,
                    evidence_filter: str):
    """Print debugging information about GO terms and pathway mappings."""
    print(f"\n=== DEBUG INFORMATION ===")

    # Collect all GO IDs from annotations
    annotation_go_ids = set()
    evidence_counts = defaultdict(int)
    aspect_counts = defaultdict(int)

    for protein_id, go_annotation in go_annotations.items():
        for annotation in go_annotation.processed_annotations:
            go_id = annotation.get('go_id')
            evidence = annotation.get('evidence', '')
            aspect = annotation.get('aspect', '')

            annotation_go_ids.add(go_id)
            evidence_counts[evidence] += 1
            aspect_counts[aspect] += 1

    # Get pathway GO IDs
    pathway_go_ids = set(pathway_df['GO_id'])

    # Find overlaps
    overlapping_go_ids = annotation_go_ids.intersection(pathway_go_ids)

    print(f"GO Terms in annotations: {len(annotation_go_ids)}")
    print(f"GO Terms in pathway assignments: {len(pathway_go_ids)}")
    print(f"Overlapping GO Terms: {len(overlapping_go_ids)}")

    if overlapping_go_ids:
        print(f"Overlapping terms: {sorted(list(overlapping_go_ids))[:10]}{'...' if len(overlapping_go_ids) > 10 else ''}")

    print(f"\nEvidence code distribution:")
    for evidence, count in sorted(evidence_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {evidence}: {count}")

    print(f"\nGO Aspect distribution:")
    for aspect, count in aspect_counts.items():
        aspect_name = {'C': 'Cellular Component', 'F': 'Molecular Function', 'P': 'Biological Process'}.get(aspect, aspect)
        print(f"  {aspect_name}: {count}")

    # Check evidence filtering impact
    print(f"\nEvidence filtering impact (filter: {evidence_filter}):")
    evidence_codes = PHYSICAL_EVIDENCE_CODES if evidence_filter == 'physical' else (RELIABLE_EVIDENCE_CODES if evidence_filter == 'reliable' else None)

    if evidence_codes:
        filtered_count = 0
        total_count = 0
        for protein_id, go_annotation in go_annotations.items():
            for annotation in go_annotation.processed_annotations:
                total_count += 1
                evidence = annotation.get('evidence', '')
                if any(evidence.startswith(code) for code in evidence_codes):
                    filtered_count += 1
        print(f"  Total annotations: {total_count}")
        print(f"  After evidence filter: {filtered_count} ({filtered_count/total_count*100:.1f}%)")


def print_detailed_stats(go_annotations: Dict[str, GoAnnotation],
                        pathway_assignments: Dict[str, List[Dict]],
                        output_df: pd.DataFrame):
    """Print detailed statistics about the pathway assignments."""
    print(f"\n=== DETAILED STATISTICS ===")

    # Pathway statistics
    print(f"\nPathway distribution:")
    pathway_gene_counts = output_df.set_index('Pathway')['Gene_Count'].to_dict()
    for pathway, gene_count in sorted(pathway_gene_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pathway}: {gene_count} genes")

    # Protein assignment statistics
    protein_pathway_counts = defaultdict(int)
    for pathway, assignments in pathway_assignments.items():
        unique_proteins = set(assignment['protein_id'] for assignment in assignments)
        for protein_id in unique_proteins:
            protein_pathway_counts[protein_id] += 1

    pathway_distribution = defaultdict(int)
    for count in protein_pathway_counts.values():
        pathway_distribution[count] += 1

    print(f"\nProtein multi-pathway assignment:")
    for num_pathways, num_proteins in sorted(pathway_distribution.items()):
        print(f"  {num_proteins} proteins assigned to {num_pathways} pathway(s)")

    # GO term usage statistics
    go_term_usage = defaultdict(int)
    for pathway, assignments in pathway_assignments.items():
        for assignment in assignments:
            go_term_usage[assignment['go_id']] += 1

    print(f"\nMost frequently used GO terms:")
    for go_id, count in sorted(go_term_usage.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {go_id}: used {count} times")

    return output_df


if __name__ == '__main__':
    result_df = main()

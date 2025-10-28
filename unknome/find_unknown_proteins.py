import json
import pandas as pd


def find_unknown_proteins(unknome_df: pd.DataFrame, yeast_mitocarta_df: pd.DataFrame, knownness_cutoff: float = 1.0) -> dict[str, str]:

    """
    Find unknown proteins from the unknome dataset that match mitochondrial proteins.

    This function compares UniProt accessions between two datasets:
    1. unknome_df: Contains protein knownness scores with uniprot_accessions (semicolon-separated)
    2. yeast_mitocarta_df: Contains mitochondrial proteins with UniprotID and Gene columns

    Args:
        unknome_df: DataFrame with columns including 'uniprot_accessions' and 'knownness'
        yeast_mitocarta_df: DataFrame with columns 'UniprotID' and 'Gene'
        knownness_cutoff: Maximum knownness score to consider (default 1.0)

    Returns:
        dict: Mapping from UniprotID (from yeast_mitocarta) to Gene name
    """

    filtered_unknome = unknome_df[unknome_df['knownness'] <= knownness_cutoff].copy()
    print(f"Found {len(filtered_unknome)} yeast proteins with knownness <= {knownness_cutoff}")

    accession_list = []

    for idx, row in filtered_unknome.iterrows():
        accessions = row['uniprot_accessions'].split(';')
        # Create a record for each individual accession
        for accession in accessions:
            accession_list.append({
                'accession': accession.strip(),
                'knownness': row['knownness'],
                'original_idx': idx
            })

    accession_df = pd.DataFrame(accession_list)
    # print(f"Expanded to {len(accession_df)} individual accession entries")

    matches = pd.merge(
        yeast_mitocarta_df[['UniprotID', 'Gene']],
        accession_df,
        left_on='UniprotID',
        right_on='accession',
        how='inner'
    )

    print(f"Found {len(matches)} matches between datasets")

    result_dict = {}
    for _, row in matches.iterrows():
        result_dict[row['UniprotID']] = row['Gene']

    return result_dict

if __name__ == "__main__":
    unknome_df = pd.read_csv("unknome/data/unknome_filtered_yeast.tsv", sep="\t")
    yeast_mitocarta_df = pd.read_csv("go_assignment/data/yeast_mitocarta.csv")

    # Test the function with different cutoffs
    knownness_cutoff = 1.0
    result = find_unknown_proteins(unknome_df, yeast_mitocarta_df, knownness_cutoff=knownness_cutoff)

    with open(f"unknome/data/unknown_proteins_cutoff_{knownness_cutoff}.json", 'w') as f:
        json.dump(result, f, indent=4)

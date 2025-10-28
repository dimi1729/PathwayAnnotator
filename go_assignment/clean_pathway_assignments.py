import pandas as pd

def remove_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate pathway assignments from a DataFrame.

    Args:
        df: DataFrame containing pathway assignments

    Returns:
        DataFrame with duplicate assignments removed
    """
    # Remove duplicate assignments
    df = df.drop_duplicates(subset=['GO_id'])

    return df

def highlight_incorrect_pathways(df: pd.DataFrame, allowed_terms: list) -> pd.DataFrame:
    """
    Highlight incorrect pathway assignments in a DataFrame.

    Args:
        df: DataFrame containing pathway assignments

    Returns:
        DataFrame with incorrect assignments highlighted
    """
    # Highlight incorrect assignments
    def incorrect(pathway: str, allowed_terms: list) -> bool:
        try:
            pathway_terms = pathway.split(' > ')
        except Exception as e:
            print(pathway)
            return
        for term in pathway_terms:
            if term not in allowed_terms:
                print(f"Invalid term: {term}")
                print(f"Pathway: {pathway}")
                # print(f"Allowed terms: {allowed_terms}")
                exit(1)
                return True
        return False

    df['incorrect'] = df['Pathway'].apply(incorrect, args=(allowed_terms,))

    return df

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('data/go_pathway_assignments.csv')

    # Remove duplicate assignments
    df = remove_repeats(df)

    # # Check for NaN values in Pathway column
    # nan_indices = df[df['Pathway'].isna()]
    # if not nan_indices.empty:
    #     print("Rows with NaN values in Pathway column:")
    #     print(nan_indices)

    # exit(1)

    # Highlight incorrect assignments
    # Parse text file and save each line in a list minus leading spaces
    allowed_terms = ["No appropriate pathway"]
    with open('data/pathway_tree.txt', 'r') as f:
        for line in f:
            allowed_terms.append(line.lstrip().rstrip())

    df = highlight_incorrect_pathways(df, allowed_terms)
    # print(df)
    # exit(0)

    # Save cleaned data
    df.to_csv('data/clean_pathway_assignments.csv', index=False)

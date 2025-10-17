import time
import requests
import polars as pl
import pickle
import json
from pathlib import Path

from typing import Optional, Union, Dict, Any

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"  # Website API search endpoint

class GoAnnotation:
    """
    Simple class to provide methods to parse uniprot results
    """

    def __init__(self, raw_result: dict):
        """
        Initialize with raw UniProt API result for a single protein entry.

        Args:
            raw_result: Raw UniProt API response entry containing uniProtKBCrossReferences
        """
        self.raw_result = raw_result
        self._processed_annotations: Optional[list[dict]] = None

    @property
    def processed_annotations(self) -> list[dict]:
        """
        Get processed GO annotations as list of dicts with go_id, term, evidence, aspect.
        Cached after first access.
        """
        if self._processed_annotations is None:
            self._processed_annotations = self._process_go_annotations()
        return self._processed_annotations

    def _process_go_annotations(self) -> list[dict]:
        """
        Process raw UniProt result to extract GO annotations.
        """
        annotations = []
        cross_refs = self.raw_result.get("uniProtKBCrossReferences", [])

        for ref in cross_refs:
            if ref.get("database") == "GO":
                go_id = ref.get("id")

                # Extract properties
                go_term = None
                go_evidence = None

                for prop in ref.get("properties", []):
                    if prop.get("key") == "GoTerm":
                        go_term = prop.get("value")
                    elif prop.get("key") == "GoEvidenceType":
                        go_evidence = prop.get("value")

                # Determine aspect from GO term prefix
                aspect = None
                if go_term:
                    if go_term.startswith("C:"):
                        aspect = "C"  # Cellular Component
                    elif go_term.startswith("F:"):
                        aspect = "F"  # Molecular Function
                    elif go_term.startswith("P:"):
                        aspect = "P"  # Biological Process

                # Clean up the term (remove prefix)
                if go_term and ":" in go_term:
                    go_term = go_term.split(":", 1)[1]

                annotations.append({
                    "go_id": go_id,
                    "term": go_term,
                    "evidence": go_evidence,
                    "aspect": aspect,
                })

        return annotations

    def get_go_ids(self, min_evidence: Optional[str] = None) -> list[str]:
        """
        Get list of GO IDs, optionally filtered by evidence level.

        Args:
            min_evidence: Minimum evidence level to include (not implemented yet)

        Returns:
            List of GO IDs (e.g., ['GO:0005743', 'GO:0031966'])
        """
        return [ann["go_id"] for ann in self.processed_annotations if ann["go_id"]]

    def get_annotations_by_aspect(self, aspect: str) -> list[dict]:
        """
        Get annotations filtered by aspect (C, F, or P).

        Args:
            aspect: 'C' for Cellular Component, 'F' for Molecular Function, 'P' for Biological Process

        Returns:
            List of annotation dicts for the specified aspect
        """
        return [ann for ann in self.processed_annotations if ann["aspect"] == aspect]

    def get_cellular_component_terms(self) -> list[str]:
        """Get list of cellular component GO terms."""
        return [ann["term"] for ann in self.get_annotations_by_aspect("C") if ann["term"]]

    def get_molecular_function_terms(self) -> list[str]:
        """Get list of molecular function GO terms."""
        return [ann["term"] for ann in self.get_annotations_by_aspect("F") if ann["term"]]

    def get_biological_process_terms(self) -> list[str]:
        """Get list of biological process GO terms."""
        return [ann["term"] for ann in self.get_annotations_by_aspect("P") if ann["term"]]

    def __len__(self) -> int:
        """Return number of GO annotations."""
        return len(self.processed_annotations)

    def __iter__(self):
        """Iterate over processed annotations."""
        return iter(self.processed_annotations)

    def __repr__(self) -> str:
        """String representation showing number of annotations."""
        return f"GoAnnotation({len(self)} annotations)"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'raw_result': self.raw_result,
            'processed_annotations': self.processed_annotations
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GoAnnotation':
        """Create GoAnnotation from dictionary."""
        instance = cls(data['raw_result'])
        instance._processed_annotations = data['processed_annotations']
        return instance

class GoAnnotationCollection:
    """
    Wrapper class for managing collections of GO annotations with save/load functionality.
    """

    def __init__(self, annotations: Optional[dict[str, GoAnnotation]] = None):
        """
        Initialize collection with optional annotations dict.

        Args:
            annotations: Dict mapping uniprot_id -> GoAnnotation
        """
        self.annotations = annotations or {}

    def __getitem__(self, key: str) -> GoAnnotation:
        """Get annotation by UniProt ID."""
        return self.annotations[key]

    def __setitem__(self, key: str, value: GoAnnotation):
        """Set annotation for UniProt ID."""
        self.annotations[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if UniProt ID is in collection."""
        return key in self.annotations

    def __len__(self) -> int:
        """Number of proteins in collection."""
        return len(self.annotations)

    def __iter__(self):
        """Iterate over UniProt IDs."""
        return iter(self.annotations)

    def items(self):
        """Get (uniprot_id, GoAnnotation) pairs."""
        return self.annotations.items()

    def keys(self):
        """Get UniProt IDs."""
        return self.annotations.keys()

    def values(self):
        """Get GoAnnotation objects."""
        return self.annotations.values()

    def get(self, key: str, default: Optional[GoAnnotation] = None):
        """Get annotation with default."""
        return self.annotations.get(key, default)

    def add_annotations(self, new_annotations):
        """Add new annotations to the collection."""
        self.annotations.update(new_annotations)

    def get_all_go_ids(self) -> set[str]:
        """Get all unique GO IDs across all proteins."""
        all_go_ids = set()
        for go_ann in self.annotations.values():
            all_go_ids.update(go_ann.get_go_ids())
        return all_go_ids

    def get_proteins_with_go_id(self, go_id: str) -> list[str]:
        """Get list of UniProt IDs that have a specific GO ID."""
        proteins = []
        for uniprot_id, go_ann in self.annotations.items():
            if go_id in go_ann.get_go_ids():
                proteins.append(uniprot_id)
        return proteins

    def filter_by_aspect(self, aspect: str):
        """Get annotations for all proteins filtered by aspect."""
        result = {}
        for uniprot_id, go_ann in self.annotations.items():
            filtered_annotations = go_ann.get_annotations_by_aspect(aspect)
            # Create a new GoAnnotation with filtered results
            new_go_ann = GoAnnotation({"uniProtKBCrossReferences": []})
            new_go_ann._processed_annotations = filtered_annotations
            result[uniprot_id] = new_go_ann
        return GoAnnotationCollection(result)

    def summary(self) -> dict:
        """Get summary statistics of the collection."""
        total_proteins = len(self.annotations)
        total_annotations = sum(len(go_ann) for go_ann in self.annotations.values())
        proteins_with_annotations = sum(1 for go_ann in self.annotations.values() if len(go_ann) > 0)
        unique_go_ids = len(self.get_all_go_ids())

        aspect_counts = {'C': 0, 'F': 0, 'P': 0}
        for go_ann in self.annotations.values():
            for annotation in go_ann:
                aspect = annotation.get('aspect')
                if aspect in aspect_counts:
                    aspect_counts[aspect] += 1

        return {
            'total_proteins': total_proteins,
            'proteins_with_annotations': proteins_with_annotations,
            'total_annotations': total_annotations,
            'unique_go_ids': unique_go_ids,
            'cellular_component_annotations': aspect_counts['C'],
            'molecular_function_annotations': aspect_counts['F'],
            'biological_process_annotations': aspect_counts['P']
        }

    def save_pickle(self, filepath):
        """
        Save collection to pickle file.

        Args:
            filepath: Path to save the pickle file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self.annotations, f)

    def save_json(self, filepath):
        """
        Save collection to JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        serializable_data = {}
        for uniprot_id, go_ann in self.annotations.items():
        serializable_data[uniprot_id] = go_ann.to_dict()

        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)

    @classmethod
    def load_pickle(cls, filepath):
        """
        Load collection from pickle file.

        Args:
            filepath: Path to the pickle file

        Returns:
            GoAnnotationCollection instance
        """
        with open(filepath, 'rb') as f:
            annotations = pickle.load(f)
        return cls(annotations)

    @classmethod
    def load_json(cls, filepath):
        """
        Load collection from JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            GoAnnotationCollection instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert back to GoAnnotation objects
        annotations = {}
        for uniprot_id, go_data in data.items():
            annotations[uniprot_id] = GoAnnotation.from_dict(go_data)

        return cls(annotations)

    @classmethod
    def from_fetch_result(cls, fetch_result):
        """
        Create collection from fetch_go_annotations_for_uniprot_ids result.

        Args:
            fetch_result: Result from fetch_go_annotations_for_uniprot_ids

        Returns:
            GoAnnotationCollection instance
        """
        return cls(fetch_result)

    def __repr__(self) -> str:
        """String representation of the collection."""
        summary = self.summary()
        return (f"GoAnnotationCollection({summary['total_proteins']} proteins, "
                f"{summary['proteins_with_annotations']} with annotations, "
                f"{summary['total_annotations']} total annotations)")


def _chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def fetch_go_annotations_for_uniprot_ids(
    accessions: list[str],
    batch_size: int = 100,
    max_retries: int = 3,
    retry_backoff: float = 1.5,
    timeout: int = 30,
) -> GoAnnotationCollection:
    """
    Fetch GO annotations for UniProt IDs and return GoAnnotation objects.

    Args:
        accessions: List of UniProt IDs
        batch_size: Number of IDs to query per API call
        max_retries: Maximum number of retry attempts
        retry_backoff: Backoff factor for retries
        timeout: Request timeout in seconds

    Returns:
        GoAnnotationCollection containing GoAnnotation objects
    """
    # Prepare output dictionary
    result: dict[str, GoAnnotation] = {}

    session = requests.Session()
    headers = {
        "Accept": "application/json",
        "User-Agent": "mito-go-mapper/1.0 (+your-email-or-url)"
    }

    for batch in _chunked(list(dict.fromkeys(accessions)), batch_size):
        # Build OR query for accessions
        q = " OR ".join(f"accession:{acc}" for acc in batch)
        params = {
            "query": q,
            "format": "json",
            "size": 500  # enough to return all matched entries for this batch
        }

        # Basic retry loop
        for attempt in range(1, max_retries + 1):
            try:
                resp = session.get(UNIPROT_SEARCH_URL, params=params, headers=headers, timeout=timeout)
                if resp.status_code in (429, 502, 503, 504):
                    # Backoff on rate-limit or transient errors
                    delay = retry_backoff ** (attempt - 1)
                    time.sleep(delay)
                    continue
                elif resp.status_code == 400:
                    # Bad request - likely invalid accession format
                    # Try each accession individually to identify valid ones
                    data = {"results": []}
                    for acc in batch:
                        try:
                            individual_params = {
                                "query": f"accession:{acc}",
                                "format": "json",
                                "size": 1
                            }
                            individual_resp = session.get(UNIPROT_SEARCH_URL, params=individual_params, headers=headers, timeout=timeout)
                            if individual_resp.status_code == 200:
                                individual_data = individual_resp.json()
                                data["results"].extend(individual_data.get("results", []))
                            # If individual query fails, skip this accession
                        except requests.RequestException:
                            # Skip this accession if it fails
                            continue
                    break
                else:
                    resp.raise_for_status()
                data = resp.json()
                break
            except requests.RequestException:
                if attempt == max_retries:
                    # Give up on this batch; leave entries as empty GoAnnotation objects
                    data = {"results": []}
                else:
                    time.sleep(retry_backoff ** (attempt - 1))
                    continue

        # Store raw results in GoAnnotation objects
        for entry in data.get("results", []):
            acc = entry.get("primaryAccession")
            if not acc:
                continue

            # Store the entire raw entry in the GoAnnotation object
            result[acc] = GoAnnotation(entry)

    # Initialize any missing accessions with empty GoAnnotation objects
    for acc in accessions:
        if acc not in result:
            result[acc] = GoAnnotation({"uniProtKBCrossReferences": []})

    return GoAnnotationCollection(result)

def fetch_go_annotations_for_uniprot_ids_legacy(
    accessions: list[str],
    batch_size: int = 100,
    max_retries: int = 3,
    retry_backoff: float = 1.5,
    timeout: int = 30,
) -> dict[str, list[dict]]:
    """
    Legacy function that returns processed annotations directly as dicts.
    Use fetch_go_annotations_for_uniprot_ids for the new GoAnnotation-based approach.

    Returns:
        Dict mapping uniprot_id -> list of annotation dicts
    """
    go_annotations = fetch_go_annotations_for_uniprot_ids(
        accessions, batch_size, max_retries, retry_backoff, timeout
    )

    # Convert GoAnnotation objects back to legacy format
    legacy_result = {}
    for acc, go_ann in go_annotations.items():
        legacy_result[acc] = go_ann.processed_annotations

    return legacy_result

if __name__ == '__main__':

    uniprots_file = "/home/abhinav22/Gohillab_AF/yeast_alphafold_results/yeast_mitocarta_plus_uniprot.csv"
    df = pl.read_csv(uniprots_file)
    uniprot_ids: list[str] = df["uniprot_id"].to_list()
    file_collection = fetch_go_annotations_for_uniprot_ids(uniprot_ids[0:5])

    print(f"Saving {file_collection}")

    file_collection.save_pickle("data/yeast_go_annotations_sample.pkl")

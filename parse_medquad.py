"""
parse_medquad.py
----------------
Downloads and parses the MedQuAD XML database into .txt files
that our RAG pipeline can read and embed into ChromaDB.

What this script does:
  1. Scans all XML files inside the MedQuAD folder
  2. Parses each XML file to extract:
       - Disease / condition name
       - Questions and answers (Q&A pairs)
       - Focus area (e.g. symptoms, treatment, diagnosis)
  3. Groups Q&A pairs by topic into structured .txt documents
  4. Saves them into backend/medical_docs/medquad/ folder
  5. Shows a summary of how many documents were created

After running this script:
  → Run build_vectors.py to embed the new documents into ChromaDB

Usage:
  python parse_medquad.py

Requirements:
  pip install lxml tqdm
  (lxml is faster and more robust than Python's built-in xml parser)
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# Try lxml first (faster), fall back to built-in xml
try:
    from lxml import etree as ET
    print("[parser] Using lxml parser (fast mode)")
except ImportError:
    import xml.etree.ElementTree as ET
    print("[parser] Using built-in xml parser")
    print("[parser] TIP: pip install lxml for faster parsing")

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False
    print("[parser] TIP: pip install tqdm for progress bars")


# ──────────────────────────────────────────────────────────────
# 1.  Configuration
# ──────────────────────────────────────────────────────────────

# Path to the cloned MedQuAD folder
# Assumes MedQuAD is cloned one level above backend/
MEDQUAD_ROOT = Path(__file__).parent / "MedQuAD"

# Output folder — inside medical_docs so RAG pipeline finds it
OUTPUT_FOLDER = Path(__file__).parent / "medical_docs" / "medquad"

# Maximum Q&A pairs per output file (keeps files readable)
MAX_QA_PER_FILE = 20

# Minimum answer length to include (filter out empty/useless answers)
MIN_ANSWER_LENGTH = 50

# Which MedQuAD source folders to include
# Comment out any you don't want
INCLUDE_FOLDERS = [
    "1_CancerGov_QA",       # Cancer information
    "2_GARD_QA",            # Genetic and Rare Diseases
    "3_GHR_QA",             # Genetics Home Reference
    "4_MedlinePlus_QA",     # MedlinePlus health topics
    "5_NIDDK_QA",           # Digestive, Diabetes, Kidney diseases
    "6_NINDS_QA",           # Neurological disorders
    "7_SeniorHealth_QA",    # Senior health topics
    "8_OrthoInfo_QA",       # Orthopaedic conditions
    "9_CDC_QA",             # CDC health topics
    "10_MPlus_Health_TopicsQA",  # More MedlinePlus topics
    "11_MPlusDrugs_QA",     # Drug information
]


# ──────────────────────────────────────────────────────────────
# 2.  XML Parser
# ──────────────────────────────────────────────────────────────

def parse_xml_file(xml_path: Path) -> dict | None:
    """
    Parse a single MedQuAD XML file and extract structured data.

    MedQuAD XML structure:
      <Document>
        <Focus>Disease or Topic Name</Focus>
        <QAPairs>
          <QAPair pid="1">
            <Question qid="..." qtype="symptoms">What are the symptoms?</Question>
            <Answer>The symptoms include...</Answer>
          </QAPair>
          ...
        </QAPairs>
      </Document>

    Returns
    -------
    dict with keys:
        focus    : str         Topic/disease name
        source   : str         Source folder name
        qa_pairs : list[dict]  List of {question, answer, qtype}
    Or None if parsing fails.
    """
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        # Get the focus (topic name)
        focus_el = root.find("Focus")
        focus = focus_el.text.strip() if focus_el is not None and focus_el.text else xml_path.stem

        # Get source folder name
        source = xml_path.parent.name

        # Extract Q&A pairs
        qa_pairs = []
        for qa_pair in root.findall(".//QAPair"):
            question_el = qa_pair.find("Question")
            answer_el   = qa_pair.find("Answer")

            if question_el is None or answer_el is None:
                continue

            question = question_el.text.strip() if question_el.text else ""
            answer   = answer_el.text.strip()   if answer_el.text   else ""
            qtype    = question_el.get("qtype", "general")

            # Skip low-quality entries
            if not question or not answer:
                continue
            if len(answer) < MIN_ANSWER_LENGTH:
                continue

            qa_pairs.append({
                "question": question,
                "answer":   answer,
                "qtype":    qtype,
            })

        if not qa_pairs:
            return None

        return {
            "focus":    focus,
            "source":   source,
            "qa_pairs": qa_pairs,
        }

    except Exception as e:
        # Silently skip malformed XML files
        return None


# ──────────────────────────────────────────────────────────────
# 3.  Text Document Builder
# ──────────────────────────────────────────────────────────────

def build_document_text(doc_data: dict) -> str:
    """
    Convert parsed XML data into a structured .txt document
    that our RAG pipeline can embed.

    Format:
        TOPIC: <focus>
        SOURCE: <source>

        OVERVIEW
        Q: <question>
        A: <answer>

        SYMPTOMS
        Q: ...
        A: ...

        TREATMENT
        Q: ...
        A: ...
        ...
    """
    focus    = doc_data["focus"]
    source   = doc_data["source"]
    qa_pairs = doc_data["qa_pairs"]

    lines = []
    lines.append(f"TOPIC: {focus}")
    lines.append(f"SOURCE: {source}")
    lines.append("")

    # Group Q&A pairs by question type
    grouped = defaultdict(list)
    for qa in qa_pairs[:MAX_QA_PER_FILE]:
        qtype = qa["qtype"].upper().replace("-", " ").replace("_", " ")
        grouped[qtype].append(qa)

    # Write each group as a section
    for section_name, pairs in grouped.items():
        lines.append(section_name)
        lines.append("-" * 40)
        for qa in pairs:
            lines.append(f"Q: {qa['question']}")
            lines.append(f"A: {qa['answer']}")
            lines.append("")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 4.  Safe filename generator
# ──────────────────────────────────────────────────────────────

def safe_filename(text: str, max_length: int = 60) -> str:
    """
    Convert a topic name into a safe filename.

    Example:
        "Type 2 Diabetes Mellitus" → "type_2_diabetes_mellitus"
    """
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)   # remove special chars
    text = re.sub(r"\s+", "_", text)        # spaces to underscores
    text = re.sub(r"-+", "_", text)         # dashes to underscores
    text = text.strip("_")
    return text[:max_length]


# ──────────────────────────────────────────────────────────────
# 5.  Main parser function
# ──────────────────────────────────────────────────────────────

def parse_medquad(
    medquad_root: Path = MEDQUAD_ROOT,
    output_folder: Path = OUTPUT_FOLDER,
) -> None:
    """
    Main function — scans all MedQuAD XML files and converts them
    to .txt documents in the output folder.

    Parameters
    ----------
    medquad_root  : Path   Root folder of the cloned MedQuAD repo
    output_folder : Path   Where to save the .txt output files
    """

    # ── Validate MedQuAD folder ───────────────────────────────
    if not medquad_root.exists():
        print(f"\n❌ MedQuAD folder not found at: {medquad_root}")
        print("\nPlease run this command first:")
        print("  git clone https://github.com/abachaa/MedQuAD.git")
        print("\nMake sure to clone it here:")
        print(f"  {medquad_root.parent}")
        return

    # ── Create output folder ──────────────────────────────────
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n[parser] Output folder: {output_folder}")

    # ── Collect all XML files ─────────────────────────────────
    all_xml_files = []
    for folder_name in INCLUDE_FOLDERS:
        folder_path = medquad_root / folder_name
        if folder_path.exists():
            xml_files = list(folder_path.glob("*.xml"))
            all_xml_files.extend(xml_files)
            print(f"[parser] Found {len(xml_files):4d} XML files in {folder_name}")
        else:
            print(f"[parser] WARNING: Folder not found: {folder_name}")

    print(f"\n[parser] Total XML files to process: {len(all_xml_files)}")

    if not all_xml_files:
        print("❌ No XML files found. Check the MedQuAD folder path.")
        return

    # ── Process each XML file ─────────────────────────────────
    success_count   = 0
    skip_count      = 0
    duplicate_names = defaultdict(int)

    iterator = tqdm(all_xml_files, desc="Parsing XML") if TQDM else all_xml_files

    for xml_path in iterator:
        if not TQDM:
            # Simple progress every 500 files
            processed = all_xml_files.index(xml_path) + 1
            if processed % 500 == 0:
                print(f"[parser] Processed {processed}/{len(all_xml_files)}…")

        # Parse the XML
        doc_data = parse_xml_file(xml_path)
        if doc_data is None:
            skip_count += 1
            continue

        # Build the text content
        text_content = build_document_text(doc_data)

        # Generate a unique filename
        base_name = safe_filename(doc_data["focus"])
        if not base_name:
            base_name = safe_filename(xml_path.stem)

        # Handle duplicate names by appending a counter
        duplicate_names[base_name] += 1
        if duplicate_names[base_name] > 1:
            base_name = f"{base_name}_{duplicate_names[base_name]}"

        output_path = output_folder / f"{base_name}.txt"

        # Write the file
        output_path.write_text(text_content, encoding="utf-8")
        success_count += 1

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  ✅ MedQuAD Parsing Complete!")
    print(f"{'=' * 55}")
    print(f"  Documents created : {success_count:,}")
    print(f"  Files skipped     : {skip_count:,}")
    print(f"  Output folder     : {output_folder}")
    print(f"{'=' * 55}")
    print(f"\n📌 Next step: rebuild the vector store by running:")
    print(f"   python build_vectors.py")


# ──────────────────────────────────────────────────────────────
# 6.  Run
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  MedQuAD Database Parser")
    print("=" * 55)
    parse_medquad()
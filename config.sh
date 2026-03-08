#!/usr/bin/env bash
set -euo pipefail

# config.sh
# Builds an RCSB search payload from simple variables and saves the search JSON.
# Edit the variables below to change the search from the frontend.

RCSB_SEARCH_URL="${RCSB_SEARCH_URL:-https://search.rcsb.org/rcsbsearch/v2/query}"
OUTPUT_JSON="${OUTPUT_JSON:-pdb_search_results.json}"
RETURN_TYPE="${RETURN_TYPE:-entry}"

# --- Search parameters (edit these) ---
# Structure keywords (OR group). Provide exact phrases as seen in the UI.
STRUCT_KEYWORDS=(
  "RNA BINDING PROTEIN, RNA"
  "DNA BINDING PROTEIN, DNA"
  "DNA"
  "RNA"
)

# Scientific name exact match
ORGANISM="${ORGANISM:-Homo sapiens}"

# Maximum (exclusive) resolution in Å (use decimal number)
MAX_RESOLUTION="${MAX_RESOLUTION:-2.5}"

# --- end of editable variables ---

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to construct JSON payload. Please install python3." >&2
  exit 2
fi

# Build KW JSON array for Python code
KW_JSON=$(printf '%s\n' "${STRUCT_KEYWORDS[@]}" | python3 -c 'import sys,json; print(json.dumps([l.rstrip("\n") for l in sys.stdin if l.rstrip("\n")]))')

PAYLOAD=$(KW_JSON="$KW_JSON" ORGANISM="$ORGANISM" MAX_RESOLUTION="$MAX_RESOLUTION" python3 - <<'PY'
import os, json
keywords = json.loads(os.environ['KW_JSON'])
org = os.environ.get('ORGANISM')
maxr = float(os.environ.get('MAX_RESOLUTION'))

kw_nodes = []
for kw in keywords:
    kw_nodes.append({
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "struct_keywords.pdbx_keywords",
            "operator": "exact_match",
            "value": kw
        }
    })

obj = {
  "query": {
    "type": "group",
    "logical_operator": "and",
    "nodes": [
      {
        "type": "group",
        "logical_operator": "or",
        "nodes": [
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_entry_info.struct_keywords",
              "operator": "contains_phrase",
              "value": "DNA"
            }
          },
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_entry_info.struct_keywords",
              "operator": "contains_phrase",
              "value": "RNA"
            }
          }
        ]
      },
      {
        "type": "terminal",
        "service": "text",
        "parameters": {
          "attribute": "rcsb_entity_source_organism.scientific_name",
          "operator": "exact_match",
          "value": "Homo sapiens"
        }
      },
      {
        "type": "terminal",
        "service": "numeric",
        "parameters": {
          "attribute": "rcsb_entry_info.resolution_combined",
          "operator": "less_or_equal",
          "value": 2.5
        }
      }
    ]
  },
  "return_type": "entry"
}

print(json.dumps(obj))
PY
)

echo "Posting search to ${RCSB_SEARCH_URL} and saving to ${OUTPUT_JSON}..."
curl -sS -H "Content-Type: application/json" -d "$PAYLOAD" "$RCSB_SEARCH_URL" -o "$OUTPUT_JSON"

if [ $? -eq 0 ]; then
  echo "Saved search results to ${OUTPUT_JSON}"
  if command -v jq >/dev/null 2>&1; then
    echo "Summary (first-level keys):"
    jq 'keys' "${OUTPUT_JSON}" | sed -n '1,200p'
  fi
else
  echo "Request failed; no output saved." >&2
  exit 3
fi

echo "Done. Edit the top variables in config.sh to modify the search from your frontend."

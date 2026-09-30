"""Small, rebuildable recording indexes; native sample files remain immutable."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def build(folder):
    folder = Path(folder)
    chunks = []
    for path in sorted((folder / "samples").glob("*.parquet")):
        metadata = pq.ParquetFile(path).metadata
        schema = pq.read_schema(path)
        offset_column = schema.get_field_index("offset")
        lower, upper = [], []
        for index in range(metadata.num_row_groups):
            stats = metadata.row_group(index).column(offset_column).statistics
            if not stats or not stats.has_min_max:
                values = pq.read_table(path, columns=["offset"]).column("offset").to_numpy()
                lower.append(float(values.min()))
                upper.append(float(values.max()))
                break
            lower.append(float(stats.min))
            upper.append(float(stats.max))
        chunks.append({"file": path.name, "start": min(lower), "end": max(upper)})
    (folder / "chunk_index.json").write_text(json.dumps(chunks), encoding="utf-8")
    schema = pa.schema(
        [
            ("offset", pa.int64()),
            ("min", pa.float64()),
            ("max", pa.float64()),
            ("mean", pa.float64()),
            ("count", pa.int64()),
            ("channel", pa.string()),
        ]
    )
    temporary = folder / "overview.tmp.parquet"
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for path in sorted((folder / "summaries").glob("*.parquet")):
            table = pq.read_table(path).select(schema.names).cast(schema)
            writer.write_table(table)
    temporary.replace(folder / "overview.parquet")
    return chunks

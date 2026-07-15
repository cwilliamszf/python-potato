import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpcr_ensemble import msa_subsample as sub

QUERY = "ACDEFGHIKLMNPQRSTVWY"


def make_seq(base: str, n_mutations: int, rng: random.Random) -> str:
    seq = list(base)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    positions = rng.sample(range(len(seq)), n_mutations)
    for p in positions:
        seq[p] = rng.choice(alphabet)
    return "".join(seq)


def build_a3m(tmp_path) -> Path:
    rng = random.Random(42)
    lines = [f">query", QUERY]
    # Family A: close to query (few mutations)
    for i in range(15):
        lines += [f">famA_{i}", make_seq(QUERY, 2, rng)]
    # Family B: far from query and from family A (many mutations)
    base_b = make_seq(QUERY, 12, rng)
    for i in range(15):
        lines += [f">famB_{i}", make_seq(base_b, 2, rng)]
    path = tmp_path / "test.a3m"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_parse_a3m_roundtrip(tmp_path):
    path = build_a3m(tmp_path)
    query, members = sub.parse_a3m(path)
    assert query.header == ">query"
    assert query.seq == QUERY
    assert len(members) == 30


def test_strip_inserts_removes_lowercase():
    assert sub.strip_inserts("AC-deFG") == "AC-FG"


def test_random_subsample_respects_fraction_and_min(tmp_path):
    path = build_a3m(tmp_path)
    _, members = sub.parse_a3m(path)
    rng = random.Random(0)
    subset = sub.random_subsample(members, fraction=0.5, min_seqs=4, rng=rng)
    assert len(subset) == 15  # 30 * 0.5
    subset_small = sub.random_subsample(members, fraction=0.01, min_seqs=4, rng=rng)
    assert len(subset_small) == 4  # floor hits min_seqs


def test_cluster_sequences_separates_families(tmp_path):
    path = build_a3m(tmp_path)
    _, members = sub.parse_a3m(path)
    clusters_by_t = sub.cluster_sequences(members, distance_thresholds=[0.3])
    clusters = clusters_by_t[0.3]
    # Expect roughly two clusters recovering family A vs family B
    sizes = sorted(len(c) for c in clusters)
    assert len(clusters) >= 2
    assert sizes[-1] >= 10 and sizes[-2] >= 10


def test_generate_diverse_msas_manifest(tmp_path):
    path = build_a3m(tmp_path)
    out_dir = tmp_path / "msas"
    manifest = sub.generate_diverse_msas(
        path,
        out_dir,
        random_fractions=[0.1, 1.0],
        n_random_replicates=2,
        cluster_distance_thresholds=[0.3],
        min_cluster_size=3,
        min_seqs=2,
    )
    assert len(manifest) > 0
    for entry in manifest:
        p = Path(entry["path"])
        assert p.exists()
        query, members = sub.parse_a3m(p)
        assert query.seq == QUERY  # query always retained
        assert len(members) == entry["depth"]

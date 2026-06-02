use std::collections::BTreeMap;

use common::types::{PointOffsetType, ScoreType, ScoredPointOffset};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(crate) struct MirageGolden {
    pub version: u32,
    pub source: String,
    pub metric: String,
    pub vector_generator: String,
    pub layer0_injected_source: String,
    pub n: usize,
    pub dim: usize,
    pub params: MirageGoldenParams,
    pub levels_zero_based: Vec<usize>,
    pub upper_insert_order_by_level: BTreeMap<usize, Vec<usize>>,
    pub layer0: Vec<Vec<usize>>,
    pub layer0_top_s: Vec<Vec<usize>>,
    pub layer0_injected: Vec<Vec<usize>>,
    pub layer0_injected_stats: Layer0Stats,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MirageGoldenParams {
    pub s: usize,
    pub r: usize,
    pub iter: usize,
    pub num_reverse_edges: usize,
    pub mirage_seed: u64,
    pub hnsw_m: usize,
    pub hnsw_m0: usize,
    pub layer0_k: usize,
    pub hnsw_level_seed: u32,
    pub shuffle_seed: u32,
    pub threads: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Layer0Stats {
    pub total_edges: usize,
    pub min_degree: usize,
    pub max_degree: usize,
    pub avg_degree: f64,
    pub degree_histogram: BTreeMap<usize, usize>,
    pub reciprocal_edges: usize,
    pub reciprocity_ratio: f64,
    pub weak_connected_components: usize,
    pub largest_weak_component: usize,
}

pub(crate) fn load_cpp_golden_n64_d8_l2() -> MirageGolden {
    let fixture = serde_json::from_str::<MirageGolden>(include_str!(
        "fixtures/mirage_cpp_golden_n64_d8_l2.json"
    ))
    .expect("MIRAGE C++ golden fixture must be valid JSON");

    assert_eq!(fixture.version, 1);
    assert!(fixture.source.contains("MIRAGE.cpp") || fixture.source.contains("Mirage::build"));
    assert!(
        fixture.layer0_injected_source.contains("authoritative")
            || fixture.layer0_injected_source.contains("diagnostic")
    );
    assert_eq!(fixture.metric, "l2");
    assert_eq!(fixture.vector_generator, "integer-affine-v1");
    assert_eq!(fixture.params.threads, 1);
    assert_eq!(fixture.params.layer0_k, fixture.params.s);
    assert_eq!(fixture.params.hnsw_m0, fixture.params.hnsw_m * 2);
    assert_eq!(fixture.levels_zero_based.len(), fixture.n);
    assert_eq!(fixture.layer0.len(), fixture.n);
    assert_eq!(fixture.layer0_top_s.len(), fixture.n);
    assert_eq!(fixture.layer0_injected.len(), fixture.n);
    fixture
}

pub(crate) fn generate_vectors(fixture: &MirageGolden) -> Vec<Vec<f32>> {
    assert_eq!(fixture.vector_generator, "integer-affine-v1");
    (0..fixture.n)
        .map(|i| {
            (0..fixture.dim)
                .map(|dim| vector_value_integer_affine_v1(i, dim))
                .collect()
        })
        .collect()
}

fn vector_value_integer_affine_v1(i: usize, dim: usize) -> f32 {
    let raw = ((i + 1) * 37 + (dim + 3) * 17 + (i * dim) * 13) % 1009;
    (raw as f32 - 504.0) / 100.0
}

pub(crate) fn l2_score(vectors: &[Vec<f32>], a: usize, b: usize) -> ScoreType {
    -vectors[a]
        .iter()
        .zip(&vectors[b])
        .map(|(left, right)| {
            let diff = left - right;
            diff * diff
        })
        .sum::<ScoreType>()
}

pub(crate) fn scored_top_s_candidates(
    fixture: &MirageGolden,
    vectors: &[Vec<f32>],
    point_id: usize,
) -> Vec<ScoredPointOffset> {
    fixture.layer0_top_s[point_id]
        .iter()
        .copied()
        .map(|idx| ScoredPointOffset {
            idx: idx as PointOffsetType,
            score: l2_score(vectors, point_id, idx),
        })
        .collect()
}

pub(crate) fn compute_layer0_stats(graph: &[Vec<usize>]) -> Layer0Stats {
    let mut stats = Layer0Stats {
        total_edges: 0,
        min_degree: graph.first().map_or(0, Vec::len),
        max_degree: 0,
        avg_degree: 0.0,
        degree_histogram: BTreeMap::new(),
        reciprocal_edges: 0,
        reciprocity_ratio: 0.0,
        weak_connected_components: 0,
        largest_weak_component: 0,
    };

    let mut edges = std::collections::BTreeSet::new();
    for (u, neighbors) in graph.iter().enumerate() {
        let degree = neighbors.len();
        stats.total_edges += degree;
        stats.min_degree = stats.min_degree.min(degree);
        stats.max_degree = stats.max_degree.max(degree);
        *stats.degree_histogram.entry(degree).or_default() += 1;
        for &v in neighbors {
            edges.insert((u, v));
        }
    }

    if !graph.is_empty() {
        stats.avg_degree = stats.total_edges as f64 / graph.len() as f64;
    }

    for &(u, v) in &edges {
        if edges.contains(&(v, u)) {
            stats.reciprocal_edges += 1;
        }
    }
    if stats.total_edges != 0 {
        stats.reciprocity_ratio = stats.reciprocal_edges as f64 / stats.total_edges as f64;
    }

    let mut undirected = vec![Vec::new(); graph.len()];
    for &(u, v) in &edges {
        if v >= graph.len() {
            continue;
        }
        undirected[u].push(v);
        undirected[v].push(u);
    }

    let mut visited = vec![false; graph.len()];
    let mut stack = Vec::new();
    for start in 0..graph.len() {
        if visited[start] {
            continue;
        }

        stats.weak_connected_components += 1;
        let mut component_size = 0;
        stack.clear();
        stack.push(start);
        visited[start] = true;

        while let Some(u) = stack.pop() {
            component_size += 1;
            for &v in &undirected[u] {
                if !visited[v] {
                    visited[v] = true;
                    stack.push(v);
                }
            }
        }

        stats.largest_weak_component = stats.largest_weak_component.max(component_size);
    }

    stats
}

pub(crate) fn assert_stats_match(actual: &Layer0Stats, expected: &Layer0Stats) {
    assert_eq!(actual.total_edges, expected.total_edges);
    assert_eq!(actual.min_degree, expected.min_degree);
    assert_eq!(actual.max_degree, expected.max_degree);
    assert!((actual.avg_degree - expected.avg_degree).abs() < 1e-6);
    assert_eq!(actual.degree_histogram, expected.degree_histogram);
    assert_eq!(actual.reciprocal_edges, expected.reciprocal_edges);
    assert!((actual.reciprocity_ratio - expected.reciprocity_ratio).abs() < 1e-6);
    assert_eq!(
        actual.weak_connected_components,
        expected.weak_connected_components
    );
    assert_eq!(
        actual.largest_weak_component,
        expected.largest_weak_component
    );
}

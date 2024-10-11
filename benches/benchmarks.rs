use std::time::Instant;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn lru_insert_benchmark(c: &mut Criterion) {}

criterion_group!(benches, lru_insert_benchmark,);
criterion_main!(benches);

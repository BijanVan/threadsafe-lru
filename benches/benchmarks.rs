use std::time::Instant;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hashbrown::HashMap;
use rand::{distributions::Uniform, prelude::Distribution};
use threadsafe_lru::LruCache;

fn lru_insert_benchmark(c: &mut Criterion) {
    let shards_count = 4;
    let cap_per_shard = 1000;
    let iterations = 3;
    let mut group_insert = c.benchmark_group("insert");

    let mut rng = rand::thread_rng();
    let between = Uniform::from(0..1_000_000);

    for elem in between.sample_iter(&mut rng).take(iterations) {
        let cache = LruCache::new(shards_count, cap_per_shard);

        group_insert.bench_with_input(
            BenchmarkId::from_parameter(format!("LRU-{}-{:?}", elem, Instant::now())),
            &elem,
            |b, &elem| {
                b.iter(|| cache.insert(elem.to_string(), elem.to_string()));
            },
        );
    }

    for elem in between.sample_iter(&mut rng).take(iterations) {
        let mut map = HashMap::new();

        group_insert.bench_with_input(
            BenchmarkId::from_parameter(format!("HashMap-{}-{:?}", elem, Instant::now())),
            &elem,
            |b, &elem| {
                b.iter(|| map.insert(elem.to_string(), elem.to_string()));
            },
        );
    }

    group_insert.finish();
}

criterion_group!(benches, lru_insert_benchmark,);
criterion_main!(benches);

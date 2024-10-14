use std::time::Instant;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hashbrown::HashMap;
use moka::sync::Cache;
use rand::{distributions::Uniform, prelude::Distribution};
use std::num::NonZeroUsize;
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
            BenchmarkId::from_parameter(format!("threadsafe-lru-{}-{:?}", elem, Instant::now())),
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

fn single_thread_benchmark(c: &mut Criterion) {
    let shards_count = 4;
    let cap_per_shard = 1000;
    let iterations = 2;
    let mut group_insert = c.benchmark_group("insert");

    let mut rng = rand::thread_rng();
    let between = Uniform::from(0..1_000_000);

    for elem in between.sample_iter(&mut rng).take(iterations) {
        let cache = LruCache::new(shards_count, cap_per_shard);

        group_insert.bench_with_input(
            BenchmarkId::from_parameter(format!("threadsafe-lru-{}-{:?}", elem, Instant::now())),
            &elem,
            |b, &elem| {
                b.iter(|| {
                    let key = elem.to_string();
                    let value = key.clone();

                    let op_type = rand::random::<u8>() % 3;
                    match op_type {
                        0 => {
                            cache.insert(key, value);
                        }
                        1 => {
                            cache.get(&key);
                        }
                        2 => {
                            cache.remove(&key);
                        }
                        _ => unreachable!(),
                    }
                });
            },
        );
    }

    for elem in between.sample_iter(&mut rng).take(iterations) {
        let cache = Cache::new((shards_count * cap_per_shard) as u64);

        group_insert.bench_with_input(
            BenchmarkId::from_parameter(format!("lru-{}-{:?}", elem, Instant::now())),
            &elem,
            |b, &elem| {
                b.iter(|| {
                    let key = elem.to_string();
                    let value = key.clone();

                    let op_type = rand::random::<u8>() % 3;
                    match op_type {
                        0 => {
                            cache.insert(key, value);
                        }
                        1 => {
                            cache.get(&key);
                        }
                        2 => {
                            cache.remove(&key);
                        }
                        _ => unreachable!(),
                    }
                });
            },
        );
    }

    group_insert.finish();
}

fn multi_thread_benchmark(_: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;

    let shards_count = 4;
    let cap_per_shard = 1000;

    let cache = Arc::new(LruCache::new(shards_count, cap_per_shard));

    const THREAD_COUNT: usize = 4;
    const OPERATIONS_PER_THREAD: usize = 100_000;

    let mut handles = vec![];

    let start = Instant::now();
    for _ in 0..THREAD_COUNT {
        let cache = Arc::clone(&cache);

        let handle = thread::spawn(move || {
            for _ in 0..OPERATIONS_PER_THREAD {
                let key = rand::random::<i32>();
                let value = rand::random::<i32>();

                let op_type = rand::random::<u8>() % 3;
                match op_type {
                    0 => {
                        cache.insert(key, value);
                    }
                    1 => {
                        cache.get(&key);
                    }
                    2 => {
                        cache.remove(&key);
                    }
                    _ => unreachable!(),
                }
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
    let elapsed = start.elapsed();
    println!("total time : {} ", elapsed.as_millis());
}

fn multi_thread_moka_benchmark(_: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;

    let shards_count = 4;
    let cap_per_shard = 1000;

    let cache = Arc::new(Cache::new(shards_count * cap_per_shard));

    const THREAD_COUNT: usize = 4;
    const OPERATIONS_PER_THREAD: usize = 100_000;

    let mut handles = vec![];

    let start = Instant::now();
    for _ in 0..THREAD_COUNT {
        let cache = Arc::clone(&cache);

        let handle = thread::spawn(move || {
            for _ in 0..OPERATIONS_PER_THREAD {
                let key = rand::random::<i32>();
                let value = rand::random::<i32>();

                let op_type = rand::random::<u8>() % 3;
                match op_type {
                    0 => {
                        cache.insert(key, value);
                    }
                    1 => {
                        cache.get(&key);
                    }
                    2 => {
                        cache.remove(&key);
                    }
                    _ => unreachable!(),
                }
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
    let elapsed = start.elapsed();
    println!("total time : {} ", elapsed.as_millis());
}

criterion_group!(
    benches,
    single_thread_benchmark,
    multi_thread_benchmark,
    multi_thread_moka_benchmark
);
criterion_main!(benches);

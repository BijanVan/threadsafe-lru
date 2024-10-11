use threadsafe_lru::LruCache;

fn main() {
    // Create a new LRU cache with 4 shards and capacity of 2 per shard
    let cache = LruCache::new(4, 2);

    // Insert items into the cache
    let five = 5;
    let six = 6;
    assert_eq!(cache.insert(five, 10), None);
    assert_eq!(cache.insert(six, 20), None);

    // Retrieve an item from the cache
    assert_eq!(cache.get(&five), Some(10));

    // Promote an item to make it more recently used
    cache.promote(&five);

    // Remove an item from the cache
    assert_eq!(cache.remove(&five), Some(10));
}

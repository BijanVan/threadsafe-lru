# threadsafe-lru

This is a thread-safe implementation of an LRU (Least Recently Used) cache in Rust.
The `LruCache` struct uses sharding to improve concurrency by splitting the cache into multiple smaller segments, each protected by a mutex.

## Example Usage

```rust
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
```

In this example, a new `LruCache` is created with 4 shards and a capacity of 2 entries per shard. 
Items are inserted using the `insert` method. 
The `get` method retrieves an item by key, promoting it to the most recently used position. 
Finally, the `remove` method deletes an item from the cache.

This implementation ensures that operations on different keys can be performed concurrently without causing race conditions due to shared state.

## API Documentation

For detailed documentation, including all methods and usage examples, refer to the [LruCache API on docs.rs](https://docs.rs/threadsafe-lru/latest/threadsafe_lru/).

## Testing

LruCache is thoroughly tested with a suite of unit tests covering various operations. You can run the tests using `cargo test`:

```sh
cargo test
```

This ensures that all functionalities work as expected and helps maintain high code quality.

## License

LruCache is licensed under the MIT license. See [LICENSE](LICENSE) for more details.

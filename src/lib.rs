#![deny(unsafe_code)]

/// # threadsafe-lru
///
/// This is a thread-safe implementation of an LRU (Least Recently Used) cache in Rust.
/// The `LruCache` struct uses sharding to improve concurrency by splitting the cache into
/// multiple smaller segments, each protected by a mutex.
///
/// ## Example Usage
///
/// ```rust
/// use threadsafe_lru::LruCache;
///
/// fn main() {
///     // Create a new LRU cache with 4 shards and capacity of 2 per shard
///     let cache = LruCache::new(4, 2);
///
///     // Insert items into the cache
///     let five = 5;
///     let six = 6;
///     assert_eq!(cache.insert(five, 10), None);
///     assert_eq!(cache.insert(six, 20), None);
///
///     // Retrieve an item from the cache
///     assert_eq!(cache.get(&five), Some(10));
///
///     // Promote an item to make it more recently used
///     cache.promote(&five);
///
///     // Remove an item from the cache
///     assert_eq!(cache.remove(&five), Some(10));
/// }
/// ```
///
/// In this example, a new `LruCache` is created with 4 shards and a capacity of 2 entries per shard.
/// Items are inserted using the `insert` method.
/// The `get` method retrieves an item by key, promoting it to the most recently used position.
/// Finally, the `remove` method deletes an item from the cache.
///
/// This implementation ensures that operations on different keys can be performed concurrently
/// without causing race conditions due to shared state.
///
use std::{
    borrow::Borrow,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

use hashbrown::HashMap;
use indexlist::{Index, IndexList};

// A thread-safe LRU
pub struct LruCache<K, V> {
    shards: Vec<Mutex<Shard<K, V>>>,
    count: AtomicUsize,
    shards_count: usize,
    cap_per_shard: usize,
}

struct Shard<K, V> {
    entries: HashMap<K, (Index<K>, V)>,
    order: IndexList<K>,
    count: usize,
}

impl<K, V> LruCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new instance of `LruCache`.
    ///
    /// The cache is divided into multiple shards to improve concurrency by distributing the entries
    /// across different locks.
    ///
    /// # Arguments
    ///
    /// * `shards_count` - The number of shards in the cache. Each shard acts as an independent LRU
    ///                    with its own capacity and order list.
    /// * `cap_per_shard` - The capacity for each shard, representing the maximum number of items it
    ///                     can hold before evicting the least recently used item(s).
    ///
    /// # Returns
    ///
    /// A new instance of `LruCache`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use threadsafe_lru::LruCache;
    ///
    /// let cache: LruCache<i32, i32> = LruCache::new(4, 2); // Creates a cache with 4 shards and capacity of 2 entries per shard.
    /// ```
    pub fn new(shards_count: usize, cap_per_shard: usize) -> LruCache<K, V> {
        let mut shards = Vec::default();
        for _ in 0..shards_count {
            shards.push(Mutex::new(Shard {
                entries: HashMap::with_capacity(cap_per_shard),
                order: IndexList::with_capacity(cap_per_shard),
                count: 0,
            }));
        }
        LruCache {
            shards,
            count: AtomicUsize::default(),
            shards_count,
            cap_per_shard,
        }
    }

    /// Inserts a new key-value pair into the cache.
    ///
    /// If the key already exists in the cache, its value is updated and it is promoted to the most
    /// recently used position.
    /// If inserting the new item causes the cache to exceed its capacity, the least recently used
    /// item will be evicted from its shard.
    ///
    /// # Arguments
    ///
    /// * `k` - The key to insert or update.
    /// * `v` - The value associated with the key.
    ///
    /// # Returns
    ///
    /// If the key already existed in the cache and was updated, the previous value is returned.
    /// Otherwise, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use threadsafe_lru::LruCache;
    ///
    /// let cache = LruCache::new(4, 2);
    ///
    /// let five = 5;
    /// let six = 6;
    ///
    /// assert_eq!(cache.insert(five, 10), None); // Inserts a new key-value pair
    /// assert_eq!(cache.insert(six, 20), None); // Inserts another new key-value pair
    /// assert_eq!(cache.insert(five, 30), Some(10)); // Updates an existing key with a new value and returns the old value
    /// ```
    pub fn insert(&self, k: K, v: V) -> Option<V> {
        let mut shard = self.shards[self.shard(&k)].lock().unwrap();
        let index = shard.entries.get(&k).map(|v| v.0);
        if shard.count == self.cap_per_shard && index.is_none() {
            if let Some(index) = shard.order.head_index() {
                shard.entries.remove(&k);
                shard.order.remove(index);
                self.count.fetch_sub(1, Ordering::Relaxed);
                shard.count -= 1;
            }
        }

        match index {
            Some(index) => {
                shard.order.remove(index);
                let index = shard.order.push_back(k.clone());
                shard.entries.insert(k, (index, v)).map(|v| v.1)
            }
            None => {
                let index = shard.order.push_back(k.clone());
                shard.entries.insert(k, (index, v));
                self.count.fetch_add(1, Ordering::Relaxed);
                shard.count += 1;
                None
            }
        }
    }

    /// Retrieves a value from the cache by key.
    ///
    /// When a key is accessed using this method, it is promoted to the most recently used position
    /// in its shard. If the key does not exist in the cache, `None` is returned.
    ///
    /// # Arguments
    ///
    /// * `k` - The key of the item to retrieve.
    ///
    /// # Returns
    ///
    /// The value associated with the key if it exists; otherwise, `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use threadsafe_lru::LruCache;
    ///
    /// let cache = LruCache::new(4, 2);
    ///
    /// let five = 5;
    /// let six = 6;
    ///
    /// assert_eq!(cache.insert(five, 10), None);
    /// assert_eq!(cache.get(&five), Some(10));
    /// ```
    ///
    /// This method ensures that frequently accessed items remain more accessible while older or less
    /// frequently accessed items are evicted when necessary.

    pub fn get<Q>(&self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized + ToOwned<Owned = K>,
        V: Clone,
    {
        let mut shard = self.shards[self.shard(k)].lock().unwrap();
        let index = shard.entries.get(k).map(|e| e.0);
        match index {
            Some(index) => {
                shard.order.remove(index);
                let index = shard.order.push_back(k.to_owned());
                let entry = shard.entries.remove(k);
                match entry {
                    Some(e) => {
                        shard.entries.insert(k.to_owned(), (index, e.1.clone()));
                        Some(e.1)
                    }
                    None => None,
                }
            }
            None => None,
        }
    }

    /// Retrieves and mutates a value from the cache by key.
    ///
    /// When a key is accessed using this method, it is promoted to the most recently used position
    /// in its shard. If the key does not exist in the cache, `None` is returned.
    ///
    /// This function provides mutable access to the value associated with a given key, allowing you
    /// to modify its contents directly without needing to retrieve and re-insert it into the cache.
    ///
    /// # Arguments
    ///
    /// * `k` - The key of the item to retrieve and mutate.
    /// * `func` - A closure that takes a mutable reference to the value (if present) and allows
    ///            for in-place modifications.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use threadsafe_lru::LruCache;
    ///
    /// let cache = LruCache::new(4, 2);
    ///
    /// let five = 5;
    /// let six = 6;
    ///
    /// assert_eq!(cache.insert(five, 10), None);
    /// cache.get_mut(&five, |v| {
    ///   if let Some(v) = v {
    ///      *v += 1
    ///    }
    /// });
    /// ```
    ///
    /// This method ensures that frequently accessed items remain more accessible while older or less
    /// frequently accessed items are evicted when necessary.
    pub fn get_mut<Q, F>(&self, k: &Q, mut func: F)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized + ToOwned<Owned = K>,
        F: FnMut(Option<&mut V>),
    {
        let mut shard = self.shards[self.shard(k)].lock().unwrap();
        let index = shard.entries.get(k).map(|e| e.0);
        if let Some(index) = index {
            shard.order.remove(index);
            let index = shard.order.push_back(k.to_owned());
            let entry = shard.entries.remove(k);
            if let Some(e) = entry {
                shard.entries.insert(k.to_owned(), (index, e.1));
            }
            func(shard.entries.get_mut(k).map(|e| &mut e.1));
        }
    }

    /// Removes a key-value pair from the cache by key.
    ///
    /// When an item is removed from the cache using this method, its corresponding entry is deleted
    /// along with any references to it in the order list. If the key does not exist in the cache,
    /// `None` is returned.
    ///
    /// This operation ensures that the cache maintains its integrity and correctly tracks the number of items stored.
    ///
    /// # Arguments
    ///
    /// * `k` - The key of the item to remove.
    ///
    /// # Returns
    ///
    /// The value associated with the key if it existed; otherwise, `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use threadsafe_lru::LruCache;
    ///
    /// let cache = LruCache::new(4, 2);
    ///
    /// let five = 5;
    /// let six = 6;
    ///
    /// assert_eq!(cache.insert(five, 10), None); // Inserts a new key-value pair
    /// assert_eq!(cache.insert(six, 20), None); // Inserts another new key-value pair
    ///
    /// assert_eq!(cache.remove(&five), Some(10)); // Removes the item with key five and returns its value
    /// assert_eq!(cache.get(&five), None); // Key five no longer exists in the cache
    /// ```
    ///
    pub fn remove<Q>(&self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let mut shard = self.shards[self.shard(k)].lock().unwrap();
        let entry = shard.entries.remove(k);
        match entry {
            Some((index, value)) => {
                shard.order.remove(index);
                self.count.fetch_sub(1, Ordering::Relaxed);
                shard.count -= 1;
                Some(value)
            }
            None => None,
        }
    }

    /// Promotes a key-value pair in the cache to the most recently used position.
    ///
    /// When an item is accessed using this method, it is promoted to the most recently used position
    /// in its shard. If the key does not exist in the cache, no action is taken.
    ///
    /// # Arguments
    ///
    /// * `k` - The key of the item to promote.
    ///
    ///
    /// # Examples
    ///
    /// ```rust
    /// use threadsafe_lru::LruCache;
    ///
    /// let cache = LruCache::new(4, 2);
    ///
    /// let five = 5;
    /// let six = 6;
    /// let seven = 7;
    ///
    /// assert_eq!(cache.insert(five, 10), None); // Inserts a new key-value pair
    /// assert_eq!(cache.insert(six, 20), None); // Inserts another new key-value pair
    ///
    /// cache.promote(&five);
    ///
    /// assert_eq!(cache.insert(seven, 30), None); // Inserts another new key-value pair
    /// assert_eq!(cache.get(&five), Some(10)); // Retrieving the promoted item
    /// ```
    ///
    pub fn promote<Q>(&self, k: &Q)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized + ToOwned<Owned = K>,
    {
        let mut shard = self.shards[self.shard(k)].lock().unwrap();
        let entry = shard.entries.remove(k);
        if let Some(entry) = entry {
            shard.order.remove(entry.0);
            let index = shard.order.push_back(k.to_owned());
            shard.entries.insert(k.to_owned(), (index, entry.1));
        }
    }

    /// Returns the total number of key-value pairs currently stored in the cache.
    ///
    /// This method provides a quick way to check how many items are present in the cache without
    /// iterating over its contents.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use threadsafe_lru::LruCache;
    ///
    /// let cache = LruCache::new(4, 2);
    ///
    /// let five = 5;
    /// let six = 6;
    ///
    /// assert_eq!(cache.insert(five, 10), None); // Inserts a new key-value pair
    /// assert_eq!(cache.len(), 1); // Cache now has one item
    ///
    /// cache.insert(six, 20);
    /// assert_eq!(cache.len(), 2); // Cache now has two items
    ///
    /// cache.remove(&five);
    /// assert_eq!(cache.len(), 1); // Cache now has only one item
    ///
    /// let new_cache: LruCache<i32, i32> = LruCache::new(4, 2);
    /// assert_eq!(new_cache.len(), 0); // New cache is empty
    /// ```
    ///
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Checks if the cache is empty.
    ///
    /// This method provides a quick way to determine whether the cache contains any key-value pairs
    /// without iterating over its contents.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use threadsafe_lru::LruCache;
    ///
    /// let cache = LruCache::new(4, 2);
    ///
    /// assert!(cache.is_empty()); // Cache is empty upon creation
    ///
    /// let five = 5;
    /// let six = 6;
    ///
    /// cache.insert(five, 10); // Inserting a key-value pair
    /// assert!(!cache.is_empty()); // Cache is no longer empty
    ///
    /// cache.remove(&five); // Removing the key-value pair
    /// assert!(cache.is_empty()); // Cache is empty again after removal
    /// ```
    ///
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn shard<Q>(&self, k: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let mut hasher = DefaultHasher::new();
        k.hash(&mut hasher);
        hasher.finish() as usize % self.shards_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let shards_count = 4;
        let cap_per_shard = 10;

        let cache: LruCache<u8, u8> = LruCache::new(shards_count, cap_per_shard);
        assert_eq!(cache.shards.len(), shards_count);

        for shard in &cache.shards {
            let lock = shard.lock().unwrap();
            assert!(lock.entries.capacity() >= cap_per_shard);
            assert_eq!(lock.count, 0);
        }

        assert_eq!(cache.shards_count, shards_count);
        assert_eq!(cache.cap_per_shard, cap_per_shard);
    }

    #[test]
    fn test_insert() {
        let shards_count = 4;
        let cap_per_shard = 2;

        let cache = LruCache::new(shards_count, cap_per_shard);

        let five = 5;
        let six = 6;
        let nine = 9;
        assert_eq!(cache.shard(&five), cache.shard(&six));
        assert_eq!(cache.shard(&five), cache.shard(&nine));

        assert_eq!(cache.insert(five, 10), None);
        assert_eq!(cache.insert(five, 10), Some(10));
        assert_eq!(cache.count.load(Ordering::Relaxed), 1);
        assert_eq!(cache.insert(six, 20), None);
        assert_eq!(cache.count.load(Ordering::Relaxed), 2);
        assert_eq!(cache.insert(nine, 30), None);
        assert_eq!(cache.count.load(Ordering::Relaxed), 2);
        assert_eq!(cache.insert(six, 20), Some(20));
        assert_eq!(cache.count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_get() {
        let shards_count = 4;
        let cap_per_shard = 2;

        let cache = LruCache::new(shards_count, cap_per_shard);

        let five = 5;
        let six = 6;
        assert_eq!(cache.shard(&five), cache.shard(&six));
        assert_eq!(cache.insert(five, 10), None);
        assert_eq!(cache.insert(six, 20), None);

        assert_eq!(cache.get(&five), Some(10));
        assert_eq!(cache.get(&six), Some(20));

        let shard = cache.shards[cache.shard(&five)].lock().unwrap();
        assert_eq!(shard.order.head(), Some(&five));
        assert_eq!(shard.order.tail(), Some(&six));

        assert_eq!(cache.get(&3), None);

        assert_eq!(cache.count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_get_mut() {
        let shards_count = 4;
        let cap_per_shard = 2;

        let cache = LruCache::new(shards_count, cap_per_shard);

        let five = 5;
        let six = 6;
        assert_eq!(cache.shard(&five), cache.shard(&six));
        assert_eq!(cache.insert(five, 10), None);
        assert_eq!(cache.insert(six, 20), None);

        cache.get_mut(&five, |v| {
            if let Some(v) = v {
                *v = 30
            }
        });
        assert_eq!(cache.get(&five), Some(30));
        assert_eq!(cache.count.load(Ordering::Relaxed), 2);

        let shard = cache.shards[cache.shard(&five)].lock().unwrap();
        assert_eq!(shard.order.tail(), Some(&five));
        assert_eq!(shard.order.head(), Some(&six));

        cache.get_mut(&3, |v| {
            if let Some(v) = v {
                *v = 10
            }
        });
        assert_eq!(cache.count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_remove() {
        let shards_count = 4;
        let cap_per_shard = 2;

        let cache = LruCache::new(shards_count, cap_per_shard);

        let five = 5;
        let six = 6;
        assert_eq!(cache.shard(&five), cache.shard(&six));
        assert_eq!(cache.insert(five, 10), None);
        assert_eq!(cache.insert(six, 20), None);

        assert_eq!(cache.remove(&five), Some(10));
        assert_eq!(cache.count.load(Ordering::Relaxed), 1);
        let shard = cache.shards[cache.shard(&six)].lock().unwrap();
        assert!(!shard.entries.contains_key(&five));
        assert_eq!(shard.order.head(), Some(&six));
        drop(shard);

        assert_eq!(cache.remove(&5), None);
        assert_eq!(cache.count.load(Ordering::Relaxed), 1);

        let new_cache: LruCache<i32, i32> = LruCache::new(shards_count, cap_per_shard);
        assert_eq!(new_cache.remove(&five), None);
        assert_eq!(new_cache.count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_promote() {
        let shards_count = 4;
        let cap_per_shard = 2;

        let cache = LruCache::new(shards_count, cap_per_shard);

        let five = 5;
        let six = 6;
        assert_eq!(cache.shard(&five), cache.shard(&six));
        assert_eq!(cache.insert(five, 10), None);
        assert_eq!(cache.insert(six, 20), None);

        let shard = cache.shards[cache.shard(&five)].lock().unwrap();
        assert_eq!(shard.order.head(), Some(&five));
        assert_eq!(shard.order.tail(), Some(&six));
        drop(shard);

        cache.promote(&five);
        let shard = cache.shards[cache.shard(&five)].lock().unwrap();
        assert_eq!(shard.order.head(), Some(&six));
        assert_eq!(shard.order.tail(), Some(&five));
        drop(shard);

        assert_eq!(cache.get(&five), Some(10));
        let shard = cache.shards[cache.shard(&five)].lock().unwrap();
        assert_eq!(shard.order.head(), Some(&six));
        assert_eq!(shard.order.tail(), Some(&five));
        drop(shard);
    }

    #[test]
    fn test_is_empty() {
        let shards_count = 4;
        let cap_per_shard = 2;

        let cache = LruCache::new(shards_count, cap_per_shard);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        let five = 5;
        assert_eq!(cache.insert(five, 10), None);
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);

        cache.remove(&five);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_len() {
        let shards_count = 4;
        let cap_per_shard = 2;

        let cache = LruCache::new(shards_count, cap_per_shard);

        let five = 5;
        let six = 6;
        assert_eq!(cache.insert(five, 10), None);
        assert_eq!(cache.len(), 1);

        cache.insert(six, 20);
        assert_eq!(cache.len(), 2);

        cache.remove(&five);
        assert_eq!(cache.len(), 1);

        let new_cache: LruCache<i32, i32> = LruCache::new(shards_count, cap_per_shard);
        assert_eq!(new_cache.len(), 0);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let shards_count = 4;
        let cap_per_shard = 2;

        let cache: Arc<LruCache<i32, i32>> = Arc::new(LruCache::new(shards_count, cap_per_shard));

        const THREAD_COUNT: usize = 10;
        const OPERATIONS_PER_THREAD: usize = 100;

        let mut handles = vec![];

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
    }
}

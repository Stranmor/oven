use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

#[derive(Debug, thiserror::Error)]
enum CacheStorageError {
    #[error("Failed to serialize cache key")]
    KeySerialization { source: serde_json::Error },
    #[error("Failed to read from cache")]
    Read { source: cacache::Error },
    #[error("Failed to serialize entry for caching")]
    EntrySerialization { source: serde_json::Error },
    #[error("Failed to write to cache")]
    Write { source: cacache::Error },
    #[error("Failed to clear cache directory {path}")]
    ClearDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("Failed to recreate cache directory {path}")]
    RecreateDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
}

/// Wrapper for cached values with timestamp for TTL validation
#[derive(Serialize, Deserialize)]
struct CachedEntry<V> {
    value: V,
    timestamp: u128,
}

/// Generic content-addressable key-value storage using cacache.
///
/// This storage provides a type-safe wrapper around cacache for arbitrary
/// key-value caching with content verification. Keys are serialized to
/// deterministic strings using hash values, and values are stored as JSON
/// using serde_json for maximum compatibility.
pub struct CacacheStorage {
    cache_dir: PathBuf,
    ttl: Option<std::time::Duration>,
}

impl CacacheStorage {
    /// Creates a new key-value storage with the specified cache directory.
    ///
    /// The directory will be created if it doesn't exist. All cache data
    /// will be stored under this directory using cacache's content-addressable
    /// storage format.
    ///
    /// # Arguments
    /// * `cache_dir` - Directory where cache data will be stored
    /// * `ttl` - Optional TTL duration. If provided, entries older than this
    ///   will be considered expired.
    pub fn new(cache_dir: PathBuf, ttl: Option<std::time::Duration>) -> Self {
        Self { cache_dir, ttl }
    }

    /// Converts a key to a deterministic cache key string using a stable hash.
    fn key_to_string<K>(&self, key: &K) -> Result<String>
    where
        K: Serialize,
    {
        let value = serde_json::to_value(key)
            .map_err(|source| CacheStorageError::KeySerialization { source })?;
        let canonical = Self::canonicalize_json(value);
        let serialized = serde_json::to_vec(&canonical)
            .map_err(|source| CacheStorageError::KeySerialization { source })?;
        let digest = Sha256::digest(serialized);
        Ok(hex::encode(digest))
    }

    fn canonicalize_json(value: Value) -> Value {
        match value {
            Value::Array(values) => {
                Value::Array(values.into_iter().map(Self::canonicalize_json).collect())
            }
            Value::Object(map) => {
                let sorted = map
                    .into_iter()
                    .map(|(key, value)| (key, Self::canonicalize_json(value)))
                    .collect::<BTreeMap<_, _>>();
                Value::Object(Map::from_iter(sorted))
            }
            value => value,
        }
    }

    /// Gets the current Unix timestamp in seconds.
    fn get_current_timestamp() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System time before UNIX epoch")
            .as_secs() as u128
    }

    /// Checks if a cached entry has expired based on TTL.
    fn is_expired(&self, timestamp: u128) -> bool {
        if let Some(ttl) = self.ttl {
            let current = Self::get_current_timestamp();
            current.saturating_sub(timestamp) > (ttl.as_secs() as u128)
        } else {
            false
        }
    }
}

#[async_trait::async_trait]
impl forge_app::KVStore for CacacheStorage {
    async fn cache_get<K, V>(&self, key: &K) -> Result<Option<V>>
    where
        K: Serialize + Sync,
        V: serde::Serialize + DeserializeOwned + Send,
    {
        let key_str = self.key_to_string(key)?;

        match cacache::read(&self.cache_dir, &key_str).await {
            Ok(data) => match serde_json::from_slice::<CachedEntry<V>>(&data) {
                Ok(entry) => {
                    if self.is_expired(entry.timestamp) {
                        Ok(None)
                    } else {
                        Ok(Some(entry.value))
                    }
                }
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        cache_key = %key_str,
                        "Discarding invalid cache entry"
                    );
                    if let Err(remove_error) = cacache::remove(&self.cache_dir, &key_str).await {
                        tracing::warn!(
                            error = %remove_error,
                            cache_key = %key_str,
                            "Failed to remove invalid cache entry"
                        );
                    }
                    Ok(None)
                }
            },
            Err(error) => {
                if matches!(error, cacache::Error::EntryNotFound(_, _)) {
                    Ok(None)
                } else {
                    Err(CacheStorageError::Read { source: error }.into())
                }
            }
        }
    }

    async fn cache_set<K, V>(&self, key: &K, value: &V) -> Result<()>
    where
        K: Serialize + Sync,
        V: serde::Serialize + Sync,
    {
        let key_str = self.key_to_string(key)?;

        let entry = CachedEntry { value, timestamp: Self::get_current_timestamp() };

        let data = serde_json::to_vec(&entry)
            .map_err(|source| CacheStorageError::EntrySerialization { source })?;

        cacache::write(&self.cache_dir, &key_str, data)
            .await
            .map_err(|source| CacheStorageError::Write { source })?;

        Ok(())
    }

    async fn cache_clear(&self) -> Result<()> {
        if !self.cache_dir.exists() {
            return Ok(());
        }
        // Use remove_dir_all + create_dir_all instead of cacache::clear because
        // cacache::clear calls remove_dir_all on every directory entry, which
        // fails with ENOTDIR when regular files (e.g. .mcp.json) are present.
        tokio::fs::remove_dir_all(&self.cache_dir)
            .await
            .map_err(|source| CacheStorageError::ClearDirectory {
                path: self.cache_dir.clone(),
                source,
            })?;
        tokio::fs::create_dir_all(&self.cache_dir)
            .await
            .map_err(|source| CacheStorageError::RecreateDirectory {
                path: self.cache_dir.clone(),
                source,
            })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use forge_app::KVStore;
    use pretty_assertions::assert_eq;
    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct TestKey {
        id: String,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestValue {
        data: String,
        count: i32,
    }

    fn test_cache_dir() -> PathBuf {
        tempfile::tempdir().unwrap().keep()
    }

    #[tokio::test]
    async fn test_get_nonexistent_key() {
        let cache_dir = test_cache_dir();
        let cache = CacacheStorage::new(cache_dir, None);

        let key = TestKey { id: "test".to_string() };
        let result: Option<TestValue> = cache.cache_get(&key).await.unwrap();

        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_set_and_get() {
        let cache_dir = test_cache_dir();
        let cache = CacacheStorage::new(cache_dir, None);

        let key = TestKey { id: "test".to_string() };
        let value = TestValue { data: "hello".to_string(), count: 42 };

        cache.cache_set(&key, &value).await.unwrap();
        let result: Option<TestValue> = cache.cache_get(&key).await.unwrap();

        assert_eq!(result, Some(value));
    }

    #[tokio::test]
    async fn test_clear() {
        let cache_dir = test_cache_dir();
        let cache = CacacheStorage::new(cache_dir, None);

        let key1 = TestKey { id: "test1".to_string() };
        let key2 = TestKey { id: "test2".to_string() };
        let value = TestValue { data: "hello".to_string(), count: 42 };

        cache.cache_set(&key1, &value).await.unwrap();
        cache.cache_set(&key2, &value).await.unwrap();

        cache.cache_clear().await.unwrap();

        let result1: Option<TestValue> = cache.cache_get(&key1).await.unwrap();
        let result2: Option<TestValue> = cache.cache_get(&key2).await.unwrap();

        assert_eq!(result1, None);
        assert_eq!(result2, None);
    }

    #[tokio::test]
    async fn test_clear_removes_regular_files_inside_cache_dir() {
        let cache_dir = test_cache_dir();
        let cache = CacacheStorage::new(cache_dir.clone(), None);

        std::fs::write(cache_dir.join(".mcp.json"), "{}").unwrap();
        let actual = cache.cache_clear().await;

        assert!(actual.is_ok());
        assert!(cache_dir.exists());
        assert!(!cache_dir.join(".mcp.json").exists());
    }

    #[tokio::test]
    async fn test_ttl_not_expired() {
        let cache_dir = test_cache_dir();
        let cache = CacacheStorage::new(cache_dir, Some(std::time::Duration::from_secs(60))); // 60 seconds TTL

        let key = TestKey { id: "test".to_string() };
        let value = TestValue { data: "hello".to_string(), count: 42 };

        cache.cache_set(&key, &value).await.unwrap();

        // Immediately retrieve - should not be expired
        let result: Option<TestValue> = cache.cache_get(&key).await.unwrap();

        assert_eq!(result, Some(value));
    }

    #[tokio::test]
    async fn test_ttl_expired() {
        let cache_dir = test_cache_dir();
        let cache = CacacheStorage::new(cache_dir, Some(std::time::Duration::from_secs(1))); // 1 second TTL

        let key = TestKey { id: "test".to_string() };
        let value = TestValue { data: "hello".to_string(), count: 42 };

        cache.cache_set(&key, &value).await.unwrap();

        // Wait for TTL to expire
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        let result: Option<TestValue> = cache.cache_get(&key).await.unwrap();

        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_ttl_none_never_expires() {
        let cache_dir = test_cache_dir();
        let cache = CacacheStorage::new(cache_dir, None); // No TTL

        let key = TestKey { id: "test".to_string() };
        let value = TestValue { data: "hello".to_string(), count: 42 };

        cache.cache_set(&key, &value).await.unwrap();

        // Wait a bit
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        // Should still be available
        let result: Option<TestValue> = cache.cache_get(&key).await.unwrap();

        assert_eq!(result, Some(value));
    }

    #[tokio::test]
    async fn test_should_not_fail_when_no_cache_dir_present() {
        let cache_dir = PathBuf::from("/tmp/forge_test_nonexistent_cache_dir_that_does_not_exist");
        let cache = CacacheStorage::new(cache_dir, None);
        let actual = cache.cache_clear().await;
        assert!(actual.is_ok());
    }
}

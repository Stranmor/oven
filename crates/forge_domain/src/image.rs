use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use derive_getters::Getters;
use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Debug, Serialize, Deserialize, Getters, PartialEq, Eq, Hash)]
pub struct Image {
    url: String,
    mime_type: String,
}

impl Image {
    pub fn new_bytes(content: Vec<u8>, mime_type: impl ToString) -> Self {
        let mime_type = mime_type.to_string();
        let base64_encoded = base64::engine::general_purpose::STANDARD.encode(&content);
        Self::new_base64(base64_encoded, mime_type)
    }

    // Returns the base64 image without the prefix.
    pub fn data(&self) -> &str {
        self.url
            .strip_prefix(&format!("data:{};base64,", self.mime_type))
            .unwrap_or(&self.url)
    }

    /// Returns a provider-safe canonical data URL for a raw data URL string.
    ///
    /// # Errors
    ///
    /// Returns an error when the URL is not a base64 data URI, uses an
    /// unsupported MIME type, or contains invalid base64 payload data.
    pub fn canonicalize_data_url(raw: &str) -> anyhow::Result<String> {
        let trimmed = raw.trim();
        if trimmed.len() < 5 || !trimmed[..5].eq_ignore_ascii_case("data:") {
            anyhow::bail!("Image URL must be a data URI")
        }
        let rest = &trimmed[5..];
        let Some((meta, data)) = rest.split_once(',') else {
            anyhow::bail!("Image data URI is missing comma separator")
        };
        let mut segments = meta.split(';');
        let Some(mime_type) = segments.next().filter(|value| !value.is_empty()) else {
            anyhow::bail!("Image data URI is missing MIME type")
        };
        if !matches!(
            mime_type,
            "image/jpeg" | "image/png" | "image/gif" | "image/webp"
        ) {
            anyhow::bail!("Unsupported image MIME type: {mime_type}")
        }
        if !segments.any(|segment| segment.eq_ignore_ascii_case("base64")) {
            anyhow::bail!("Image data URI must be base64 encoded")
        }
        let normalized = data
            .chars()
            .filter(|c| !c.is_ascii_whitespace())
            .collect::<String>();
        STANDARD
            .decode(&normalized)
            .map_err(|e| anyhow::anyhow!("Invalid image base64 payload: {e}"))?;
        Ok(format!("data:{mime_type};base64,{normalized}"))
    }

    /// Returns a provider-safe canonical data URL.
    ///
    /// # Errors
    ///
    /// Returns an error when the image URL is not a base64 data URI, uses an
    /// unsupported MIME type, or contains invalid base64 payload data.
    pub fn canonical_data_url(&self) -> anyhow::Result<String> {
        Self::canonicalize_data_url(&self.url)
    }

    pub fn new_base64(base64_encoded: String, mime_type: impl ToString) -> Self {
        let mime_type = mime_type.to_string();
        let content = format!("data:{mime_type};base64,{base64_encoded}");
        Self { url: content, mime_type }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::Image;

    #[test]
    fn canonicalize_data_url_strips_whitespace() {
        let actual = Image::canonicalize_data_url("data:image/png;base64,iVBO\nRw0KGgo=");
        let expected = Ok("data:image/png;base64,iVBORw0KGgo=".to_string());
        assert_eq!(actual.map_err(|e| e.to_string()), expected);
    }

    #[test]
    fn canonicalize_data_url_rejects_unsupported_mime() {
        let actual = Image::canonicalize_data_url("data:image/bmp;base64,AAAA");
        let expected = "Unsupported image MIME type: image/bmp".to_string();
        assert_eq!(actual.unwrap_err().to_string(), expected);
    }

    #[test]
    fn canonicalize_data_url_rejects_invalid_base64() {
        let actual = Image::canonicalize_data_url("data:image/png;base64,not valid base64");
        assert!(actual.is_err());
    }

    #[test]
    fn canonicalize_data_url_accepts_uppercase_data_scheme() {
        let actual = Image::canonicalize_data_url("DATA:image/png;base64,iVBORw0KGgo=");
        let expected = Ok("data:image/png;base64,iVBORw0KGgo=".to_string());
        assert_eq!(actual.map_err(|e| e.to_string()), expected);
    }
}

use serde::{Deserialize, Serialize};
use url::Url;

/// Configuration for the Festoon webhook.
///
/// When `url` is set, the gateway POSTs the full request/response bodies
/// (as JSON) to this endpoint after every proxied AI request.
#[derive(Debug, Default, Clone, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(default, rename_all = "kebab-case")]
pub struct WebhookConfig {
    /// The URL to POST captured interactions to (e.g. http://localhost:8000/api/gateway/ingest).
    /// If None, webhooks are disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<Url>,
}

impl WebhookConfig {
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.url.is_some()
    }
}

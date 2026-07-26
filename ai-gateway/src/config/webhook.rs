use std::fmt;

use serde::{Deserialize, Serialize};
use url::Url;

/// Configuration for the Festoon webhook.
///
/// When `url` is set, the gateway POSTs the full request/response bodies
/// (as JSON) to this endpoint after every proxied AI request.
///
/// When `secret` is set, each POST carries it as an `x-ingest-secret`
/// header. The Festoon API rejects unauthenticated ingest whenever its
/// own `FESTOON_GATEWAY_INGEST_SECRET` is configured, so the two must
/// match for capture to work.
#[derive(Default, Clone, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(default, rename_all = "kebab-case")]
pub struct WebhookConfig {
    /// The URL to POST captured interactions to (e.g. http://localhost:8000/api/gateway/ingest).
    /// If None, webhooks are disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<Url>,

    /// Shared secret sent as the `x-ingest-secret` header. If None, the
    /// POST is unauthenticated — only valid against an API that has no
    /// ingest secret configured.
    ///
    /// Never serialized: config dumps must not leak it.
    #[serde(skip_serializing)]
    pub secret: Option<String>,
}

impl WebhookConfig {
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.url.is_some()
    }
}

/// Manual `Debug` so the secret is redacted in logs and traces.
impl fmt::Debug for WebhookConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WebhookConfig")
            .field("url", &self.url)
            .field("secret", &self.secret.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

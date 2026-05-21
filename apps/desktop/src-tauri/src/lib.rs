use serde::{Deserialize, Serialize};

#[cfg(windows)]
const TAVILY_CREDENTIAL_TARGET: &str = "Heliopause.Tavily.ApiToken";
const TAVILY_SEARCH_URL: &str = "https://api.tavily.com/search";
const TAVILY_MAX_RESULTS: u8 = 5;

#[derive(Debug, Serialize)]
struct CommandError {
    code: &'static str,
    message: String,
    retryable: Option<bool>,
}

impl CommandError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            retryable: None,
        }
    }

    fn retryable(code: &'static str, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            code,
            message: message.into(),
            retryable: Some(retryable),
        }
    }
}

#[derive(serde::Serialize)]
struct SystemMemoryInfo {
    total_bytes: u64,
    available_bytes: u64,
}

#[derive(Serialize)]
struct TavilyTokenStatus {
    available: bool,
    configured: bool,
    reason: Option<String>,
}

#[derive(Deserialize)]
struct WebSearchRequest {
    query: String,
    #[serde(default, alias = "maxResults")]
    max_results: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
struct WebSearchResult {
    title: String,
    url: String,
    snippet: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
struct WebSearchResponse {
    results: Vec<WebSearchResult>,
}

#[tauri::command]
fn system_memory_info() -> SystemMemoryInfo {
    system_memory_info_impl()
}

#[tauri::command]
fn save_tavily_token(token: String) -> Result<(), CommandError> {
    let token = token.trim();
    if token.is_empty() {
        return Err(CommandError::new(
            "invalid_tavily_token",
            "Tavily token must not be empty.",
        ));
    }
    save_tavily_token_impl(token)
}

#[tauri::command]
fn tavily_token_status() -> TavilyTokenStatus {
    tavily_token_status_impl()
}

#[tauri::command]
fn delete_tavily_token() -> Result<(), CommandError> {
    delete_tavily_token_impl()
}

#[tauri::command]
async fn web_search(request: WebSearchRequest) -> Result<WebSearchResponse, CommandError> {
    let request = normalize_web_search_request(request)?;
    let Some(token) = read_tavily_token_impl()? else {
        return Err(CommandError::new(
            "tavily_token_missing",
            "No Tavily token is configured.",
        ));
    };

    let response = reqwest::Client::new()
        .post(TAVILY_SEARCH_URL)
        .bearer_auth(token)
        .json(&serde_json::json!({
            "query": request.query,
            "search_depth": "basic",
            "max_results": request.max_results,
            "include_answer": false,
            "include_raw_content": false,
            "include_images": false,
            "auto_parameters": false,
        }))
        .send()
        .await
        .map_err(|error| CommandError::retryable("tavily_error", error.to_string(), true))?;

    let status = response.status();
    if !status.is_success() {
        let retryable = status.as_u16() == 429 || status.is_server_error();
        return Err(CommandError::retryable(
            "tavily_error",
            format!("Tavily search failed with HTTP status {}.", status.as_u16()),
            retryable,
        ));
    }

    let value = response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| CommandError::retryable("tavily_error", error.to_string(), true))?;
    parse_tavily_search_response(value)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            system_memory_info,
            save_tavily_token,
            tavily_token_status,
            delete_tavily_token,
            web_search,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn normalize_web_search_request(
    request: WebSearchRequest,
) -> Result<WebSearchRequest, CommandError> {
    let query = request.query.trim().to_string();
    if query.is_empty() {
        return Err(CommandError::new(
            "invalid_web_search_request",
            "Search query must not be empty.",
        ));
    }
    if query.chars().count() > 500 {
        return Err(CommandError::new(
            "invalid_web_search_request",
            "Search query must be 500 characters or fewer.",
        ));
    }
    let max_results = request.max_results.unwrap_or(TAVILY_MAX_RESULTS);
    if max_results == 0 || max_results > TAVILY_MAX_RESULTS {
        return Err(CommandError::new(
            "invalid_web_search_request",
            "max_results must be between 1 and 5.",
        ));
    }
    Ok(WebSearchRequest {
        query,
        max_results: Some(max_results),
    })
}

fn parse_tavily_search_response(
    value: serde_json::Value,
) -> Result<WebSearchResponse, CommandError> {
    let results = value
        .get("results")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| CommandError::new("tavily_error", "Tavily response is missing results."))?;

    Ok(WebSearchResponse {
        results: results
            .iter()
            .filter_map(|item| {
                let title = item.get("title")?.as_str()?.trim();
                let url = item.get("url")?.as_str()?.trim();
                let snippet = item.get("content")?.as_str()?.trim();
                if title.is_empty() || url.is_empty() {
                    return None;
                }
                Some(WebSearchResult {
                    title: title.to_string(),
                    url: url.to_string(),
                    snippet: snippet.to_string(),
                })
            })
            .take(TAVILY_MAX_RESULTS as usize)
            .collect(),
    })
}

#[cfg(target_os = "linux")]
fn system_memory_info_impl() -> SystemMemoryInfo {
    let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") else {
        return SystemMemoryInfo {
            total_bytes: 0,
            available_bytes: 0,
        };
    };
    let total_bytes = meminfo_kib(&meminfo, "MemTotal:").unwrap_or(0) * 1024;
    let available_bytes = meminfo_kib(&meminfo, "MemAvailable:").unwrap_or(0) * 1024;
    SystemMemoryInfo {
        total_bytes,
        available_bytes,
    }
}

#[cfg(not(target_os = "linux"))]
fn system_memory_info_impl() -> SystemMemoryInfo {
    SystemMemoryInfo {
        total_bytes: 0,
        available_bytes: 0,
    }
}

#[cfg(target_os = "linux")]
fn meminfo_kib(meminfo: &str, key: &str) -> Option<u64> {
    let line = meminfo.lines().find(|line| line.starts_with(key))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

#[cfg(windows)]
fn save_tavily_token_impl(token: &str) -> Result<(), CommandError> {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::Security::Credentials::{
        CredWriteW, CREDENTIALW, CRED_PERSIST_LOCAL_MACHINE, CRED_TYPE_GENERIC,
    };

    let mut target = wide_null(TAVILY_CREDENTIAL_TARGET);
    let mut user = wide_null("tavily");
    let mut token_bytes = token.as_bytes().to_vec();
    let credential = CREDENTIALW {
        Flags: 0,
        Type: CRED_TYPE_GENERIC,
        TargetName: target.as_mut_ptr(),
        Comment: std::ptr::null_mut(),
        LastWritten: FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        },
        CredentialBlobSize: token_bytes.len() as u32,
        CredentialBlob: token_bytes.as_mut_ptr(),
        Persist: CRED_PERSIST_LOCAL_MACHINE,
        AttributeCount: 0,
        Attributes: std::ptr::null_mut(),
        TargetAlias: std::ptr::null_mut(),
        UserName: user.as_mut_ptr(),
    };

    let ok = unsafe { CredWriteW(&credential, 0) };
    if ok == 0 {
        return Err(last_windows_credential_error("save_tavily_token failed"));
    }
    Ok(())
}

#[cfg(not(windows))]
fn save_tavily_token_impl(_token: &str) -> Result<(), CommandError> {
    Err(CommandError::new(
        "web_search_unavailable",
        "Tavily token storage is only available in the Windows desktop app.",
    ))
}

#[cfg(windows)]
fn tavily_token_status_impl() -> TavilyTokenStatus {
    match read_tavily_token_impl() {
        Ok(token) => TavilyTokenStatus {
            available: true,
            configured: token.is_some(),
            reason: None,
        },
        Err(error) => TavilyTokenStatus {
            available: true,
            configured: false,
            reason: Some(error.message),
        },
    }
}

#[cfg(not(windows))]
fn tavily_token_status_impl() -> TavilyTokenStatus {
    TavilyTokenStatus {
        available: false,
        configured: false,
        reason: Some("Tavily token storage is only available in the Windows desktop app.".into()),
    }
}

#[cfg(windows)]
fn delete_tavily_token_impl() -> Result<(), CommandError> {
    use windows_sys::Win32::Foundation::{GetLastError, ERROR_NOT_FOUND};
    use windows_sys::Win32::Security::Credentials::{CredDeleteW, CRED_TYPE_GENERIC};

    let target = wide_null(TAVILY_CREDENTIAL_TARGET);
    let ok = unsafe { CredDeleteW(target.as_ptr(), CRED_TYPE_GENERIC, 0) };
    if ok == 0 {
        let last_error = unsafe { GetLastError() };
        if last_error == ERROR_NOT_FOUND {
            return Ok(());
        }
        return Err(last_windows_credential_error("delete_tavily_token failed"));
    }
    Ok(())
}

#[cfg(not(windows))]
fn delete_tavily_token_impl() -> Result<(), CommandError> {
    Err(CommandError::new(
        "web_search_unavailable",
        "Tavily token storage is only available in the Windows desktop app.",
    ))
}

#[cfg(windows)]
fn read_tavily_token_impl() -> Result<Option<String>, CommandError> {
    use windows_sys::Win32::Foundation::{GetLastError, ERROR_NOT_FOUND};
    use windows_sys::Win32::Security::Credentials::{
        CredFree, CredReadW, CREDENTIALW, CRED_TYPE_GENERIC,
    };

    let target = wide_null(TAVILY_CREDENTIAL_TARGET);
    let mut credential: *mut CREDENTIALW = std::ptr::null_mut();
    let ok = unsafe { CredReadW(target.as_ptr(), CRED_TYPE_GENERIC, 0, &mut credential) };
    if ok == 0 {
        let last_error = unsafe { GetLastError() };
        if last_error == ERROR_NOT_FOUND {
            return Ok(None);
        }
        return Err(last_windows_credential_error("read_tavily_token failed"));
    }
    if credential.is_null() {
        return Ok(None);
    }

    let token_bytes = unsafe {
        let credential_ref = &*credential;
        let bytes = std::slice::from_raw_parts(
            credential_ref.CredentialBlob,
            credential_ref.CredentialBlobSize as usize,
        );
        let token_bytes = bytes.to_vec();
        CredFree(credential.cast());
        token_bytes
    };
    let token = String::from_utf8(token_bytes).map_err(|_| {
        CommandError::new("tavily_token_invalid", "Stored Tavily token is not valid UTF-8.")
    })?;
    Ok(Some(token))
}

#[cfg(not(windows))]
fn read_tavily_token_impl() -> Result<Option<String>, CommandError> {
    Err(CommandError::new(
        "web_search_unavailable",
        "Tavily search is only available in the Windows desktop app.",
    ))
}

#[cfg(windows)]
fn last_windows_credential_error(context: &str) -> CommandError {
    use windows_sys::Win32::Foundation::GetLastError;
    let last_error = unsafe { GetLastError() };
    CommandError::new(
        "credential_error",
        format!("{} with Windows error {}.", context, last_error),
    )
}

#[cfg(windows)]
fn wide_null(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_web_search_request() {
        let request = normalize_web_search_request(WebSearchRequest {
            query: "  hello  ".into(),
            max_results: None,
        })
        .expect("request should normalize");

        assert_eq!(request.query, "hello");
        assert_eq!(request.max_results, Some(5));
    }

    #[test]
    fn rejects_invalid_web_search_request() {
        assert!(normalize_web_search_request(WebSearchRequest {
            query: " ".into(),
            max_results: Some(1),
        })
        .is_err());

        assert!(normalize_web_search_request(WebSearchRequest {
            query: "hello".into(),
            max_results: Some(6),
        })
        .is_err());
    }

    #[test]
    fn parses_only_safe_tavily_result_fields() {
        let response = parse_tavily_search_response(serde_json::json!({
            "answer": "hidden",
            "images": ["hidden"],
            "usage": { "credits": 1 },
            "results": [{
                "title": "Tavily",
                "url": "https://docs.tavily.com/",
                "content": "Search docs",
                "score": 0.99,
                "raw_content": "raw page",
                "favicon": "https://example.test/favicon.ico",
                "images": ["hidden"]
            }]
        }))
        .expect("response should parse");

        assert_eq!(
            response,
            WebSearchResponse {
                results: vec![WebSearchResult {
                    title: "Tavily".into(),
                    url: "https://docs.tavily.com/".into(),
                    snippet: "Search docs".into(),
                }],
            },
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn non_windows_token_status_is_unavailable() {
        let status = tavily_token_status_impl();
        assert!(!status.available);
        assert!(!status.configured);
    }
}

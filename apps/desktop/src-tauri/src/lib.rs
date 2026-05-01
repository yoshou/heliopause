#[derive(serde::Serialize)]
struct SystemMemoryInfo {
    total_bytes: u64,
    available_bytes: u64,
}

#[tauri::command]
fn system_memory_info() -> SystemMemoryInfo {
    system_memory_info_impl()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![system_memory_info])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
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

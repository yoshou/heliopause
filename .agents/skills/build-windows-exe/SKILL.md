---
name: build-windows-exe
description: Build or rebuild the Heliopause desktop Windows .exe with Tauri. Use when the user asks for a Windows exe, Windows build, Tauri Windows artifact, or to fix/rebuild a blank Windows app.
---

# Build Windows Exe

## Workflow

Use this skill only in the Heliopause repo. Build the Windows executable through Tauri, not by running `cargo build` directly. A direct Cargo build can produce an executable that opens to a blank white window because Tauri's normal asset preparation is bypassed.

1. Check the worktree first:

```bash
git status --short
```

Do not stage or revert unrelated user changes.

2. Confirm the Windows GNU target exists:

```bash
rustup target list --installed
```

If `x86_64-pc-windows-gnu` is missing, install it with:

```bash
rustup target add x86_64-pc-windows-gnu
```

3. Run the canonical build:

```bash
pnpm --filter @heliopause/desktop tauri build --target x86_64-pc-windows-gnu
```

This command runs the frontend build before compiling Tauri and embeds the current `apps/desktop/dist` assets.

4. If Tauri refuses to build because the NPM and Rust Tauri versions are mismatched, align the Rust crate to the installed `@tauri-apps/api` major/minor version, then rerun the Tauri build. For example, when `@tauri-apps/api` is `2.11.0`:

```bash
cargo update --manifest-path apps/desktop/src-tauri/Cargo.toml -p tauri --precise 2.11.0
pnpm --filter @heliopause/desktop tauri build --target x86_64-pc-windows-gnu
```

Commit the resulting `apps/desktop/src-tauri/Cargo.lock` change only if it was necessary for the build and the user asked for a commit.

## Output

The Windows executable is written here:

```text
apps/desktop/src-tauri/target/x86_64-pc-windows-gnu/release/heliopause.exe
```

Report its timestamp and size after building:

```bash
stat -c '%y %s %n' apps/desktop/src-tauri/target/x86_64-pc-windows-gnu/release/heliopause.exe
```

If the user mentions a blank white window, rebuild with the Tauri command above and explicitly say that direct Cargo builds should be avoided for this app.

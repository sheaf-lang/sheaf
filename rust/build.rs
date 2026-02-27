use std::env;
use std::path::PathBuf;

fn main() {
    // Search order for IREE runtime libraries:
    // 1. IREE_BUILD_DIR env var (CMake build tree)
    // 2. IREE_DIST_DIR env var (nightly dist flat layout)
    // 3. ~/bin/iree-build (default CMake build)
    // 4. IREE_RUNTIME_LIB_DIR env var (explicit lib directory)

    let found = try_cmake_layout()
        || try_dist_layout()
        || try_explicit_lib_dir();

    if !found {
        println!("cargo:warning=IREE runtime not found");
        println!("cargo:warning=Set IREE_BUILD_DIR (CMake build) or IREE_DIST_DIR (nightly dist)");
        println!("cargo:warning=Building without IREE runtime support");
        println!("cargo:rustc-check-cfg=cfg(iree_runtime)");
        return;
    }

    // macOS system frameworks
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }

    // System libs
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // Tell Rust code that IREE is available
    println!("cargo:rustc-check-cfg=cfg(iree_runtime)");
    println!("cargo:rustc-cfg=iree_runtime");

    println!("cargo:rerun-if-env-changed=IREE_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=IREE_DIST_DIR");
    println!("cargo:rerun-if-env-changed=IREE_RUNTIME_LIB_DIR");
    println!("cargo:rerun-if-changed=build.rs");
}

/// CMake build tree layout: iree-build/runtime/src/iree/runtime/libiree_runtime_unified.a
fn try_cmake_layout() -> bool {
    let iree_build_dir = env::var("IREE_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = env::var("HOME").unwrap_or_default();
            PathBuf::from(home).join("bin/iree-build")
        });

    let runtime_dir = iree_build_dir.join("runtime/src/iree/runtime");
    let unified_lib = runtime_dir.join("libiree_runtime_unified.a");

    if !unified_lib.exists() {
        return false;
    }

    println!("cargo:rustc-link-search=native={}", runtime_dir.display());
    println!("cargo:rustc-link-lib=static=iree_runtime_unified");

    let flatcc_dir = iree_build_dir.join("build_tools/third_party/flatcc");
    if flatcc_dir.join("libflatcc_parsing.a").exists() {
        println!("cargo:rustc-link-search=native={}", flatcc_dir.display());
        println!("cargo:rustc-link-lib=static=flatcc_parsing");
    }

    true
}

/// Nightly dist flat layout: iree-dist-*/lib/libiree_runtime_unified.a
fn try_dist_layout() -> bool {
    let dist_dir = match env::var("IREE_DIST_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => return false,
    };

    let lib_dir = dist_dir.join("lib");
    let unified_lib = lib_dir.join("libiree_runtime_unified.a");

    if !unified_lib.exists() {
        return false;
    }

    link_from_dir(&lib_dir)
}

/// Explicit lib directory: all .a files in one place
fn try_explicit_lib_dir() -> bool {
    let lib_dir = match env::var("IREE_RUNTIME_LIB_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => return false,
    };

    let unified_lib = lib_dir.join("libiree_runtime_unified.a");
    if !unified_lib.exists() {
        return false;
    }

    link_from_dir(&lib_dir)
}

/// Link iree_runtime_unified + flatcc_parsing from a single directory
fn link_from_dir(lib_dir: &PathBuf) -> bool {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=iree_runtime_unified");

    if lib_dir.join("libflatcc_parsing.a").exists() {
        println!("cargo:rustc-link-lib=static=flatcc_parsing");
    }
    if lib_dir.join("libflatcc_runtime.a").exists() {
        println!("cargo:rustc-link-lib=static=flatcc_runtime");
    }

    true
}

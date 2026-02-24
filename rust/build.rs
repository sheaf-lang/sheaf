use std::env;
use std::path::PathBuf;

fn main() {
    let iree_build_dir = env::var("IREE_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = env::var("HOME").expect("HOME not set");
            PathBuf::from(home).join("bin/iree-build")
        });

    let runtime_dir = iree_build_dir.join("runtime/src/iree/runtime");
    let unified_lib = runtime_dir.join("libiree_runtime_unified.a");

    if !unified_lib.exists() {
        println!("cargo:warning=IREE runtime not found at {}", unified_lib.display());
        println!("cargo:warning=Set IREE_BUILD_DIR to your IREE build directory");
        println!("cargo:warning=Building without IREE runtime support");
        return;
    }

    println!("cargo:rustc-link-search=native={}", runtime_dir.display());
    println!("cargo:rustc-link-lib=static=iree_runtime_unified");

    // FlatCC (used by IREE for flatbuffer parsing/verification)
    let flatcc_dir = iree_build_dir.join("build_tools/third_party/flatcc");
    println!("cargo:rustc-link-search=native={}", flatcc_dir.display());
    println!("cargo:rustc-link-lib=static=flatcc_parsing");

    // macOS system frameworks
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }

    // System libs
    println!("cargo:rustc-link-lib=dylib=c++");

    // Tell Rust code that IREE is available
    println!("cargo:rustc-check-cfg=cfg(iree_runtime)");
    println!("cargo:rustc-cfg=iree_runtime");

    // Rerun if IREE build changes
    println!("cargo:rerun-if-env-changed=IREE_BUILD_DIR");
    println!("cargo:rerun-if-changed=build.rs");
}

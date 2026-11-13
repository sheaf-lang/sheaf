// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Legacy AOT sidecars must not affect source evaluation or imports.

use std::path::PathBuf;
use std::process::Command;

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        for attempt in 0..128 {
            let path = std::env::temp_dir().join(format!(
                "sheaf-legacy-aot-{}-{attempt}", std::process::id()
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("create fixture: {error}"),
            }
        }
        panic!("no unused fixture directory");
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn source_evaluation_ignores_legacy_aot_sidecars() {
    let binary = std::fs::canonicalize(
        std::env::var_os("SHEAF_TEST_BINARY").expect("Bazel-built Sheaf binary")
    ).unwrap();
    let compiler = std::fs::canonicalize(
        std::env::var_os("IREE_COMPILE").expect("matching IREE compiler")
    ).unwrap();
    let fixture = Fixture::new();
    let definition = "(defn legacy-square [x] (* x x))";
    let call = "(println (sum (legacy-square (tensor [2.0 3.0]))))";
    std::fs::write(fixture.0.join("model.shf"), definition).unwrap();
    std::fs::write(fixture.0.join("main.shf"), format!("{definition}\n{call}")).unwrap();
    std::fs::write(fixture.0.join("module.json"), "invalid legacy manifest").unwrap();
    std::fs::write(fixture.0.join("module.vmfb"), "invalid legacy bytecode").unwrap();

    for flags in [vec![], vec!["--trace"], vec!["--blame"], vec!["--mem-profile"],
        vec!["--blame", "--mem-profile"]] {
        let output = Command::new(&binary)
            .arg("main.shf")
            .args(&flags)
            .args(["--device", "cpu"])
            .env("IREE_COMPILE", &compiler)
            .current_dir(&fixture.0)
            .output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{flags:?}: {stderr}");
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "13.0", "{flags:?}");
        assert!(!stderr.contains("module.json"), "{flags:?}: {stderr}");
        assert!(!stderr.contains("module.vmfb"), "{flags:?}: {stderr}");
    }

    let output = Command::new(&binary)
        .args(["-c", &format!("(use \"./model.shf\") {call}"), "--device", "cpu"])
        .env("IREE_COMPILE", &compiler)
        .current_dir(&fixture.0)
        .output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "13.0");
    assert!(!stderr.contains("module.json"), "{stderr}");
    assert!(!stderr.contains("module.vmfb"), "{stderr}");
    assert_eq!(std::fs::read_to_string(fixture.0.join("module.json")).unwrap(),
        "invalid legacy manifest");
    assert_eq!(std::fs::read_to_string(fixture.0.join("module.vmfb")).unwrap(),
        "invalid legacy bytecode");
}

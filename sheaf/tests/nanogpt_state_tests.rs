// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

use std::fs;
use std::process::Command;

#[test]
fn train_step_compiles_once_on_start_and_resume() {
    let binary = fs::canonicalize(std::env::var("SHEAF_TEST_BINARY").unwrap()).unwrap();
    let compiler = fs::canonicalize(std::env::var("IREE_COMPILE").unwrap()).unwrap();
    let device = std::env::var("SHEAF_SIGNATURE_DEVICE").unwrap_or_else(|_| "cpu".into());
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let fixture = std::env::temp_dir().join(format!("sheaf-nanogpt-state-{}-{nonce}", std::process::id()));
    fs::create_dir_all(fixture.join("out-shakespeare-char")).unwrap();
    fs::create_dir_all(fixture.join("data/shakespeare_char")).unwrap();
    for (variable, filename) in [("NANOGPT_MODEL_SOURCE", "model.shf"), ("NANOGPT_TRAIN_SOURCE", "train.shf")] {
        fs::copy(std::env::var(variable).unwrap(), fixture.join(filename)).unwrap();
    }
    fs::write(fixture.join("out-shakespeare-char/config.json"), r#"{"model_args": {
        "n_layer": 1, "n_head": 2, "n_embd": 16, "block_size": 8,
        "vocab_size": 4, "bias": false, "dropout": 0.0
    }}"#).unwrap();
    fs::write(fixture.join("data/shakespeare_char/vocab.json"), r#"{"a":0,"b":1,"c":2,"d":3}"#).unwrap();
    fs::write(fixture.join("data/shakespeare_char/input.txt"), "abcd".repeat(128)).unwrap();

    let mut compilations = Vec::new();
    for (phase, marker) in [("start", "No checkpoint; initializing"), ("resume", "Resuming from")] {
        let cache = fixture.join("__sheaf__");
        if cache.exists() { fs::remove_dir_all(cache).unwrap(); }
        let output = Command::new(&binary)
            .args(["--device", &device, "-v", "train.shf"])
            .env("IREE_COMPILE", &compiler).current_dir(&fixture).output().unwrap();
        fs::write(fixture.join(format!("{phase}.stdout")), &output.stdout).unwrap();
        fs::write(fixture.join(format!("{phase}.stderr")), &output.stderr).unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{phase}: {stderr}\nlogs: {}", fixture.display());
        assert!(stdout.contains(marker), "{phase}: {stdout}");
        assert!(stdout.contains("Step 100"), "{phase}: {stdout}");
        assert!(!stdout.to_lowercase().contains("| loss: nan"), "{phase}: {stdout}");
        assert!(!stderr.contains("jit: train-step skipped"), "{phase}: {stderr}");
        compilations.push(stderr.matches("jit: compiling train-step [").count());
        assert!(fixture.join("out-shakespeare-char/ckpt.safetensors").is_file());
        if phase == "start" {
            let output = Command::new(&binary).args(["--device", &device, "-c",
                "(print (keys (io \"load\" \"out-shakespeare-char/ckpt.safetensors\")))"])
                .env("IREE_COMPILE", &compiler).current_dir(&fixture).output().unwrap();
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            assert!(!String::from_utf8_lossy(&output.stdout).contains(":loss"),
                    "resume must cover a checkpoint without the transient loss field");
        }
    }
    assert_eq!(compilations, [1, 1], "start/resume compilation counts; logs: {}", fixture.display());
    fs::remove_dir_all(fixture).unwrap();
}

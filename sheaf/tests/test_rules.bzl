"""Rust test helpers for the Sheaf Cargo package."""

load("@crates//:defs.bzl", "aliases", "all_crate_deps", "crate_edition")
load("@rules_rust//rust:defs.bzl", "rust_test")

_CARGO_PACKAGE = "sheaf"


def _sheaf_program_test_impl(ctx):
    executable = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.symlink(
        output = executable,
        target_file = ctx.executable._sheaf,
        is_executable = True,
    )

    runfiles = ctx.runfiles(
        files = [ctx.file._iree_compile] + ctx.files.data,
    ).merge(
        ctx.attr._iree_compile[DefaultInfo].default_runfiles,
    ).merge(ctx.attr._sheaf[DefaultInfo].default_runfiles)
    for target in ctx.attr.data:
        runfiles = runfiles.merge(target[DefaultInfo].default_runfiles)

    return [
        DefaultInfo(executable = executable, runfiles = runfiles),
        RunEnvironmentInfo(
            environment = {"IREE_COMPILE": ctx.file._iree_compile.short_path},
        ),
    ]

_sheaf_program_test = rule(
    implementation = _sheaf_program_test_impl,
    test = True,
    attrs = {
        "data": attr.label_list(allow_files = True),
        "_iree_compile": attr.label(
            allow_single_file = True,
            default = "@iree_compiler//:iree-compile",
        ),
        "_sheaf": attr.label(
            cfg = "target",
            default = "//sheaf:bin",
            executable = True,
        ),
    },
)

def sheaf_program_test(name, expression, data = [], device = "cpu", jit_strict = False, **kwargs):
    _sheaf_program_test(
        name = name,
        args = (["--jit-strict"] if jit_strict else []) + ["--device", device, "-c", expression],
        data = ["regressions.shf"] + data,
        tags = [device],
        **kwargs
    )

def sheaf_integration_test(name, data = [], env = None):
    test_env = {"RUST_TEST_THREADS": "1"}
    if env:
        test_env.update(env)
    rust_test(
        name = name,
        srcs = [name + ".rs"],
        aliases = aliases(package_name = _CARGO_PACKAGE),
        data = data,
        edition = crate_edition(package_name = _CARGO_PACKAGE),
        env = test_env,
        rustc_env = {
            "CARGO_MANIFEST_DIR": "sheaf",
        },
        deps = ["//sheaf:compiler_lib"] + all_crate_deps(
            normal = True,
            package_name = _CARGO_PACKAGE,
        ),
    )

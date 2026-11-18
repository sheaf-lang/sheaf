"""Executable rule for running a Sheaf example with Bazel-managed data."""

def _single_file(target, attribute_name):
    files = target.files.to_list()
    if len(files) != 1:
        fail("%s must provide exactly one file" % attribute_name)
    return files[0]

def _runfile_path(ctx, file):
    if file.short_path.startswith("../"):
        return file.short_path[3:]
    return ctx.workspace_name + "/" + file.short_path

def _shell_quote(value):
    return "'" + value.replace("'", "'\"'\"'") + "'"

def _sheaf_example_impl(ctx):
    executable = ctx.actions.declare_file(ctx.label.name + "_runner.sh")
    sheaf = ctx.executable.sheaf
    script = _single_file(ctx.attr.script, "script")
    runtime_files = [sheaf, script]
    links = [(script, script.basename)]

    for target, destination in ctx.attr.files.items():
        file = _single_file(target, "files")
        runtime_files.append(file)
        links.append((file, destination))

    link_commands = []
    for file, destination in links:
        link_commands.extend([
            "mkdir -p \"$workdir\"/%s" % _shell_quote(destination.rpartition("/")[0] or "."),
            "ln -sfn \"$(resolve_runfile %s)\" \"$workdir\"/%s" % (
                _shell_quote(_runfile_path(ctx, file)),
                _shell_quote(destination),
            ),
        ])

    content = """#!/usr/bin/env bash
set -euo pipefail

resolve_runfile() {
  local logical_path="$1"
  if [[ -n "${RUNFILES_DIR:-}" ]]; then
    printf '%%s\\n' "$RUNFILES_DIR/$logical_path"
  elif [[ -d "$0.runfiles" ]]; then
    printf '%%s\\n' "$0.runfiles/$logical_path"
  elif [[ -n "${RUNFILES_MANIFEST_FILE:-}" ]]; then
    grep -m1 "^${logical_path} " "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-
  else
    printf 'Cannot resolve Bazel runfile: %%s\\n' "$logical_path" >&2
    exit 1
  fi
}

workdir="${XDG_CACHE_HOME:-$HOME/.cache}/sheaf/examples/%s"
%s
cd "$workdir"
exec "$(resolve_runfile %s)" %s
""" % (
        ctx.label.name,
        "\n".join(link_commands),
        _shell_quote(_runfile_path(ctx, sheaf)),
        _shell_quote(script.basename),
    )

    ctx.actions.write(executable, content, is_executable = True)
    return [DefaultInfo(
        executable = executable,
        files = depset([executable]),
        runfiles = ctx.runfiles(files = runtime_files),
    )]

sheaf_example = rule(
    implementation = _sheaf_example_impl,
    executable = True,
    attrs = {
        "files": attr.label_keyed_string_dict(allow_files = True),
        "script": attr.label(allow_single_file = [".shf"], mandatory = True),
        "sheaf": attr.label(
            cfg = "target",
            default = Label("//sheaf:bin"),
            executable = True,
        ),
    },
)

"""Repository rule fetching the Bazel-native IREE runtime pinned by Cargo.toml."""

_IREE_SOURCES = {
    "3.12.0": struct(
        commit = "2b05c5dbb2f2ecb27c0d3941e80ee8d2f16e890d",
        sha256 = "8920c8457d71a48322b7258bc0776d734f2e44a87c3b68031bbfb19299cd3ee5",
    ),
}

_FLATCC_COMMIT = "9362cd00f0007d8cbee7bff86e90fb4b6b227ff3"
_FLATCC_SHA256 = "f77f842e996f5bbfa25305a7b38b40d1325fb44154f2bf9880c58356ece92c62"
_VULKAN_HEADERS_COMMIT = "df60f0316899460eeaaefa06d2dd7e4e300c1604"
_VULKAN_HEADERS_SHA256 = "70de3ec1eec289552052dfb886a49a4578b65d84fd536da3254731c8165cb3e6"

def _iree_version(cargo_toml):
    for line in cargo_toml.splitlines():
        if line.startswith("iree-version"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip("\"")
    fail("iree-version not found in sheaf/Cargo.toml")

def _replace(repository_ctx, path, old, new):
    content = repository_ctx.read(path)
    if old not in content:
        fail("expected text not found in {}".format(path))
    repository_ctx.file(path, content.replace(old, new))

def _iree_source_repository_impl(repository_ctx):
    version = _iree_version(repository_ctx.read(repository_ctx.attr.cargo_toml))
    source = _IREE_SOURCES.get(version)
    if not source:
        fail("no IREE source archive configured for version {}".format(version))
    repository_ctx.download_and_extract(
        url = "https://github.com/iree-org/iree/archive/{}.tar.gz".format(source.commit),
        sha256 = source.sha256,
        stripPrefix = "iree-{}".format(source.commit),
    )
    repository_ctx.delete(".bazelignore")
    repository_ctx.download_and_extract(
        url = "https://github.com/dvidelabs/flatcc/archive/{}.tar.gz".format(_FLATCC_COMMIT),
        output = "third_party/flatcc",
        sha256 = _FLATCC_SHA256,
        stripPrefix = "flatcc-{}".format(_FLATCC_COMMIT),
    )
    repository_ctx.download_and_extract(
        url = "https://github.com/KhronosGroup/Vulkan-Headers/archive/{}.tar.gz".format(_VULKAN_HEADERS_COMMIT),
        output = "third_party/vulkan_headers",
        sha256 = _VULKAN_HEADERS_SHA256,
        stripPrefix = "Vulkan-Headers-{}".format(_VULKAN_HEADERS_COMMIT),
    )

    flatcc_build = repository_ctx.read("build_tools/third_party/flatcc/BUILD.overlay")
    flatcc_build = flatcc_build.replace(
        "cc_library(\n    name = \"compiler\",",
        "cc_library(\n    name = \"compiler\",\n    copts = [\"-w\"],",
    )
    repository_ctx.file(
        "third_party/flatcc/BUILD.bazel",
        flatcc_build,
    )
    repository_ctx.file(
        "third_party/nccl/BUILD.bazel",
        repository_ctx.read("build_tools/third_party/nccl/BUILD.overlay"),
    )
    repository_ctx.file(
        "third_party/vulkan_headers/BUILD.bazel",
        repository_ctx.read("build_tools/third_party/vulkan_headers/BUILD.overlay"),
    )

    repository_ctx.file(
        "build_tools/bazel/runtime_only_tblgen.bzl",
        """def gentbl_cc_library(**kwargs):
    fail("compiler targets are not available in the runtime-only IREE repository")

def gentbl_filegroup(**kwargs):
    fail("compiler targets are not available in the runtime-only IREE repository")

def td_library(**kwargs):
    fail("compiler targets are not available in the runtime-only IREE repository")
""",
    )
    _replace(
        repository_ctx,
        "build_tools/bazel/build_defs.oss.bzl",
        "load(\"@llvm-project//mlir:tblgen.bzl\", \"gentbl_cc_library\", \"gentbl_filegroup\", \"td_library\")\n",
        "load(\"//build_tools/bazel:runtime_only_tblgen.bzl\", \"gentbl_cc_library\", \"gentbl_filegroup\", \"td_library\")\n",
    )
    _replace(
        repository_ctx,
        "build_tools/bazel/build_defs.oss.bzl",
        """def iree_cc_library(includes = [], system_includes = [], **kwargs):
    \"\"\"Base function for all cc_library targets.

    This is a pass-through to the native cc_library, which integrators can
    customize with additional flags as needed. Prefer to use the compiler
    and runtime versions instead.

    Note that Bazel does not distinguish between includes and system_includes,
    but CMake does. So we allow them to be separate and glom them together
    here.
    \"\"\"
    cc_library(
        includes = includes + system_includes,
        **kwargs
    )
""",
        """def iree_cc_library(copts = [], includes = [], system_includes = [], **kwargs):
    \"\"\"Base function for all cc_library targets.

    This is a pass-through to the native cc_library, which integrators can
    customize with additional flags as needed. Prefer to use the compiler
    and runtime versions instead.

    Note that Bazel does not distinguish between includes and system_includes,
    but CMake does. So we allow them to be separate and glom them together
    here.
    \"\"\"
    cc_library(
        copts = copts + [\"-w\"],
        includes = includes + system_includes,
        **kwargs
    )
""",
    )
    _replace(
        repository_ctx,
        "build_tools/bazel/iree_flatcc.bzl",
        "@com_github_dvidelabs_flatcc//:flatcc",
        "//third_party/flatcc:flatcc",
    )
    _replace(
        repository_ctx,
        "build_tools/bazel/iree_flatcc.bzl",
        "\"-I runtime/src\",",
        "\"-I $$(dirname $(location //runtime/src:schema_include_root))\",",
    )
    _replace(
        repository_ctx,
        "build_tools/bazel/iree_flatcc.bzl",
        "srcs = srcs + includes,",
        "srcs = srcs + includes + [\"//runtime/src:schema_include_root\"],",
    )
    runtime_build_path = "runtime/src/BUILD.bazel"
    repository_ctx.file(
        runtime_build_path,
        repository_ctx.read(runtime_build_path) + """
filegroup(
    name = "schema_include_root",
    srcs = ["BUILD.bazel"],
    visibility = ["//visibility:public"],
)
""",
    )
    _replace(
        repository_ctx,
        "runtime/src/iree/base/internal/flatcc/BUILD.bazel",
        "@com_github_dvidelabs_flatcc//:",
        "//third_party/flatcc:",
    )
    _replace(
        repository_ctx,
        "runtime/src/iree/hal/drivers/cuda/BUILD.bazel",
        "@nccl//:headers",
        "//third_party/nccl:headers",
    )
    _replace(
        repository_ctx,
        "runtime/src/iree/hal/drivers/vulkan/BUILD.bazel",
        "@vulkan_headers",
        "//third_party/vulkan_headers:vulkan_headers",
    )
    _replace(
        repository_ctx,
        "runtime/src/iree/hal/drivers/vulkan/util/BUILD.bazel",
        "@vulkan_headers",
        "//third_party/vulkan_headers:vulkan_headers",
    )

    isa_build = "runtime/src/iree/vm/bytecode/isa/BUILD.bazel"
    _replace(
        repository_ctx,
        isa_build,
        "        \"//build_tools/scripts:generate_vm_isa.py\",\n    ],",
        "    ],\n    tools = [\"//build_tools/scripts:generate_vm_isa\"],",
    )
    _replace(
        repository_ctx,
        isa_build,
        "        \"python3\",\n        \"$(rootpath //build_tools/scripts:generate_vm_isa.py)\",",
        "        \"$(execpath //build_tools/scripts:generate_vm_isa)\",",
    )
    _replace(
        repository_ctx,
        isa_build,
        "$(rootpath isa.json)",
        "$(execpath isa.json)",
    )

    _add_metal_build_files(repository_ctx)

def _add_metal_build_files(repository_ctx):
    repository_ctx.file(
        "runtime/src/iree/hal/drivers/metal/BUILD.bazel",
        """load("@rules_cc//cc:defs.bzl", "objc_library")

package(default_visibility = ["//visibility:public"])

objc_library(
    name = "metal",
    non_arc_srcs = [
        "builtin_executables.m",
        "direct_allocator.m",
        "direct_command_buffer.m",
        "executable.m",
        "metal_buffer.m",
        "metal_device.m",
        "metal_driver.m",
        "nop_executable_cache.m",
        "shared_event.m",
        "staging_buffer.m",
    ],
    hdrs = glob(["*.h"]),
    copts = ["-w"],
    sdk_frameworks = [
        "Foundation",
        "Metal",
    ],
    deps = [
        "//runtime/src/iree/async",
        "//runtime/src/iree/async/util:proactor_pool",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base/internal",
        "//runtime/src/iree/base/internal:arena",
        "//runtime/src/iree/base/internal/flatcc:parsing",
        "//runtime/src/iree/base/threading",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/drivers/metal/builtin",
        "//runtime/src/iree/hal/utils:deferred_command_buffer",
        "//runtime/src/iree/hal/utils:executable_debug_info",
        "//runtime/src/iree/hal/utils:executable_header",
        "//runtime/src/iree/hal/utils:file_transfer",
        "//runtime/src/iree/hal/utils:files",
        "//runtime/src/iree/hal/utils:queue_emulation",
        "//runtime/src/iree/hal/utils:queue_host_call_emulation",
        "//runtime/src/iree/hal/utils:resource_set",
        "//runtime/src/iree/schemas:executable_debug_info_c_fbs",
        "//runtime/src/iree/schemas:metal_executable_def_c_fbs",
        "//runtime/src:runtime_defines",
    ],
)
"""
    )
    repository_ctx.file(
        "runtime/src/iree/hal/drivers/metal/builtin/BUILD.bazel",
        """load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")

package(default_visibility = ["//visibility:public"])

iree_c_embed_data(
    name = "builtin",
    srcs = [
        "copy_buffer_generic.metal",
        "fill_buffer_generic.metal",
    ],
    c_file_output = "metal_buffer_kernels.c",
    flatten = True,
    h_file_output = "metal_buffer_kernels.h",
    identifier = "metal_buffer_kernels",
)
""",
    )
    repository_ctx.file(
        "runtime/src/iree/hal/drivers/metal/registration/BUILD.bazel",
        """load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library")

package(default_visibility = ["//visibility:public"])

iree_runtime_cc_library(
    name = "registration",
    srcs = ["driver_module.c"],
    hdrs = ["driver_module.h"],
    defines = ["IREE_HAVE_HAL_METAL_DRIVER_MODULE=1"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base/tooling:flags",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/drivers/metal",
    ],
)
""",
    )

    drivers_path = "runtime/src/iree/hal/drivers/BUILD.bazel"
    drivers = repository_ctx.read(drivers_path)
    drivers = drivers.replace(
        "# AMDGPU is special and is conditioned on availability of ROCM.\n",
        """config_setting(
    name = "metal_enabled",
    flag_values = {
        ":enabled_drivers": "metal",
    },
)

# AMDGPU is special and is conditioned on availability of ROCM.
""",
    )
    metal_select_anchor = "           select({\n               \":null_enabled\""
    if metal_select_anchor not in drivers:
        fail("failed to locate the IREE driver dependency list")
    drivers = drivers.replace(
        metal_select_anchor,
        "           select({\n               \":metal_enabled\": [\"//runtime/src/iree/hal/drivers/metal/registration\"],\n               \"//conditions:default\": [],\n           }) +\n           select({\n               \":null_enabled\"",
    )
    repository_ctx.file(drivers_path, drivers)

iree_source_repository = repository_rule(
    implementation = _iree_source_repository_impl,
    attrs = {
        "cargo_toml": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
    },
)

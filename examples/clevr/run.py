"""
CLEVR Neuro-Symbolic Reasoning Demo
"""

import os
import sys

import jax
import jax.numpy as jnp

from sheaf import Sheaf

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

import data


def extract_operation_names(query):
    """Extract operation names from query in depth-first order."""
    ops = []

    def traverse(q):
        if not isinstance(q, list) or len(q) == 0:
            return

        op = q[0]

        # Process all sub-queries first (depth-first)
        for i in range(1, len(q)):
            arg = q[i]
            if isinstance(arg, list):
                traverse(arg)

        # Add this operation
        ops.append(op)

    traverse(query)
    return ops


def format_detailed_pipeline(scene_dict, query, answer, predicted, steps):
    """Format the query execution pipeline as readable text with attention values."""
    lines = []

    # Header
    lines.append(f"\nQuery: {query}")

    # Input tensor
    n_objects = len(scene_dict["objects"])
    lines.append(f"Input: Scene [batch=1, n_objects={n_objects}, features=9]")
    lines.append("    ↓")

    # Steps with attention values
    for step_idx, step in enumerate(steps):
        op_name = step["op"]
        attention = jnp.array(step["attention"])

        # Handle batch/scalar dimensions
        if attention.ndim == 2:
            attention = attention[0]
        if attention.ndim == 0:
            attention = jnp.ones(n_objects)

        # Format the step
        step_title = op_name.replace("-", " ").title().replace(" ", "-")
        lines.append(f"[{step_title}] {op_name}")

        # Add operation description
        if "filter" in op_name:
            lines.append("  Operation: Dot product (scene @ W_embed) + Sigmoid")
        elif op_name in ["leftmost", "rightmost"]:
            direction = "smallest" if "leftmost" in op_name else "largest"
            lines.append(
                f"  Operation: Softmax over x-coordinates ({direction} weighted by attention)"
            )
        elif op_name == "unique":
            lines.append("  Operation: Softmax over objectness")
        elif "query" in op_name:
            attr_type = op_name.replace("query-", "").upper()
            lines.append(f"  Operation: Extract {attr_type} logits")

        # Add attention/logits values
        if "query" in op_name:
            # For query operations, show logits with attribute names
            attr_type = op_name.replace("query-", "").lower()
            if attr_type == "color":
                attr_names = data.COLORS
            elif attr_type == "shape":
                attr_names = data.SHAPES
            elif attr_type == "position":
                attr_names = ["x", "y"]
            else:
                attr_names = [f"dim_{i}" for i in range(len(attention))]

            logit_labels = []
            for i, name in enumerate(attr_names):
                if i < len(attention):
                    logit_labels.append(f"{name}: {float(attention[i]):.2f}")
            lines.append(f"  Logits: [{', '.join(logit_labels)}]")
        else:
            # For filter/selection operations, show attention per object
            objects = scene_dict["objects"]
            att_labels = []
            for i, obj in enumerate(objects):
                if i < len(attention):
                    label = f"{obj['color']}_{obj['shape']}"
                    att_val = float(attention[i])
                    att_labels.append(f"{label}: {att_val:.2f}")
            lines.append(f"  Attention: [{', '.join(att_labels)}]")

        lines.append("    ↓")

    # Final answer
    lines.append(f"[Decision] argmax(logits): {str(predicted).upper()}")

    return "\n".join(lines)


def load_model():
    shf = Sheaf()
    model_dir = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(model_dir, "model.shf")

    # Change to model directory so relative imports (use ./utils.shf) work
    old_cwd = os.getcwd()
    os.chdir(model_dir)
    try:
        with open(model_path) as f:
            shf.load(f.read())
    finally:
        os.chdir(old_cwd)

    return shf


def load_params(shf):
    # Parameters are now initialized with optimized values directly
    return shf.init_clevr_params(jax.random.PRNGKey(0))


def test_query(shf, scene_dict, query, answer, params, with_steps=False):
    # Test a single query and return success/failure
    scene = jnp.expand_dims(data.scene_to_tensor(scene_dict), 0)

    # Get steps if requested
    steps = []
    if with_steps:
        query_result = shf.execute_query_with_steps(scene, query, params)
        if isinstance(query_result, (list, tuple)) and len(query_result) == 2:
            result, attention_list = query_result
            # Match operation names with attention tensors
            op_names = extract_operation_names(query)
            for i, op_name in enumerate(op_names):
                if i < len(attention_list):
                    steps.append({"op": op_name, "attention": attention_list[i]})
        else:
            result = query_result
    else:
        result = shf.execute_query(scene, query, params)

    op = query[0]
    if op == "query-color":
        predicted = data.COLORS[int(jnp.argmax(result[0]))]
        success = predicted == answer
    elif op == "query-shape":
        predicted = data.SHAPES[int(jnp.argmax(result[0]))]
        success = predicted == answer
    elif op == "exists?":
        predicted = float(result[0]) > 0.5
        success = predicted == answer
    else:
        predicted, success = None, False

    return {
        "query": query,
        "expected": answer,
        "predicted": predicted,
        "success": success,
        "scene": scene_dict,
        "steps": steps,
    }


def run_tests(shf, params, demo_scene, num_tests=10, show_detailed=True):
    # Run random tests and report accuracy
    key = jax.random.PRNGKey(42)
    passed = 0
    last_result = None

    print("-" * 58)
    print("Test queries:\n")

    for i in range(num_tests):
        key, scene_key, query_key = jax.random.split(key, 3)
        scene = data.generate_scene(scene_key, n_objects=4)
        query, answer = data.generate_query(scene, query_key)

        result = test_query(shf, scene, query, answer, params, with_steps=False)
        status = "PASS" if result["success"] else "FAIL"
        print(
            f"[{status}] {result['query']} -> {result['predicted']} (expected: {result['expected']})"
        )

        if result["success"]:
            passed += 1

    print(f"\nAccuracy: {passed}/{num_tests} ({100 * passed / num_tests:.1f}%)")

    # Show detailed pipeline for custom query: "color of rightmost square"
    if show_detailed:
        print("-" * 58)
        print("\nDetailed operations for query: 'color of rightmost square'\n")
        print("Symbolic attention shaping:")

        # Custom query: color of rightmost square
        custom_query = ["query-color", ["rightmost", ["filter-shape", ":square"]]]

        # Execute with steps (scene was already generated and displayed)
        result = test_query(
            shf, demo_scene, custom_query, None, params, with_steps=True
        )

        if result["steps"]:
            pipeline = format_detailed_pipeline(
                result["scene"],
                custom_query,
                None,  # No expected answer for demo
                result["predicted"],
                result["steps"],
            )
            print(pipeline)

    return passed / num_tests


def main():
    print("CLEVR Neuro-Symbolic Reasoning\n")

    shf = load_model()

    params = load_params(shf)

    # Generate a random scene with at least one square for the demo query
    import time

    demo_seed = int(time.time() * 1000) % 10000
    demo_key = jax.random.PRNGKey(demo_seed)

    # Generate scenes until we get one with at least one square
    demo_scene = None
    max_attempts = 100
    for attempt in range(max_attempts):
        candidate_scene = data.generate_scene(demo_key, n_objects=5)
        has_square = any(obj["shape"] == "square" for obj in candidate_scene["objects"])
        if has_square:
            demo_scene = candidate_scene
            break
        demo_key = jax.random.PRNGKey(demo_seed + attempt + 1)

    if demo_scene is None:
        # Fallback: force create a scene with a square
        demo_scene = data.generate_scene(demo_key, n_objects=5)
        demo_scene["objects"][0]["shape"] = "square"

    # Display the scene tensor and decoded objects
    scene_tensor = data.scene_to_tensor(demo_scene)
    print("Input tensor:")
    print(f"  Shape: {scene_tensor.shape} (5 objects × 9 features)")
    print(f"  Features: [red, green, blue, yellow, circle, square, triangle, x, y]")
    print(f"\n  Raw tensor values:")
    for i, obj_tensor in enumerate(scene_tensor):
        print(f"    Object {i + 1}: {obj_tensor}")

    print("\nDecoded input:")
    for i, obj in enumerate(demo_scene["objects"]):
        print(
            f"  - {obj['color']:7} {obj['shape']:8}   x={obj['x']:.2f}    y={obj['y']:.2f}"
        )

    print()

    # Run tests with the pre-generated scene for detailed demo
    run_tests(shf, params, demo_scene, num_tests=10)


if __name__ == "__main__":
    main()

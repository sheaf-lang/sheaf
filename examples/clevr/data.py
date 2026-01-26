"""
CLEVR-style scene generation for neuro-symbolic reasoning.

Scenes contain colored shapes at 2D positions.
"""

import jax
import jax.numpy as jnp

# Vocabulary
COLORS = ["red", "green", "blue", "yellow"]
SHAPES = ["circle", "square", "triangle"]

# Scene dimensions
MAX_OBJECTS = 5
FEATURE_DIM = 9  # color(4) + shape(3) + position(2)


def generate_scene(key, n_objects=4):
    """
    Generate a random scene with n_objects.

    Returns dict with 'objects' list and 'n_objects' count.
    Each object has: shape, color, x, y (positions in [0.1, 0.9])
    """
    keys = jax.random.split(key, n_objects * 4)
    objects = []

    for i in range(n_objects):
        shape = SHAPES[int(jax.random.randint(keys[i * 4], (), 0, len(SHAPES)))]
        color = COLORS[int(jax.random.randint(keys[i * 4 + 1], (), 0, len(COLORS)))]
        x = float(jax.random.uniform(keys[i * 4 + 2], minval=0.1, maxval=0.9))
        y = float(jax.random.uniform(keys[i * 4 + 3], minval=0.1, maxval=0.9))
        objects.append({"shape": shape, "color": color, "x": x, "y": y})

    return {"objects": objects, "n_objects": n_objects}


def scene_to_tensor(scene):
    """
    Convert scene dict to tensor [MAX_OBJECTS, FEATURE_DIM].

    Each object encoded as: [color_onehot(4), shape_onehot(3), x, y]
    Unused slots are zero-padded.
    """
    features = jnp.zeros((MAX_OBJECTS, FEATURE_DIM))

    for i, obj in enumerate(scene["objects"]):
        color_idx = COLORS.index(obj["color"])
        shape_idx = SHAPES.index(obj["shape"])

        vec = jnp.array(
            [1.0 if j == color_idx else 0.0 for j in range(4)]
            + [1.0 if j == shape_idx else 0.0 for j in range(3)]
            + [obj["x"], obj["y"]]
        )

        features = features.at[i].set(vec)

    return features


def generate_query(scene, key):
    """
    Generate a random query about the scene as an S-expression.

    Query types:
    - "What color is the leftmost [shape]?"
    - "Is there a [color] [shape]?"
    - "What shape is left of the [color] object?"

    Returns (query_sexp, answer).
    """
    objects = scene["objects"]
    query_type = int(jax.random.randint(key, (), 0, 3))

    if query_type == 0:
        # Query color of leftmost shape
        shape = SHAPES[int(jax.random.randint(key, (), 0, len(SHAPES)))]
        candidates = [o for o in objects if o["shape"] == shape]
        if candidates:
            leftmost = min(candidates, key=lambda o: o["x"])
            return [
                "query-color",
                ["leftmost", ["filter-shape", f":{shape}"]],
            ], leftmost["color"]

    elif query_type == 1:
        # Existence query
        color = COLORS[int(jax.random.randint(key, (), 0, len(COLORS)))]
        shape = SHAPES[int(jax.random.randint(key, (), 0, len(SHAPES)))]
        exists = any(o["color"] == color and o["shape"] == shape for o in objects)
        return [
            "exists?",
            ["intersect", ["filter-color", f":{color}"], ["filter-shape", f":{shape}"]],
        ], exists

    else:
        # Spatial relation query
        color = COLORS[int(jax.random.randint(key, (), 0, len(COLORS)))]
        candidates = [o for o in objects if o["color"] == color]
        if candidates:
            ref = candidates[0]
            left_objs = [o for o in objects if o["x"] < ref["x"] - 0.1]
            if left_objs:
                nearest = max(left_objs, key=lambda o: o["x"])
                return [
                    "query-shape",
                    ["unique", ["left-of", ["unique", ["filter-color", f":{color}"]]]],
                ], nearest["shape"]

    # Fallback: simple existence
    color = COLORS[0]
    exists = any(o["color"] == color for o in objects)
    return ["exists?", ["filter-color", f":{color}"]], exists


def generate_batch(key, batch_size=32, n_objects=4):
    """Generate a batch of (scenes, queries, answers)."""
    keys = jax.random.split(key, batch_size * 2)

    scenes, queries, answers = [], [], []
    for i in range(batch_size):
        scene = generate_scene(keys[i * 2], n_objects)
        query, answer = generate_query(scene, keys[i * 2 + 1])
        scenes.append(scene_to_tensor(scene))
        queries.append(query)
        answers.append(answer)

    return jnp.stack(scenes), queries, answers

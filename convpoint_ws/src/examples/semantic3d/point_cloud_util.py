import numpy as np
import open3d


def _label_to_colors(labels):
    map_label_to_color = {
        #0: [255, 255, 255],  # white
        0: [0, 0, 0],  # black -> unlabeled
        1: [0, 0, 255],  # blue -> man-made terrain
        2: [128, 0, 0],  # maroon -> natural terrain
        3: [255, 0, 255],  # fuchisia -> high vegetation
        4: [0, 128, 0],  # green -> low vegetation
        5: [255, 0, 0],  # red -> buildings
        6: [128, 0, 128],  # purple -> hard scape
        7: [0, 0, 128],  # navy -> scanning artefacts
        8: [128, 128, 0],  # olive -> cars
    }
    return np.array([map_label_to_color[label] for label in labels]).astype(np.int32)


def _label_to_colors_one_hot(labels):
    map_label_to_color = np.array(
        [
            [255, 255, 255],
            [0, 0, 255],
            [128, 0, 0],
            [255, 0, 255],
            [0, 128, 0],
            [255, 0, 0],
            [128, 0, 128],
            [0, 0, 128],
            [128, 128, 0],
        ]
    )
    num_labels = len(labels)
    labels_one_hot = np.zeros((num_labels, 9))
    labels_one_hot[np.arange(num_labels), labels] = 1
    return np.dot(labels_one_hot, map_label_to_color).astype(np.int32)


def colorize_point_cloud(point_cloud, labels):
    if len(point_cloud.points) != len(labels):
        raise ValueError("len(point_cloud.points) != len(labels)")
    if len(labels) < 1e6:
        print("_label_to_colors_one_hot used")
        #colors = _label_to_colors_one_hot(labels)
        colors = _label_to_colors(labels)
    else:
        colors = _label_to_colors(labels)
    # np.testing.assert_equal(colors, colors_v2)
    point_cloud.colors = open3d.utility.Vector3dVector()  # Clear it to save memory
    point_cloud.colors = open3d.utility.Vector3dVector(colors)


def load_labels(label_path):
    # Assuming each line is a valid int
    with open(label_path, "r") as f:
        labels = [float(line) for line in f]
    return np.array(labels, dtype=np.int32)


def write_labels(label_path, labels):
    with open(label_path, "w") as f:
        for label in labels:
            f.write("%d\n" % label)

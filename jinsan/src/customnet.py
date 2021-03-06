custom_model = {
    "input_channel": 3,
    "depth_multiple": 1.0,
    "width_multiple": 1.0,
    "backbone":
        # [repeat, module, args]
        [
            [1, "Conv", [6, 5, 1, 0]],
            [1, "MaxPool", [2]],
            [1, "Conv", [16, 5, 1, 0]],
            [1, "MaxPool", [2]],
            [1, "GlobalAvgPool", []],
            [1, "Flatten", []],
            [1, "Linear", [120, "ReLU"]],
            [1, "Linear", [84, "ReLU"]],
            [1, "Linear", [9]]
        ]
}

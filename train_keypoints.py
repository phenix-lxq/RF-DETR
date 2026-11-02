"""关键点训练：默认读取 dataset/keypoints，保存到 runs/keypoints。"""

import training_common as common


def run(args):
    from rfdetr import RFDETRKeypointPreview
    from rfdetr.config import KeypointTrainConfig, RFDETRKeypointPreviewConfig
    from rfdetr.datasets import infer_coco_keypoint_schema, infer_yolo_keypoint_schema
    from rfdetr.datasets.yolo import find_yolo_data_file

    annotation = args.data / "train" / "_annotations.coco.json"
    schema = (
        infer_coco_keypoint_schema(annotation)
        if annotation.is_file()
        else infer_yolo_keypoint_schema(find_yolo_data_file(args.data))
    )
    print(f"关键点类别：{schema.class_names}；各类别关键点数：{schema.num_keypoints_per_class}")
    model_kwargs, train_kwargs, model_config, train_config = common.prepare_configs(
        args, RFDETRKeypointPreviewConfig, KeypointTrainConfig,
        model_overrides={
            "num_classes": len(schema.class_names),
            "num_keypoints_per_class": schema.num_keypoints_per_class,
        },
        train_overrides={
            "lr_encoder": args.lr,
            "class_names": schema.class_names,
            "keypoint_oks_sigmas": schema.keypoint_oks_sigmas,
            "keypoint_flip_pairs": schema.keypoint_flip_pairs,
        },
    )
    if args.check:
        common.check_dataset_loading(model_config, train_config, args.run_test)
        return

    model = RFDETRKeypointPreview(**model_kwargs)
    model.train(**train_kwargs)


def main(argv=None):
    return common.main("keypoints", run, argv)


if __name__ == "__main__":
    raise SystemExit(main())

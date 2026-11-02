"""实例分割训练：默认读取 dataset/segmentation，保存到 runs/segmentation。"""

import training_common as common


def run(args):
    from rfdetr import RFDETRSegNano
    from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig

    model_kwargs, train_kwargs, model_config, train_config = common.prepare_configs(
        args, RFDETRSegNanoConfig, SegmentationTrainConfig,
    )
    if args.check:
        common.check_dataset_loading(model_config, train_config, args.run_test)
        return

    model = RFDETRSegNano(**model_kwargs)
    model.train(**train_kwargs)


def main(argv=None):
    return common.main("segmentation", run, argv)


if __name__ == "__main__":
    raise SystemExit(main())

"""目标检测训练：默认读取 dataset/detection，保存到 runs/detection。"""

import training_common as common


def run(args):
    from rfdetr import RFDETRNano
    from rfdetr.config import RFDETRNanoConfig, TrainConfig

    model_kwargs, train_kwargs, model_config, train_config = common.prepare_configs(
        args, RFDETRNanoConfig, TrainConfig,
    )
    if args.check:
        common.check_dataset_loading(model_config, train_config, args.run_test)
        return

    model = RFDETRNano(**model_kwargs)
    model.train(**train_kwargs)


def main(argv=None):
    return common.main("detection", run, argv)


if __name__ == "__main__":
    raise SystemExit(main())

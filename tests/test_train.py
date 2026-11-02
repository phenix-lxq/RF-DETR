"""用临时标注数据验证三种任务，不下载模型或执行训练。"""

from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import training_common as train
import train_detection
import train_keypoints
import train_segmentation


ENTRYPOINTS = {
    "detection": train_detection,
    "segmentation": train_segmentation,
    "keypoints": train_keypoints,
}


def make_dataset(root: Path, task: str, dataset_format: str) -> None:
    for split in ("train", "valid", "test"):
        split_dir = root / split
        if dataset_format == "coco":
            split_dir.mkdir(parents=True)
            Image.new("RGB", (64, 64), "white").save(split_dir / "sample.jpg")
            category = {"id": 3, "name": "object"}
            annotation = {
                "id": 1, "image_id": 1, "category_id": 3,
                "bbox": [8, 8, 40, 40], "area": 1600, "iscrowd": 0,
            }
            if task == "segmentation":
                annotation["segmentation"] = [[8, 8, 48, 8, 48, 48, 8, 48]]
            if task == "keypoints":
                category["keypoints"] = ["left_tip", "right_tip"]
                category["skeleton"] = [[1, 2]]
                annotation["keypoints"] = [16, 16, 2, 32, 32, 2]
                annotation["num_keypoints"] = 2
            data = {
                "images": [{"id": 1, "file_name": "sample.jpg", "width": 64, "height": 64}],
                "categories": [category], "annotations": [annotation],
            }
            (split_dir / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")
        else:
            (split_dir / "images").mkdir(parents=True)
            (split_dir / "labels").mkdir()
            Image.new("RGB", (64, 64), "white").save(split_dir / "images" / "sample.jpg")
            label = {
                "detection": "0 .5 .5 .625 .625",
                "segmentation": "0 .125 .125 .75 .125 .75 .75 .125 .75",
                "keypoints": "0 .5 .5 .625 .625 .25 .25 2 .5 .5 2",
            }[task]
            (split_dir / "labels" / "sample.txt").write_text(label + "\n", encoding="utf-8")
    if dataset_format == "yolo":
        data = {
            "names": ["object"], "nc": 1,
            "train": "train/images", "val": "valid/images", "test": "test/images",
        }
        if task == "keypoints":
            data.update(kpt_shape=[2, 3], kpt_names=["left_tip", "right_tip"], flip_idx=[1, 0])
        # JSON 是有效的 YAML；保留真实的 data.yaml 文件名来测试格式识别。
        (root / "data.yaml").write_text(json.dumps(data), encoding="utf-8")


class TrainingEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_all_tasks_and_formats_load_without_initializing_models(self):
        from rfdetr import RFDETRKeypointPreview, RFDETRNano, RFDETRSegNano

        with ExitStack() as stack:
            for model_class in (RFDETRNano, RFDETRSegNano, RFDETRKeypointPreview):
                stack.enter_context(patch.object(
                    model_class, "__init__", side_effect=AssertionError("--check must not initialize a model"),
                ))
            for task in train.TASKS:
                for dataset_format in ("coco", "yolo"):
                    with self.subTest(task=task, dataset_format=dataset_format):
                        root = self.root / task / dataset_format
                        make_dataset(root, task, dataset_format)
                        output = StringIO()
                        with redirect_stdout(output):
                            result = ENTRYPOINTS[task].main([
                                "--data", str(root), "--device", "cpu", "--check", "--run-test",
                                "--output", str(root / "results"),
                            ])
                        self.assertEqual(result, 0)
                        self.assertIn("test 样本加载成功", output.getvalue())
                        self.assertFalse((root / "results").exists())

    def test_training_dispatch_uses_official_api_and_custom_keypoint_schema(self):
        from rfdetr import RFDETRKeypointPreview, RFDETRNano, RFDETRSegNano

        model_classes = (RFDETRNano, RFDETRSegNano, RFDETRKeypointPreview)
        for task, model_class in zip(train.TASKS, model_classes):
            with self.subTest(task=task):
                root = self.root / task
                make_dataset(root, task, "coco")
                with patch.object(model_class, "__init__", return_value=None) as constructor:
                    with patch.object(model_class, "train") as fit, redirect_stdout(StringIO()):
                        result = ENTRYPOINTS[task].main([
                            "--data", str(root), "--device", "cpu", "--epochs", "1",
                        ])
                self.assertEqual(result, 0)
                constructor.assert_called_once()
                fit.assert_called_once()
                self.assertEqual(fit.call_args.kwargs["epochs"], 1)
                self.assertEqual(fit.call_args.kwargs["device"], "cpu")
                self.assertEqual(fit.call_args.kwargs["output_dir"], str(train.PROJECT_DIR / "runs" / task))
                if task == "keypoints":
                    self.assertEqual(constructor.call_args.kwargs["num_keypoints_per_class"], [2])
                    self.assertEqual(fit.call_args.kwargs["keypoint_flip_pairs"], [0, 1])
                    self.assertEqual(fit.call_args.kwargs["lr_encoder"], 1e-5)

    def test_empty_dataset_has_actionable_error(self):
        output = StringIO()
        with redirect_stderr(output):
            for entry in ENTRYPOINTS.values():
                result = entry.main(["--data", str(self.root), "--check"])
                self.assertEqual(result, 1)
        self.assertIn("data.yaml", output.getvalue())

    def test_missing_coco_image_is_rejected(self):
        make_dataset(self.root, "detection", "coco")
        (self.root / "valid" / "sample.jpg").unlink()
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "图片不存在"):
            train.check_coco_annotations(self.root, "detection", False)

    def test_boxes_alone_are_not_accepted_as_coco_segmentation(self):
        make_dataset(self.root, "detection", "coco")
        with self.assertRaisesRegex(ValueError, "segmentation"):
            train.check_coco_annotations(self.root, "segmentation", False)

    def test_keypoint_count_mismatch_is_rejected(self):
        make_dataset(self.root, "keypoints", "coco")
        file = self.root / "train" / "_annotations.coco.json"
        data = json.loads(file.read_text(encoding="utf-8"))
        data["annotations"][0]["keypoints"] = [16, 16, 2]
        file.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "数量"):
            train.check_coco_annotations(self.root, "keypoints", False)

    def test_resume_defaults_to_checkpoint_output_directory(self):
        checkpoint = self.root / "custom_run" / "last.ckpt"
        args = train.parse_args("detection", ["--resume", str(checkpoint)])
        self.assertEqual(args.output, checkpoint.parent)

    def test_default_paths_and_learning_rates_for_each_task(self):
        for task in train.TASKS:
            with self.subTest(task=task):
                args = train.parse_args(task, [])
                self.assertEqual(args.task, task)
                self.assertEqual(args.data, train.PROJECT_DIR / "dataset" / task)
                self.assertEqual(args.output, train.PROJECT_DIR / "runs" / task)
                self.assertEqual(args.lr, 1e-5 if task == "keypoints" else 1e-4)

    def test_resolution_divisor_depends_on_task(self):
        for task in train.TASKS:
            with self.subTest(task=task):
                root = self.root / task
                make_dataset(root, task, "coco")
                args = train.parse_args(task, ["--data", str(root), "--resolution", "385"])
                with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "倍数"):
                    ENTRYPOINTS[task].run(args)


if __name__ == "__main__":
    unittest.main()

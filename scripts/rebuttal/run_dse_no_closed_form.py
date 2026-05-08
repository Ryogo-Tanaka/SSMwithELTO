#!/usr/bin/env python3
"""DSE_no_closed_form training driver.

Subclasses FullExperimentPipeline and applies patch_trainer_for_no_closed_form
to swap df_state / df_obs to SGD subclasses (V_A / V_B as nn.Parameters), add
those Parameters to the phi optimizer's param list, and no-op the closed-form
final operator refit. The realization (CCA), encoder, decoder, and training
schedule are unchanged.

Usage:
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 python scripts/rebuttal/run_dse_no_closed_form.py \
    --config configs/rebuttal/quad_image_reconstruction_no_closed_form.yaml \
    --data data/rkn_quad/quad1_n.npz \
    --output results/rebuttal/dse_ablation/no_closed_form/seed1 \
    --device cuda \
    --seed 42 \
    --skip-analysis
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datetime import datetime
import json
from typing import Any, Dict

import torch

from scripts.run_quadlink_experiment import (  # noqa: E402
    FullExperimentPipeline,
    load_experiment_config,
    parse_args,
    set_random_seed,
)
from src.training.two_stage_trainer import TwoStageTrainer  # noqa: E402
from src.utils.gpu_utils import select_device  # noqa: E402

from src.rebuttal.two_stage_trainer_patch import (  # noqa: E402
    patch_trainer_for_no_closed_form,
)


class NoClosedFormPipeline(FullExperimentPipeline):
    """Variant pipeline applying patch_trainer_for_no_closed_form."""

    def step_2_training_execution(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        print("\n" + "=" * 5)
        print("Step 2: Training (DSE_no_closed_form variant)")
        print("=" * 5)

        start_time = datetime.now()

        experiment_mode = data_dict.get("experiment_mode", "reconstruction")
        print(f"Training mode: {experiment_mode}")

        use_kalman = self.config.get("training", {}).get("use_kalman_filtering", False)
        print(f"Kalman filtering: {'enabled' if use_kalman else 'disabled'}")

        if "training" in self.config:
            self.config["training"]["experiment_mode"] = experiment_mode

        trainer = TwoStageTrainer(
            config=self.config,
            device=self.device,
            output_dir=str(self.output_dir),
            use_kalman_filtering=use_kalman,
        )

        # === DSE_no_closed_form patch: V_A/V_B as Parameters, skip closed-form refit ===
        patch_trainer_for_no_closed_form(trainer)

        print("Starting integrated training...")
        if experiment_mode == "target_prediction":
            target_train = self._extract_targets_from_dict(data_dict, "train")
            target_val = (
                self._extract_targets_from_dict(data_dict, "val")
                if data_dict.get("val") is not None
                else None
            )
            integrated_results = trainer.train_integrated(
                Y_train=data_dict["train"],
                Y_val=data_dict["val"],
                target_train=target_train,
                target_val=target_val,
            )
        else:
            integrated_results = trainer.train_integrated(
                Y_train=data_dict["train"],
                Y_val=data_dict["val"],
            )

        total_elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Integrated training done ({total_elapsed:.1f}s)")

        training_results = {
            "integrated": integrated_results,
            "phase1_metrics": integrated_results["phase1_metrics"],
            "phase2_losses": integrated_results["phase2_losses"],
            "integrated_metrics": integrated_results["integrated_metrics"],
            "total_time": total_elapsed,
            "use_kalman": use_kalman,
        }

        self._plot_training_progress(training_results)

        results_path = self.output_dir / "logs" / "training_results.json"
        with open(results_path, "w") as f:
            serializable_results = self._make_json_serializable(training_results)
            json.dump(serializable_results, f, indent=2)

        self.experiment_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": "integrated_training_complete",
                "total_time": total_elapsed,
                "epochs": len(integrated_results.get("integrated_metrics", [])),
                "use_kalman": use_kalman,
            }
        )

        return {"trainer": trainer, "results": training_results}


def main() -> int:
    args = parse_args()

    print("Experiment Start (DSE_no_closed_form)")
    print("=" * 5)

    if args.seed is not None:
        set_random_seed(args.seed)

    config = load_experiment_config(args.config)

    if args.use_kalman:
        config.setdefault("training", {})["use_kalman_filtering"] = True

    device = torch.device(args.device) if args.device else select_device()
    output_dir = Path(args.output)

    pipeline = NoClosedFormPipeline(config, output_dir, device)

    try:
        data_dict = pipeline.step_1_data_loading(args.data)
        training_result = pipeline.step_2_training_execution(data_dict)

        if not args.skip_analysis:
            pipeline.step_3_model_analysis(training_result["trainer"], data_dict)

        pipeline.finalize_experiment(training_result["trainer"])

        print("\nDSE_no_closed_form experiment completed successfully")
        print(f"Results: {output_dir}")

    except Exception as e:
        print(f"\nDSE_no_closed_form experiment error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

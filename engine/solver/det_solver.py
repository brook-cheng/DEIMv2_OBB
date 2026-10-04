"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import os
import time
import json
import datetime
import math

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler
from .early_stopping import (
    EarlyStoppingConfig,
    EarlyStoppingState,
    RESTORED_METRIC_TOLERANCE,
)


class DetSolver(BaseSolver):

    def _load_stage_checkpoint(self, epoch: int) -> None:
        """Refresh model/EMA from best_stg1.pth at stop_epoch (all ranks).

        best_stg1.pth is written by rank0 only (save_on_master at the previous
        epoch's validation); every rank loads it here at stop_epoch. Without a
        barrier the non-writing ranks can read the file while rank0 is still
        streaming it -> EOFError on torch.load (server 20260820_182623,
        rank1). Same barrier pattern as the best.pth restore at the end of
        fit().
        """
        ckpt = str(self.output_dir / "best_stg1.pth")
        if dist_utils.is_dist_available_and_initialized():
            torch.distributed.barrier()
        saved_epoch = self.last_epoch
        if os.path.exists(ckpt):
            self.load_resume_state(ckpt)
        else:
            print(f"[warn] {ckpt} missing at stop_epoch {epoch} — skipping EMA refresh")
        self.last_epoch = saved_epoch
        self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
        print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")
        if self.early_stopping is not None:
            self.early_stopping.reset_patience()
            print(
                f"Early-stopping patience reset at stage transition "
                f"(epoch {epoch}); global best preserved "
                f"(best_epoch={self.early_stopping.best_epoch}, "
                f"best_mAP50_95={self.early_stopping.best_observed_metric:.4f})"
            )

    def _load_checkpoint_fresh_schedule(self, path: str) -> None:
        """-r <file>: checkpoint training — load state, train a FRESH schedule.

        The checkpoint seeds weights/EMA/optimizer (+kendall state, restored
        by the deferred load after kendall construction), but the epoch
        schedule restarts from scratch: last_epoch=-1, fresh early stopping,
        fresh LR warmup. A checkpoint saved at epoch N can therefore seed a
        new run of ANY length; true interruption-continuation is `-r auto`.
        """
        print(f"Resume checkpoint from {path}")
        self.load_resume_state(path)
        self.last_epoch = -1
        self._init_early_stopping()
        self.cfg._lr_warmup_scheduler = None
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler
        print(
            f"[resume] fresh schedule: {self.cfg.epoches} epochs "
            f"from the loaded checkpoint"
        )

    def _maybe_auto_recover(self) -> None:
        """Auto-resume from output_dir checkpoints when recovery is enabled.

        Tier 1: ``last.pth`` (written atomically after every epoch's
        validation). Tier 2: newest ``checkpoint{NNNN}.pth`` snapshot. A
        fresh start only when nothing usable exists; an unusable tier falls
        through instead of aborting. With dist initialized, a barrier
        precedes the loads so every rank resumes from the same tier.
        """
        if self.output_dir is None:
            return
        candidates = [self.output_dir / "last.pth"] + sorted(
            self.output_dir.glob("checkpoint*.pth"), reverse=True
        )
        if dist_utils.is_dist_available_and_initialized():
            torch.distributed.barrier()
        for cand in candidates:
            if not cand.exists():
                continue
            try:
                self.load_resume_state(str(cand))
            except Exception as e:
                print(f"[recovery] {cand.name} unusable ({e}); trying next tier")
                continue
            print(f"[recovery] resumed from {cand} at epoch {self.last_epoch}")
            return
        if any(cand.exists() for cand in candidates):
            print("[recovery] no usable checkpoint — starting fresh")

    def fit(
        self,
    ):
        self._init_early_stopping()
        # Fit-path resume load is deferred past kendall construction so
        # kendall/kendall_optimizer state restores too; `--test-only` (eval
        # path) still loads immediately in BaseSolver.
        self._defer_fit_resume = True
        self.train()
        self._defer_fit_resume = False
        args = self.cfg

        # Kendall Uncertainty Weighting：可学习 σ² 自动平衡 loss 量纲
        # weight_dict 作为固定先验乘子 p_i = w_i / mean(w)，全程不洗掉。
        # After self.train() (materializes self.criterion for the prior) and
        # BEFORE the recovery load, so self.kendall / self.kendall_optimizer
        # exist as attributes and their state restores from the checkpoint.
        self.kendall = None
        self.kendall_optimizer = None
        kw_cfg = self.cfg.yaml_cfg.get("KendallWeighting", {})
        if kw_cfg.get("enabled", False):
            from .kendall import KendallWeighting

            loss_names = kw_cfg.get(
                "loss_names", ["loss_mal", "loss_bbox", "loss_kld", "loss_fgl"]
            )
            # 从 criterion 配置读取 weight_dict，计算再归一先验 p_i
            wd = self.criterion.weight_dict
            raw_prior = [wd.get(n, 1.0) for n in loss_names]
            mean_p = sum(raw_prior) / len(raw_prior)
            prior = [p / mean_p for p in raw_prior]

            self.kendall = KendallWeighting(
                loss_names=loss_names,
                init_log_sigma=kw_cfg.get("init_log_sigma", 0.0),
                prior=prior,
            )
            self.kendall_optimizer = torch.optim.Adam(
                [self.kendall.log_sigma],
                lr=kw_cfg.get("sigma_lr", 0.001),
            )
            print(
                f"[KendallWeighting] enabled — "
                f"prior={[f'{p:.3f}' for p in prior]}"
            )

        if getattr(self.cfg, "recovery", False):
            self._maybe_auto_recover()
        elif self.cfg.resume:
            self._load_checkpoint_fresh_schedule(self.cfg.resume)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)

        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i in [194, 195]:
                print(f"Index {i}: {name} - requires_grad: {param.requires_grad}")

        comet_exp = getattr(self.cfg, "_comet_experiment", None)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                args.lr_gamma,
                iter_per_epoch,
                total_epochs=args.epoches,
                warmup_iter=args.warmup_iter,
                flat_epochs=args.flat_epoch,
                no_aug_epochs=args.no_aug_epoch,
            )
            self.self_lr_scheduler = True
        n_parameters = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        print(f"number of trainable parameters: {n_parameters}")

        n_parameters = sum(
            [p.numel() for p in self.model.parameters() if not p.requires_grad]
        )
        print(f"number of non-trainable parameters: {n_parameters}")

        top1 = 0
        stop_early = False
        best_stat = {
            "epoch": -1,
        }
        # evaluate again before resume training
        box_mode = getattr(
            self.postprocessor,
            "box_mode",
            "hbb",
        )
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                box_mode=box_mode,
            )

            if box_mode == "hbb":
                for k in test_stats:
                    best_stat["epoch"] = self.last_epoch
                    best_stat[k] = test_stats[k][0]
                    top1 = test_stats[k][0]
            else:
                v = test_stats.get("mAP50_95", 0)
                best_stat["epoch"] = self.last_epoch
                best_stat["mAP50_95"] = v
                top1 = v
                print(f"best_stat: {best_stat}")
                if (
                    self.early_stopping is not None
                    and self.early_stopping.best_epoch < 0
                ):
                    self.early_stopping.initialize_from_metric(
                        v, self.last_epoch
                    )
                    print(
                        f"Initialized early-stopping state from pre-resume "
                        f"EMA validation: best_mAP50_95={v:.4f} "
                        f"at epoch {self.last_epoch}"
                    )

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        if start_epoch >= args.epoches:
            print(
                f"[fit] nothing to train: start_epoch {start_epoch} >= "
                f"epoches {args.epoches} — the restored checkpoint "
                f"(epoch {self.last_epoch}) is already past the target; "
                f"raise epoches to continue, or use -r <file> for a fresh "
                f"schedule"
            )

        _max_optimizer_steps = args.yaml_cfg.get("max_optimizer_steps")
        _fail_on_zero_grad = args.yaml_cfg.get("fail_on_zero_grad", False)

        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self._load_stage_checkpoint(epoch)

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                use_amp=args.yaml_cfg.get("use_amp", False),
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                comet_exp=comet_exp,
                comet_step=epoch,
                kendall=self.kendall,
                kendall_optimizer=self.kendall_optimizer,
                max_optimizer_steps=_max_optimizer_steps,
                fail_on_zero_grad=_fail_on_zero_grad,
                output_dir=self.output_dir,
                nan_max_events=args.yaml_cfg.get("nan_max_events", 10),
                gradient_accumulation_steps=getattr(
                    args, "gradient_accumulation_steps", 1
                ),
            )

            if train_stats.pop("_step_cap_reached", False):
                print(f"[Diagnostic] step cap reached at epoch {epoch}. Stopping.")
                self._diagnostic_exit = True
                break

            if not self.self_lr_scheduler:  # update by epoch
                if (
                    self.lr_warmup_scheduler is None
                    or self.lr_warmup_scheduler.finished()
                ):
                    self.lr_scheduler.step()

            self.last_epoch += 1

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                comet_exp=comet_exp,
                comet_step=epoch,
                box_mode=box_mode,
            )

            if box_mode == "hbb":
                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)
                    if k in best_stat:
                        best_stat["epoch"] = (
                            epoch
                            if test_stats[k][0] > best_stat[k]
                            else best_stat["epoch"]
                        )
                        best_stat[k] = max(best_stat[k], test_stats[k][0])
                    else:
                        best_stat["epoch"] = epoch
                        best_stat[k] = test_stats[k][0]
                    if best_stat[k] > top1:
                        best_stat_print["epoch"] = epoch
                        top1 = best_stat[k]
                        if (
                            self.output_dir
                            and epoch >= self.train_dataloader.collate_fn.stop_epoch
                        ):
                            dist_utils.save_on_master_atomic(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
            else:
                # Log all metrics to writer
                if self.writer and dist_utils.is_main_process():
                    for k, v in test_stats.items():
                        self.writer.add_scalar(f"Test/{k}", v, epoch)
                # Checkpoint selection: mAP@0.5:0.95 (same standard as HBB's AP@0.5:0.95)
                v = test_stats.get("mAP50_95", 0)
                if v > best_stat.get("mAP50_95", -1.0):
                    best_stat["epoch"] = epoch
                    best_stat["mAP50_95"] = v
                if v > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = v
                    if (
                        self.output_dir
                        and epoch >= self.train_dataloader.collate_fn.stop_epoch
                    ):
                        dist_utils.save_on_master_atomic(
                            self.state_dict(), self.output_dir / "best_stg2.pth"
                        )
                    else:
                        dist_utils.save_on_master_atomic(
                            self.state_dict(), self.output_dir / "best_stg1.pth"
                        )

                # Early-stopping update (OBB): monitors EMA mAP50_95, saves
                # best.pth on observed improvement, computes the local stop
                # decision. The DDP broadcast of the decision happens AFTER the
                # epoch's log entry and checkpoint writes (see Edit 4.7).
                _, stop_early = self._update_early_stopping(v, epoch)

            if box_mode == "hbb":
                best_stat_print[k] = max(best_stat[k], top1)
                print(f"best_stat: {best_stat_print}")  # global best

                if best_stat["epoch"] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master_atomic(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master_atomic(
                            self.state_dict(), self.output_dir / "best_stg1.pth"
                        )

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {"epoch": -1}
                    self.ema.decay -= 0.0001
                    saved_epoch = self.last_epoch
                    if dist_utils.is_dist_available_and_initialized():
                        torch.distributed.barrier()
                    self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                    self.last_epoch = saved_epoch
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")
            else:
                if self.writer and dist_utils.is_main_process():
                    self.writer.add_scalar(
                        f"Test/best_mAP", best_stat.get("mAP", 0), epoch
                    )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

            # Checkpoint after validation so last.pth carries the exact
            # early-stopping state (interruption-resume continuity).
            if self.output_dir:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(
                        self.output_dir / f"checkpoint{epoch:04}.pth"
                    )
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master_atomic(self.state_dict(), checkpoint_path)

            # Design ordering: the loop exits only after the epoch's log entry
            # and checkpoint writes have completed; rank 0 then broadcasts the
            # stop decision so every rank breaks together.
            stop_early = self._sync_early_stopping(stop_early)

            if stop_early:
                print(
                    f"[EarlyStopping] stopping at epoch {epoch} after "
                    f"{self.early_stopping_config.patience} epochs without "
                    f"significant improvement."
                )
                break

        stop_reason = "max_epochs"
        if getattr(self, "_diagnostic_exit", False):
            stop_reason = "diagnostic"
        elif stop_early:
            stop_reason = "early_stopping"
        stop_epoch = self.last_epoch
        print(f"[Training] exit reason: {stop_reason} at epoch {stop_epoch}")
        self._finalize_training(stop_reason, stop_epoch, box_mode)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def _init_early_stopping(self):
        """Build early-stopping config and state from cfg.yaml_cfg.

        Runs before self.train() so resume loading can restore the persisted
        early-stopping state. Missing/disabled config leaves
        self.early_stopping = None (current training behavior preserved).
        """
        self.early_stopping_config = EarlyStoppingConfig.from_yaml(self.cfg.yaml_cfg)
        self.early_stopping = (
            EarlyStoppingState() if self.early_stopping_config.enabled else None
        )
        self._diagnostic_exit = False

    def _sync_early_stopping(self, stop_early):
        """Broadcast early-stopping state and stop decision to all ranks (DDP).

        Rank 0 already updated the state. Non-main ranks overwrite their local
        state from the broadcast payload. Returns the agreed stop decision.
        Safe when early stopping is disabled (returns stop_early unchanged).
        """
        if self.early_stopping is None:
            return stop_early
        if not dist_utils.is_dist_available_and_initialized():
            return stop_early
        if dist_utils.is_main_process():
            payload = [self.early_stopping.state_dict(), stop_early]
        else:
            payload = [None, False]
        torch.distributed.broadcast_object_list(payload, src=0)
        if not dist_utils.is_main_process():
            self.early_stopping.load_state_dict(payload[0])
            stop_early = bool(payload[1])
        return stop_early

    def _update_early_stopping(self, metric, epoch):
        """Update ES state on rank 0, save best.pth on observed improvement.

        Returns (should_save_best, should_stop) with only the local stop
        decision; the caller broadcasts it via ``_sync_early_stopping`` AFTER
        the epoch's log entry and checkpoint writes (design ordering).
        """
        if self.early_stopping is None:
            return False, False
        should_save_best = False
        should_stop = False
        if dist_utils.is_main_process():
            should_save_best = self.early_stopping.update(
                metric, epoch, self.early_stopping_config.min_delta
            )
            if should_save_best and self.output_dir:
                dist_utils.save_on_master_atomic(
                    self.state_dict(), self.output_dir / "best.pth"
                )
            should_stop = self.early_stopping.should_stop(
                epoch,
                self.early_stopping_config.min_epochs,
                self.early_stopping_config.patience,
            )
        return should_save_best, should_stop

    def _finalize_training(self, stop_reason, stop_epoch, box_mode):
        """Restore best.pth for normal exits, re-validate, write metadata.

        Capture stop/best metadata BEFORE loading best.pth so the record is
        independent of the restored checkpoint's last_epoch. Never restores on
        the diagnostic path. Never overwrites last.pth.
        """
        if self.early_stopping is None:
            return
        best_epoch = self.early_stopping.best_epoch
        best_metric = self.early_stopping.best_observed_metric
        meta_best = float(best_metric) if math.isfinite(best_metric) else None
        meta = {
            "stop_reason": stop_reason,
            "stop_epoch": stop_epoch,
            "best_epoch": best_epoch,
            "best_mAP50_95": meta_best,
            "restored_mAP50_95": None,
            "epochs_after_best": max(0, stop_epoch - best_epoch),
            "restore_skipped": False,
            "restore_match": None,
        }
        can_restore = (
            self.early_stopping_config.restore_best
            and stop_reason != "diagnostic"
            and self.output_dir is not None
            and (self.output_dir / "best.pth").exists()
        )
        if can_restore:
            if dist_utils.is_dist_available_and_initialized():
                torch.distributed.barrier()
            self.load_resume_state(str(self.output_dir / "best.pth"))
            module = self.ema.module if self.ema else self.model
            restored_stats, _ = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                box_mode=box_mode,
            )
            restored = restored_stats.get("mAP50_95", 0)
            meta["restored_mAP50_95"] = restored
            meta["restore_match"] = (
                abs(restored - best_metric) <= RESTORED_METRIC_TOLERANCE
            )
            if not meta["restore_match"]:
                print(
                    f"[EarlyStopping] WARNING restored mAP50_95={restored:.4f} "
                    f"differs from best {best_metric:.4f} by more than "
                    f"{RESTORED_METRIC_TOLERANCE}"
                )
        else:
            meta["restore_skipped"] = True

        if self.output_dir and dist_utils.is_main_process():
            with (self.output_dir / "final_run_meta.json").open("w") as f:
                json.dump(meta, f, indent=2)
        print(f"[Training] final metadata: {meta}")

    def val(self):
        self.eval()

        if self.ema:
            module = self.ema.module
        else:
            module = self.model
        box_mode = getattr(
            self.postprocessor,
            "box_mode",
            "hbb",
        )
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            box_mode=box_mode,
        )

        if self.output_dir and box_mode == "hbb":
            dist_utils.save_on_master_atomic(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return

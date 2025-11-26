from time import time
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Trainer:
	def __init__(
		self,
		model,
		criterion,
		optimizer,
		train_loader,
		val_loader,
		scheduler,
		device,
		psnr,
		ssim,
		lpips,
		use_amp=False,
		use_grad_scaler=False,
		val_interval=1,
		save_interval=50,
		checkpoint_prefix=None,
	):
		self.device = device
		self.device_type = "cuda" if str(device).startswith("cuda") else "cpu"
		self.model = model.to(device)
		self.criterion = criterion
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.scheduler = scheduler
		self.psnr = psnr
		self.ssim = ssim
		self.lpips = lpips
		self.amp = use_amp
		self.scaler = torch.amp.GradScaler(enabled=(self.device_type == "cuda" and use_grad_scaler))
		self.best_loss = float("inf")
		self.best_state = None
		self.checkpoint_prefix = checkpoint_prefix or self.model.__class__.__name__
		self.visual_dir = "results"
		self.mid_title = "Model Output"
		self.val_interval = val_interval
		self.save_interval = save_interval

	def autocast(self):
		return torch.amp.autocast(
			device_type=self.device_type,
			dtype=torch.bfloat16,
			enabled=(self.amp and self.device_type == "cuda"),
		)

	def _checkpoint_path(self, suffix):
		os.makedirs("checkpoints", exist_ok=True)
		return os.path.join("checkpoints", f"{self.checkpoint_prefix}_{suffix}.pth")

	def _save_state(self, epoch, loss, save_last=True):
		state = {
			"model": self.model.state_dict(),
			"optimizer": self.optimizer.state_dict(),
			"epoch": epoch,
			"loss": loss,
		}
		# if save_last:
		# 	state_last = {k: v for k, v in state.items()}
		# 	torch.save(state_last, self._checkpoint_path("last"))
		# if loss < self.best_loss:
		# 	self.best_loss = loss
		# 	state_best = {k: v for k, v in state.items()}
		# 	torch.save(state_best, self._checkpoint_path("best"))
		# 	self.best_state = state_best

	def _step_scheduler(self):
		if self.scheduler is not None:
			self.scheduler.step()

	def _train_loop(self, start, epochs, step_fn):
		self.model.train()
		print(f"Starting training from epoch {start}\n")
		for epoch in range(start, epochs):
			t0 = time()
			total_loss = 0.0
			total_metrics = {}
			num_batches = len(self.train_loader)
			step_count = 0
			for step_count, batch in enumerate(self.train_loader, start=1):
				self.optimizer.zero_grad()
				with self.autocast():
					loss, metrics = step_fn(batch)

				self.scaler.scale(loss).backward()
				self.scaler.step(self.optimizer)
				self.scaler.update()
				self._step_scheduler()
				total_loss += loss.item()
				for key, value in metrics.items():
					total_metrics[key] = total_metrics.get(key, 0) + value
				bar_len = 50
				progress = bar_len * step_count // num_batches
				print("█" * progress + "-" * (bar_len - progress), end=" \r", flush=True)

			if step_count == 0:
				continue

			avg_loss = total_loss / step_count
			avg_metrics = {key: value / step_count for key, value in total_metrics.items()}
			tT = time() - t0
			print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.6f}, Time: {tT:.3f} seconds" + "".join(f", {key.upper()}: {value:.3f}" for key, value in avg_metrics.items()))

			save_last = (epoch + 1) % self.save_interval == 0 or (epoch + 1) == epochs
			self._save_state(epoch, avg_loss, save_last=save_last)

			if epoch % self.val_interval == 0 and epoch != 0:
				self.val(epoch)

		if self.best_state is not None:
			self.model.load_state_dict(self.best_state["model"])
			print(f"Loaded best model from epoch {self.best_state['epoch'] + 1} with loss {self.best_state['loss']:.6f}")

	def val(self, epoch):
		total_metrics = {}
		num_batches = 0
		for i, batch in enumerate(self.val_loader):
			lrs, hrs = batch
			metrics = self.infer(lrs, hrs, suffix=f"val_{epoch}", save_img=(i == 0))
			
			for k, v in metrics.items():
				total_metrics[k] = total_metrics.get(k, 0) + v
			num_batches += 1
		
		if num_batches > 0:
			avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
			print(f"Validation Epoch [{epoch}] - " + ", ".join(f"{k}: {v:.3f}" for k, v in avg_metrics.items()))
		
		torch.cuda.empty_cache()

	def infer(self, lrs, hrs, out=None, suffix="test", save_img=True):
		prev_mode = self.model.training
		lrs = lrs.to(self.device)
		hrs = hrs.to(self.device)

		if out is None:
			self.model.eval()
			with torch.no_grad():
				out = self._generate_output(lrs, hrs)
		else:
			out = out.to(self.device)
		out = out.clamp(0, 1)

		psnr_val = self.psnr(out, hrs).item()
		ssim_val = self.ssim(out, hrs).item()
		lpips_val = self.lpips(out, hrs).item()

		if save_img:
			out_cpu = out.detach().cpu()
			lr_cpu = lrs.detach().cpu()
			hr_cpu = hrs.detach().cpu()

			batch_size = out_cpu.shape[0]
			fig, axes = plt.subplots(3, batch_size, figsize=(max(batch_size * 6, 12), 18), squeeze=False)
			fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.05)
			
			for idx in range(batch_size):
				axes[0, idx].imshow(lr_cpu[idx].permute(1, 2, 0).float())
				axes[0, idx].set_title(f"LR #{idx}")
				axes[1, idx].imshow(out_cpu[idx].permute(1, 2, 0).float())
				axes[1, idx].set_title(f"{self.mid_title} #{idx}")
				axes[2, idx].imshow(hr_cpu[idx].permute(1, 2, 0).float())
				axes[2, idx].set_title(f"HR #{idx}")
			for row in range(3):
				for col in range(batch_size):
					axes[row, col].axis("off")
					axes[row, col].title.set_fontsize(18)
			axes[1, 0].set_ylabel(
				f"PSNR: {psnr_val:.3f}\nSSIM: {ssim_val:.3f}\nLPIPS: {lpips_val:.3f}",
				fontsize=18,
			)

			os.makedirs(self.visual_dir, exist_ok=True)
			plt.savefig(os.path.join(self.visual_dir, f"{self.checkpoint_prefix}_{suffix}.png"))
			plt.close(fig)
			del out_cpu, lr_cpu, hr_cpu

		if prev_mode:
			self.model.train()
		
		metrics = {"PSNR": psnr_val, "SSIM": ssim_val, "LPIPS": lpips_val}
		del lrs, hrs, out
		return metrics

	def _generate_output(self, lrs=None, hrs=None):
		raise NotImplementedError


class INRTrainer(Trainer):
	def __init__(
		self,
		model,
		criterion,
		optimizer,
		train_loader,
		val_loader,
		scheduler,
		device,
		psnr,
		ssim,
		lpips,
		use_amp=False,
		use_grad_scaler=False,
		val_interval=1,
		save_interval=1,
		checkpoint_prefix=None,
	):
		super().__init__(
			model,
			criterion,
			optimizer,
			train_loader,
			val_loader,
			scheduler,
			device,
			psnr,
			ssim,
			lpips,
			use_amp,
			use_grad_scaler,
			val_interval,
			save_interval,
			checkpoint_prefix,
		)

		self.visual_dir = "results/inr"
		self.mid_title = "INR Output"
		self.save_interval = save_interval

	def _create_query_grid(self, batch_size, height, width):
		y = torch.linspace(0.0, 1.0, steps=height, device=self.device)
		x = torch.linspace(0.0, 1.0, steps=width, device=self.device)
		grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
		coords = torch.stack((grid_x, grid_y), dim=-1)
		coords = coords.view(1, height * width, 2).expand(batch_size, -1, -1)
		return coords

	def train(self, start, epochs):
		def step_fn(batch):
			lrs, hrs = batch
			lrs = lrs.to(self.device)
			hrs = hrs.to(self.device)
			hrs_out = self._generate_output(lrs, hrs)
			recon_loss = self.criterion(hrs_out, hrs)
			return recon_loss, {}

		self._train_loop(start, epochs, step_fn)

	def _generate_output(self, lrs, hrs):
		grid = self._create_query_grid(lrs.shape[0], hrs.shape[2], hrs.shape[3])
		preds = self.model(lrs, grid)
		return preds.transpose(1, 2).reshape_as(hrs)

class DIPTrainer(Trainer):
	def __init__(
		self,
		model,
		criterion,
		optimizer,
		train_loader,
		val_loader,
		scheduler,
		device,
		psnr,
		ssim,
		lpips,
		use_amp=False,
		use_grad_scaler=False,
		val_interval=50,
		save_interval=50,
		checkpoint_prefix=None,
	):
		super().__init__(
			model,
			criterion,
			optimizer,
			train_loader,
			val_loader,
			scheduler,
			device,
			psnr,
			ssim,
			lpips,
			use_amp,
			use_grad_scaler,
			val_interval,
			save_interval,
			checkpoint_prefix,
		)

		self.visual_dir = "results/dip"
		self.mid_title = "DIP Output"
		self.save_interval = save_interval

	def train(self, start, epochs):
		def step_fn(batch):
			lrs, hrs = batch
			lrs = lrs.to(self.device)
			hrs = hrs.to(self.device)
			hrs_out = self._generate_output()
			lrs_out = F.interpolate(hrs_out, size=lrs.shape[-2:], mode="bicubic")
			loss = self.criterion(lrs_out, lrs)

			return loss, {}

		self._train_loop(0, epochs, step_fn)

	def _generate_output(self, lrs=None, hrs=None):
		return self.model()


class DiffusionTrainer(Trainer):
	def __init__(
		self,
		diff,
		criterion,
		optimizer,
		scheduler,
		train_loader,
		val_loader,
		device,
		psnr,
		ssim,
		lpips,
		use_amp=False,
		use_grad_scaler=False,
		val_interval=2,
		save_interval=2,
		checkpoint_prefix=None,
	):
		super().__init__(
			diff,
			criterion,
			optimizer,
			train_loader,
			val_loader,
			scheduler,
			device,
			psnr,
			ssim,
			lpips,
			use_amp,
			use_grad_scaler,
			val_interval,
			save_interval,
		)

		self.diff = self.model
		self.checkpoint_prefix = checkpoint_prefix or self.diff.model.__class__.__name__
		self.visual_dir = "results/diffusion"
		self.mid_title = "Diffusion Output"

	def train(self, start, epochs):
		def step_fn(batch):
			lrs, hrs = batch
			lrs = lrs.to(self.device)
			hrs = hrs.to(self.device)
			self.diff.lrs_up = F.interpolate(lrs, scale_factor=8, mode="bicubic")
			_, e, pred_e = self.diff(hrs, self.diff.lrs_up)
			return self.criterion(pred_e, e), {}

		self._train_loop(start, epochs, step_fn)
		if self.best_state is not None:
			self.diff.load_state_dict(self.best_state["model"])
			print(f"Loaded best model from epoch {self.best_state['epoch'] + 1} with loss {self.best_state['loss']:.6f}")

	def _generate_output(self, lrs, hrs):
		return self.diff.sample(lrs.shape[0], lr_up=F.interpolate(lrs, scale_factor=8, mode="bicubic"))
	
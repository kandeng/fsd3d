"""Entry point: train both paradigms, save checkpoints & logs to workspace/.

Usage:
    cd src/fsd3d/decoder
    python main.py
"""

import sys
import os
import datetime
import torch

from fsd3d.decoder.context import ContextAssembler
from fsd3d.decoder.transformer import FSD3DTransformerDecoder
from fsd3d.decoder.action_head import ActionHead
from fsd3d.decoder.autoregressive import AutoregressiveWrapper
from fsd3d.training.cfm import train_cfm
from fsd3d.training.autoregressive import train_autoregressive

WORKSPACE = os.path.join(os.path.dirname(__file__), "workspace")


class TeeLogger:
    """Log to both console and a file simultaneously."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def main():
    seed = 42
    torch.manual_seed(seed)

    os.makedirs(WORKSPACE, exist_ok=True)

    log_path = os.path.join(WORKSPACE, "training_log.txt")
    tee = TeeLogger(log_path)
    sys.stdout = tee

    print(f"FSD3D Decoder Training — {datetime.datetime.now().isoformat()}")
    print(f"Workspace: {os.path.abspath(WORKSPACE)}")
    print(f"Seed: {seed}")
    print()

    # 1. Build dataset
    assembler = ContextAssembler()
    z1, context = assembler.build_synthetic_data()
    print(f"z1 shape:      {z1.shape}  (left + right dodge)")
    print(f"context shape: {context.shape}")
    print()

    # 2. Train CFM model
    print("=" * 60)
    print("Training CFM model …")
    print("=" * 60)
    cfm_decoder = FSD3DTransformerDecoder()
    cfm_action_head = ActionHead()
    cfm_decoder, cfm_action_head = train_cfm(
        cfm_decoder, cfm_action_head, z1, context,
        epochs=5000, batch_size=64, lr=1e-3, seed=seed, log_interval=500,
    )

    # 3. Train AR model
    print()
    print("=" * 60)
    print("Training Autoregressive model …")
    print("=" * 60)
    ar_decoder = FSD3DTransformerDecoder()
    ar_action_head = ActionHead()
    ar_wrapper = AutoregressiveWrapper(ar_decoder, ar_action_head)
    ar_wrapper = train_autoregressive(
        ar_wrapper, z1, context,
        epochs=300, batch_size=64, lr=1e-3, seed=seed, log_interval=50,
    )

    # 4. Save checkpoints
    print()
    print("=" * 60)
    print("Saving checkpoints …")
    print("=" * 60)

    cfm_dec_path = os.path.join(WORKSPACE, "cfm_decoder.pt")
    cfm_ah_path  = os.path.join(WORKSPACE, "cfm_action_head.pt")
    ar_path      = os.path.join(WORKSPACE, "ar_wrapper.pt")
    z1_path      = os.path.join(WORKSPACE, "z1.pt")
    ctx_path     = os.path.join(WORKSPACE, "context.pt")

    torch.save(cfm_decoder.state_dict(), cfm_dec_path)
    torch.save(cfm_action_head.state_dict(), cfm_ah_path)
    torch.save(ar_wrapper.state_dict(), ar_path)
    torch.save(z1, z1_path)
    torch.save(context, ctx_path)

    print(f"  {cfm_dec_path}")
    print(f"  {cfm_ah_path}")
    print(f"  {ar_path}")
    print(f"  {z1_path}")
    print(f"  {ctx_path}")

    # 5. Quick validation
    print()
    print("=" * 60)
    print("Quick validation …")
    print("=" * 60)

    from fsd3d.inference.cfm import infer_cfm_euler
    from fsd3d.inference.autoregressive import infer_autoregressive
    import torch.nn.functional as F

    cfm_decoder.eval()
    cfm_action_head.eval()
    cfm_final, cfm_traj = infer_cfm_euler(cfm_decoder, cfm_action_head, context, euler_steps=10, seed=99)
    cfm_mses = [F.mse_loss(cfm_final[0], z1[i]).item() for i in range(z1.size(0))]
    cfm_mse = min(cfm_mses)
    cfm_idx = cfm_mses.index(cfm_mse)
    print(f"  CFM final MSE vs z1[{cfm_idx}]: {cfm_mse:.6f}")

    ar_wrapper.eval()
    ar_clean = ar_wrapper.generate(context.expand(1, -1, -1), horizon=16, noise_step=-1, noise_sigma=0.0)
    ar_clean_mses = [F.mse_loss(ar_clean[0], z1[i]).item() for i in range(z1.size(0))]
    print(f"  AR clean MSE:  {min(ar_clean_mses):.6f}")

    print()
    print("Training complete. To visualise, run:")
    print("    cd src/fsd3d/decoder && python visualize.py")

    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()

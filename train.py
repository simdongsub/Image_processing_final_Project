"""동일 조건 학습, 고정 분할 기록, 에폭 단위 재개."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from dataset import create_lol_dataloaders
from model import MODES, EnhancementConfig, ZeroDCE
from loss import ZeroDCELoss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--low-dir', required=True)
    p.add_argument('--high-dir')
    p.add_argument('--test-low-dir', required=True, help='공식 평가 low 폴더; 무작위 test 분할을 방지')
    p.add_argument('--test-high-dir', required=True)
    p.add_argument('--mode', choices=MODES, default='none')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=10, help='예비 실험용 초깃값; 본 실험 전 공통 확정')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--crop-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--max_hours', '--max-hours', type=float, default=3)
    p.add_argument('--output', default='results/runs')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--module-config', help='검증 세트로 선정한 EnhancementConfig JSON')
    return p


def run(args):
    if args.epochs < 1 or args.crop_size < 16 or not 0 < args.max_hours <= 3 or args.lr <= 0:
        raise ValueError('epochs>=1, crop-size>=16, lr>0, 0<max_hours<=3이어야 합니다.')
    seed_all(args.seed)
    config = EnhancementConfig(**json.loads(Path(args.module_config).read_text())) if args.module_config else EnhancementConfig()
    loaders = create_lol_dataloaders(low_dir=args.low_dir, high_dir=args.high_dir,
        test_low_dir=args.test_low_dir, test_high_dir=args.test_high_dir,
        val_ratio=args.val_ratio, seed=args.seed, split_seed=args.split_seed,
        crop_size=args.crop_size, batch_size=args.batch_size, num_workers=args.num_workers)
    manifest = {k:[{'low':str(p.low.resolve()), 'high':str(p.high.resolve()) if p.high else None}
                   for p in v.dataset.pairs] for k,v in loaders.items()}
    run_dir = Path(args.output)/f'{args.mode}_seed{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir/'last.pt'
    settings = vars(args).copy()
    settings.pop('resume')
    model = ZeroDCE(args.mode,config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4)
    objective = ZeroDCELoss()
    start_epoch, elapsed, history = 0, 0.0, []
    if args.resume:
        # 자신의 학습 코드가 저장한 체크포인트만 로드하세요.
        saved = torch.load(checkpoint_path,map_location=args.device,weights_only=False)
        if saved['settings'] != settings or saved['manifest'] != manifest or saved['module_config'] != asdict(config):
            raise ValueError('재개 설정 또는 데이터 분할이 기존 실행과 다릅니다.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch,elapsed,history = saved['epoch'],saved['elapsed_seconds'],saved['history']
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if torch.cuda.is_available() and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        loaders['train'].generator.set_state(saved['loader_rng'].cpu())
    elif checkpoint_path.exists():
        raise FileExistsError(f'{checkpoint_path}가 있습니다. --resume 또는 새 --output을 사용하세요.')
    (run_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (run_dir/'config.json').write_text(json.dumps(dict(settings=settings,module_config=asdict(config)),indent=2))
    wall_start = time.monotonic()
    budget = args.max_hours*3600
    def save(epoch, complete):
        payload = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
            elapsed_seconds=elapsed+time.monotonic()-wall_start,complete=complete,
            settings=settings,module_config=asdict(config),manifest=manifest,history=history,
            environment=dict(torch_version=torch.__version__,device=args.device,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            loader_rng=loaders['train'].generator.get_state())
        temporary = run_dir/'last.tmp'
        torch.save(payload,temporary); temporary.replace(checkpoint_path)
        (run_dir/'history.json').write_text(json.dumps(history,indent=2))
    for epoch in range(start_epoch,args.epochs):
        epoch_start = time.monotonic()
        if elapsed+epoch_start-wall_start >= budget:
            break
        # 중단 시 마지막 완료 에폭부터 재개합니다. 부분 에폭 가중치는 저장하지 않습니다.
        save(epoch,False)
        model.train(); values = []
        for low in loaders['train']:
            if elapsed+time.monotonic()-wall_start >= budget:
                print('시간 한도 도달: 부분 에폭은 폐기. 이 실행은 비교 대상에서 제외합니다.'); return
            low = low.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            enhanced,curves,_ = model(low)
            loss,_ = objective(low,enhanced,curves)
            if not torch.isfinite(loss):
                raise RuntimeError('학습 손실이 유한하지 않습니다.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.1)
            optimizer.step(); values.append(loss.item())
        model.eval(); validation = []
        with torch.no_grad():
            for batch in loaders['val']:
                low = (batch[0] if isinstance(batch,(list,tuple)) else batch).to(args.device)
                enhanced,curves,_ = model(low)
                validation.append(objective(low,enhanced,curves)[0].item())
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(values)),
                            val_loss=float(np.mean(validation)),seconds=time.monotonic()-epoch_start))
        save(epoch+1,epoch+1==args.epochs and elapsed+time.monotonic()-wall_start<=budget)
        print(json.dumps(history[-1]),flush=True)
    print(f'체크포인트: {checkpoint_path}; 본 실험은 complete=true인 동일 에폭 실행만 비교하세요.')

if __name__ == '__main__':
    run(parser().parse_args())

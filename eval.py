"""저장된 분할로 평가, 공정 비교 검사, 이미지별 지표와 비교 그림 저장."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from dataset import ImagePair, LowLightDataset
from model import ZeroDCE, EnhancementConfig, gaussian, luminance, edge_magnitude


def metrics(output, reference, saturation=0.98, flat_threshold=0.02):
    mse = (output-reference).square().mean().item()
    # RGB 채널별 Gaussian SSIM, 11x11 유효 영역, data_range=1.
    mu, target_mu = gaussian(output,1.5), gaussian(reference,1.5)
    variance = gaussian(output.square(),1.5)-mu.square()
    target_variance = gaussian(reference.square(),1.5)-target_mu.square()
    covariance = gaussian(output*reference,1.5)-mu*target_mu
    ssim = ((2*mu*target_mu+0.01**2)*(2*covariance+0.03**2)
           /((mu.square()+target_mu.square()+0.01**2)*(variance+target_variance+0.03**2)))
    if min(output.shape[-2:]) > 10:
        ssim = ssim[:,:,5:-5,5:-5]
    flat = edge_magnitude(gaussian(luminance(reference),1.5)) < flat_threshold
    residual = (luminance(output)-luminance(reference))[flat]
    return dict(psnr=float(-10*np.log10(max(mse,1e-12))),ssim=ssim.mean().item(),
        saturation_ratio=(luminance(output)>=saturation).float().mean().item(),
        flat_residual_std=residual.std(unbiased=False).item() if residual.numel() else None,
        color_mean_error=(output.mean((2,3))-reference.mean((2,3))).abs().mean().item())


def to_image(x):
    return Image.fromarray((x[0].detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8'))


def comparison(path, panels):
    width,height = panels[0][1].size
    canvas = Image.new('RGB',(width*len(panels),height+24),'white')
    draw = ImageDraw.Draw(canvas)
    for i,(label,picture) in enumerate(panels):
        canvas.paste(picture,(i*width,24)); draw.text((i*width+4,4),label,fill='black')
    canvas.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compare',nargs='+',default=['none','bright','noise','both'])
    p.add_argument('--runs',default='results/runs')
    p.add_argument('--seeds',nargs='+',type=int,default=[0])
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--output',default='results/evaluation')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--repeats',type=int,default=10)
    args = p.parse_args()
    if args.repeats<1 or args.warmup<0:
        p.error('repeats>=1, warmup>=0이어야 합니다.')
    out = Path(args.output)/args.split; out.mkdir(parents=True,exist_ok=True)
    baseline_contract = None
    summaries = []
    for seed in args.seeds:
        models = {}; checkpoints = {}
        for mode in args.compare:
            path = Path(args.runs)/f'{mode}_seed{seed}'/'last.pt'
            ckpt = torch.load(path,map_location=args.device,weights_only=False)
            if not ckpt['complete']:
                raise ValueError(f'미완료/시간 초과 실행은 비교할 수 없습니다: {path}')
            if ckpt['settings']['mode'] != mode or ckpt['settings']['seed'] != seed:
                raise ValueError(f'체크포인트 mode/seed 불일치: {path}')
            keys = ('epochs','lr','batch_size','crop_size','split_seed','val_ratio','num_workers')
            contract = ({k:ckpt['settings'][k] for k in keys},ckpt['manifest'],ckpt['module_config'])
            if baseline_contract is None:
                baseline_contract = contract
            elif contract != baseline_contract:
                raise ValueError('데이터 분할, 에폭, 학습 설정 또는 모듈 설정이 다릅니다.')
            model = ZeroDCE(mode,EnhancementConfig(**ckpt['module_config'])).to(args.device)
            model.load_state_dict(ckpt['model']); model.eval()
            models[mode] = model; checkpoints[mode] = str(path)
        manifest = baseline_contract[1][args.split]
        pairs = [ImagePair(Path(row['low']),Path(row['high']) if row['high'] else None) for row in manifest]
        dataset = LowLightDataset(pairs,return_reference=True)
        rows = []; times = {mode:[] for mode in models}
        image_dir = out/f'seed{seed}'; image_dir.mkdir(exist_ok=True)
        def sync():
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
        with torch.inference_mode():
            for index,(low,reference) in enumerate(dataset):
                low,reference = low[None].to(args.device),reference[None].to(args.device)
                panels = [('input',to_image(low))]; outputs = {}; scores = {}
                for mode,model in models.items():
                    for _ in range(args.warmup): model(low)
                    sync(); start = time.perf_counter()
                    for _ in range(args.repeats): enhanced,_,maps = model(low)
                    sync(); times[mode].append((time.perf_counter()-start)*1000/args.repeats)
                    outputs[mode] = to_image(enhanced)
                    scores[mode] = metrics(enhanced,reference)
                    rows.append(dict(index=index,file=str(pairs[index].low),mode=mode,**scores[mode]))
                    panels.append((mode,outputs[mode]))
                    for name,value in maps.items():
                        to_image(value.expand(-1,3,-1,-1)).save(image_dir/f'{index:04d}_{mode}_{name}.png')
                panels.append(('reference',to_image(reference)))
                comparison(image_dir/f'{index:04d}_comparison.png',panels)
                # 입력의 최저/최고 밝기 화소 주변 패치. 모든 구성에서 같은 좌표 사용.
                y = luminance(low)[0,0]; width,height = panels[0][1].size
                for label,position in [('dark',y.argmin().item()),('bright',y.argmax().item())]:
                    cy,cx = divmod(position,width); size=min(96,width,height)
                    left=max(0,min(width-size,cx-size//2)); top=max(0,min(height-size,cy-size//2))
                    box=(left,top,left+size,top+size)
                    comparison(image_dir/f'{index:04d}_{label}.png',[(name,img.crop(box).resize((size*2,size*2))) for name,img in panels])
        with (image_dir/'per_image.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        for mode,model in models.items():
            selected=[r for r in rows if r['mode']==mode]
            summary=dict(seed=seed,mode=mode,checkpoint=checkpoints[mode],
                parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                inference_ms=float(np.mean(times[mode])),device=args.device,images=len(pairs))
            for key in ('psnr','ssim','saturation_ratio','flat_residual_std','color_mean_error'):
                values=[r[key] for r in selected if r[key] is not None]
                summary[key]=float(np.mean(values)) if values else None
            summaries.append(summary)
        if 'none' in models:
            baseline={r['index']:r['psnr'] for r in rows if r['mode']=='none'}
            rankings={mode:sorted([dict(index=r['index'],file=r['file'],delta_psnr_db=r['psnr']-baseline[r['index']])
                        for r in rows if r['mode']==mode],key=lambda r:r['delta_psnr_db']) for mode in models if mode!='none'}
            (image_dir/'rankings.json').write_text(json.dumps(rankings,indent=2))
    aggregate=[]
    for mode in args.compare:
        selected=[r for r in summaries if r['mode']==mode]
        entry=dict(mode=mode,seeds=args.seeds,std_definition='population, ddof=0')
        for key in ('psnr','ssim','saturation_ratio','flat_residual_std','color_mean_error','inference_ms'):
            values=[r[key] for r in selected if r[key] is not None]
            entry[key]=dict(mean=float(np.mean(values)),std=float(np.std(values))) if values else None
        aggregate.append(entry)
    result=dict(split=args.split,ssim='RGB Gaussian 11x11 sigma=1.5; valid border; range=1',
        psnr='RGB MSE, maximum reported 120dB',saturation_threshold=0.98,flat_edge_threshold=0.02,
        timing='batch=1, warmed up per image, model forward only; excludes I/O and metrics',
        runs=summaries,aggregate=aggregate)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()

"""
RTIOD/LTDv2 Optimized Detection Script
=======================================
- TTA (Test-Time Augmentation) for +1-3% mAP boost
- Thermal-optimized confidence threshold
- Class-specific NMS for better separation
- FP16 inference for speed
- max_det=100 (competition safe)

Usage:
  python detect.py submission.type=val
  python detect.py submission.type=test
"""

from ultralytics import YOLO
from src.datasets.dataset import load_datasets
import os
import hydra
import torch
import json
from tqdm import tqdm
from omegaconf import DictConfig


@hydra.main(config_path='config', config_name='config', version_base="1.3")   
def main(args: DictConfig):
    """
    Generate submission predictions with thermal-optimized inference.
    
    Key optimizations:
    - TTA enabled for +1-3% mAP (slower but worth it for competition)
    - Lower conf=0.10 for thermal (catches more objects)
    - IoU=0.45 for thermal blur (less aggressive NMS)
    - max_det=100 (safe limit, matches training)
    """
    
    # =========================================================================
    # SETUP
    # =========================================================================
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"RTIOD INFERENCE - THERMAL OPTIMIZED")
    print(f"{'='*70}")
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU:           {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print(f"Device:        CPU (slower)")
    
    # Model verification
    if not os.path.exists(args.modelCheckpoint):
        print(f"\n❌ Model not found: {args.modelCheckpoint}")
        print(f"   Check path or train first with: python train.py")
        return
    
    model = YOLO(args.modelCheckpoint)
    model.to(device)
    
    # =========================================================================
    # INFERENCE PARAMETERS (THERMAL-OPTIMIZED)
    # =========================================================================
    # These are tuned for LTDv2 thermal imagery based on research
    
    IMGSZ = getattr(args, 'imgsz', 480)           # Match training resolution
    CONF = 0.10                                    # Lower for thermal (was 0.08-0.12)
    IOU = 0.45                                     # Slightly lower for thermal blur
    MAX_DET = 100                                  # Safe limit, matches training
    USE_TTA = True                                 # +1-3% mAP boost
    USE_HALF = True                                # FP16 for speed
    
    print(f"Model:         {args.modelCheckpoint}")
    print(f"Image size:    {IMGSZ}px")
    print(f"Confidence:    {CONF} (thermal-optimized)")
    print(f"IoU NMS:       {IOU} (thermal-adjusted)")
    print(f"Max dets:      {MAX_DET}")
    print(f"TTA:           {'ENABLED ✓' if USE_TTA else 'disabled'}")
    print(f"Half (FP16):   {'ENABLED ✓' if USE_HALF else 'disabled'}")
    print(f"{'='*70}\n")
    
    # =========================================================================
    # LOAD DATASET
    # =========================================================================
    train, val, test, collate_fn = load_datasets(args)
    
    if args.submission.type == 'val':
        templatePath = args.submission.valTemplate
        dataset = val
        print(f"Mode: VALIDATION ({len(dataset)} images)")
    elif args.submission.type == 'test':
        templatePath = args.submission.testTemplate
        dataset = test
        print(f"Mode: TEST ({len(dataset)} images)")
    else:
        print(f"❌ Invalid submission.type: {args.submission.type}")
        print(f"   Use: submission.type=val or submission.type=test")
        return
    
    # Load submission template
    with open(templatePath, 'r') as f:
        submission = json.load(f)
    
    print(f"Template:      {templatePath}")
    print(f"{'='*70}\n")
    
    # =========================================================================
    # INFERENCE LOOP
    # =========================================================================
    total_detections = 0
    empty_images = 0
    
    for i in tqdm(range(len(dataset)), desc="Processing", unit="img"):
        imgPath = dataset.get_img_path(i)
        
        # Run prediction with thermal-optimized settings
        results = model.predict(
            source=imgPath,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            device=device,
            verbose=False,
            agnostic_nms=False,      # Class-specific NMS (better for multi-class)
            augment=USE_TTA,         # TTA for +1-3% mAP
            half=USE_HALF,           # FP16 for speed
        )
        
        result = results[0]
        
        # Extract predictions safely
        if len(result.boxes) > 0:
            # xyxy format: [x1, y1, x2, y2] absolute coordinates
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            conf = result.boxes.conf.cpu().numpy().tolist()
            # Labels: add 1 to convert 0-indexed to 1-indexed (competition format)
            labels = (result.boxes.cls.int() + 1).cpu().numpy().tolist()
            total_detections += len(boxes)
        else:
            boxes = []
            conf = []
            labels = []
            empty_images += 1
        
        # Update submission dict
        img_id_str = str(dataset.ids[i])
        if img_id_str not in submission:
            tqdm.write(f"⚠️  Missing template ID: {img_id_str}, skipping")
            continue
        
        submission[img_id_str]['boxes'] = boxes
        submission[img_id_str]['scores'] = conf
        submission[img_id_str]['labels'] = labels
    
    # =========================================================================
    # SAVE PREDICTIONS
    # =========================================================================
    os.makedirs('submissions', exist_ok=True)
    output_path = f'submissions/predictions_{args.submission.type}.json'
    
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Predictions saved:  {output_path}")
    print(f"📊 Images processed:   {len(dataset)}")
    print(f"🎯 Total detections:   {total_detections}")
    print(f"📦 Avg dets/image:     {total_detections / len(dataset):.1f}")
    print(f"⚪ Empty images:       {empty_images} ({100*empty_images/len(dataset):.1f}%)")
    print(f"{'='*70}\n")
    
    # Quick sanity check
    if total_detections == 0:
        print("⚠️  WARNING: No detections! Check model path and confidence threshold.")
    elif total_detections / len(dataset) < 1.0:
        print("⚠️  Low detection rate - consider lowering confidence threshold.")
    
    print("Ready for submission!")


if __name__ == "__main__":
    main()
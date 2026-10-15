# Cropper

Perspective warp to 63×88 mm. Default detector: **obb-v6**. 90° landscape → portrait here; 180° printed-up is handled in [`pipeline.py`](../../pipeline.py).

```powershell
$env:PYTHONPATH="src"
python -m cropper photo.jpg --out output/cropper/cards --conf 0.8 --preview
python -m cropper export C:\other-project\ptcg_cropper
```

Export copies this package plus **`obb-v6.pt`** from [`models/`](../../models/README.md). In the other project: `python crop_from_obb.py photo.jpg --out cards`.

Docs: [`docs/cropper.md`](../../docs/cropper.md)

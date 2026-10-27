"""Everything both arms share. Nothing here knows which scorer is running.

``backbone``   the frozen DINOv3 encoder, loaded through timm
``geometry``   the downscale rule and the tile layout it produces
``loading``    image handles that decode on demand
``thresholds`` the threshold rules, derived from held-out normals, and the F1 of each
``datasets``   MVTec AD, MVTec AD 2 and VisA behind one interface
``evaluation`` ground truth and the metrics, all at native resolution
"""

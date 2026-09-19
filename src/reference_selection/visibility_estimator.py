def visibility_proxy(region, pose, name):
    # Coarse projection proxy only. Cannot detect sunglasses, hair, hands or true occlusion.
    if not region.any():return 0.
    side=-1 if name.startswith('left') else 1 if name.startswith('right') else 0
    return max(.15,1-max(0,side*pose['yaw_proxy'])*1.5)

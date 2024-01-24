


def l2_pose_regularization(poses):
    l2loss = []
    for pose in poses:
        for p in pose:
            if len(p)>0:
                l2loss.append((p[0]**2).mean())
    return sum(l2loss) / len(l2loss)

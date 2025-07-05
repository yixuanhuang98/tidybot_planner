import math

def dot(a, b):
    """Dot product helper function from controller.py"""
    return a[0] * b[0] + a[1] * b[1]

def intersect(d, f, r, use_t1=False):
    """Line-circle intersection from controller.py"""
    # https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm/1084899%231084899
    a = dot(d, d)
    b = 2 * dot(f, d)
    c = dot(f, f) - r * r
    discriminant = (b * b) - (4 * a * c)
    if discriminant >= 0:
        if use_t1:
            t1 = (-b - math.sqrt(discriminant)) / (2 * a + 1e-6)
            if 0 <= t1 <= 1:
                return t1
        else:
            t2 = (-b + math.sqrt(discriminant)) / (2 * a + 1e-6)
            if 0 <= t2 <= 1:
                return t2
    return None

def distance(pt1, pt2):
    """Calculate distance between two points from controller.py"""
    return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

def restrict_heading_range(h):
    """Normalize heading to [-π, π] range from controller.py"""
    return (h + math.pi) % (2 * math.pi) - math.pi 
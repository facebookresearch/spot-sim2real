import numpy as np


def get_xyzxyz(centroid, extents):
    x1 = centroid[0] - (extents[0] / 2.0)
    y1 = centroid[1] - (extents[1] / 2.0)
    z1 = centroid[2] - (extents[2] / 2.0)

    x2 = centroid[0] + (extents[0] / 2.0)
    y2 = centroid[1] + (extents[1] / 2.0)
    z2 = centroid[2] + (extents[2] / 2.0)

    return np.array([x1, y1, z1]), np.array([x2, y2, z2])


# don't remove this keep it incase new logic fails
def intersect_ray_with_aabb(ray_origin, ray_direction, box_min, box_max):
    t_min = (box_min[0] - ray_origin[0]) / ray_direction[0]
    t_max = (box_max[0] - ray_origin[0]) / ray_direction[0]

    if t_min > t_max:
        t_min, t_max = t_max, t_min

    ty_min = (box_min[1] - ray_origin[1]) / ray_direction[1]
    ty_max = (box_max[1] - ray_origin[1]) / ray_direction[1]

    if ty_min > ty_max:
        ty_min, ty_max = ty_max, ty_min

    if (t_min > ty_max) or (ty_min > t_max):
        return False, None, None

    if ty_min > t_min:
        t_min = ty_min

    if ty_max < t_max:
        t_max = ty_max

    tz_min = (box_min[2] - ray_origin[2]) / ray_direction[2]
    tz_max = (box_max[2] - ray_origin[2]) / ray_direction[2]

    if tz_min > tz_max:
        tz_min, tz_max = tz_max, tz_min

    if (t_min > tz_max) or (tz_min > t_max):
        return False, None, None

    if tz_min > t_min:
        t_min = tz_min

    if tz_max < t_max:
        t_max = tz_max

    # If t_min is negative, the intersection is behind the ray origin
    if t_min < 0 and t_max < 0:
        return False, None, None

    STATIC_OFFSET = 0.0
    print("t_min", t_min)
    t_min -= STATIC_OFFSET if t_min > STATIC_OFFSET else 0.0
    # Return the intersection points (if needed)
    intersection_point_1 = ray_origin + t_min * ray_direction
    intersection_point_2 = ray_origin + t_max * ray_direction

    return True, intersection_point_1, intersection_point_2, t_min, t_max


def midpoint(x1, x2):
    return (x1 + x2) / 2.0


def angle_between_vectors(v1, v2):
    # Ensure the vectors are numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Compute the dot product
    dot_product = np.dot(v1, v2)

    # Compute the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Compute the cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)

    # Clip the cosine value to the range [-1, 1] to avoid numerical issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Compute the angle in radians
    angle_radians = np.arccos(cos_angle)

    # Convert the angle to degrees (optional)
    angle_degrees = np.degrees(angle_radians)

    return angle_radians, angle_degrees


if __name__ == "__main__":
    # receptacle details from CG
    bbox_extents = np.array([0.9, 0.7, 0.5])
    bbox_centers = np.array([4.0, -1.9, 0.5])
    boxMin, boxMax = get_xyzxyz(bbox_centers, bbox_extents)
    print("boxMin", boxMin, "boxMax", boxMax)

    # PASTE your waypoint in robot_xy; only x,y, keep z as it is
    # obtain xy,yaw from Concept graph based on detection confs & area. Rank top 5 etc.
    robot_xy = np.array(
        [4.651140979172928, -3.7375389203182516, bbox_centers[-1] + 0.1]
    )
    yaw_cg = 88.59926264693141

    # radir important
    raydir = (bbox_centers - robot_xy) / np.linalg.norm(bbox_centers - robot_xy)

    # ray AABB intersection but useless because we find nearest face/edge using vector similarity
    # intersects, pt1, pt2 = intersect_ray_with_aabb(robot_xy_z, raydir, boxMin, boxMax)
    intersects = True

    if intersects:
        # select face based on vector similarity;
        # additionally use occupancy map here,
        # use raymarching to find ray collisions

        STATIC_OFFSET = 0.7  # adjustment for base 0.5 + extra offset
        (x1, y1), (x2, y2) = boxMin[:2], boxMax[:2]

        face_1 = np.array([midpoint(x1, x2), y1 - STATIC_OFFSET])
        face_1_vector = bbox_centers[:2] - face_1  # x1,y1, x2,y1
        face_1_vector = face_1_vector / np.linalg.norm(face_1_vector)

        face_2 = np.array([midpoint(x1, x2), y2 + STATIC_OFFSET])
        face_2_vector = bbox_centers[:2] - face_2  # x1,y2, x2,y2
        face_2_vector = face_2_vector / np.linalg.norm(face_2_vector)

        face_3 = np.array([x1 - STATIC_OFFSET, midpoint(y1, y2)])
        face_3_vector = bbox_centers[:2] - face_3  # x1, y1, x1, y2
        face_3_vector = face_3_vector / np.linalg.norm(face_3_vector)

        face_4 = np.array([x2 + STATIC_OFFSET, midpoint(y1, y2)])
        face_4_vector = bbox_centers[:2] - face_4  # x2, y1, x2, y2
        face_4_vector = face_4_vector / np.linalg.norm(face_4_vector)

        faces = [
            (face_1_vector, face_1),
            (face_2_vector, face_2),
            (face_3_vector, face_3),
            (face_4_vector, face_4),
        ]
        angles_betwn_approach_vector_and_faces = [
            angle_between_vectors(raydir[:2], face[0])[1] for face in faces
        ]
        min_idx = np.argmin(angles_betwn_approach_vector_and_faces)
        min_angle = angles_betwn_approach_vector_and_faces[min_idx]
        nearestfacevector, nearestface = faces[min_idx]
        yaw_calc = angle_between_vectors(np.array([1, 0]), nearestfacevector)[1]

        print("XY, use this in dynamic nav/nav", nearestface)
        print("Yaw_cg", yaw_cg, "yaw_cal", yaw_calc)
        # either use yaw from cg or from calculations

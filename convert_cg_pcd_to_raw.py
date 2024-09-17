import argparse
import pickle

import numpy as np
import open3d as o3d


def downsample_pcd(pcd, voxel_size=0.05):
    """
    Downsamples the point cloud using a voxel grid filter.
    """
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    return downsampled_pcd


def save_pcd_as_pickle(downsampled_pcd, output_file="downsampled.pkl"):
    """
    Save the downsampled point cloud as a .pkl file (nx3 format).
    """
    points = np.array(downsampled_pcd.points)
    # Convert Open3D vector of points to a list of lists
    # points_list = [list(point) for point in points]

    # Dump the points to a pickle file
    with open(output_file, "wb") as f:
        pickle.dump(points, f)


def pick_points(pcd):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def main():
    parser = argparse.ArgumentParser(description="Process and downsample a .pcd file.")
    parser.add_argument(
        "pcd_file",
        type=str,
        help="Path to the input .pcd file.",
        default="/home/tushar/Desktop/try2_habitat_llm/fre_apt_cleaned_preprocessed_kavit/rgb_cloud/pointcloud.pcd",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Flag to visualize the point cloud."
    )

    args = parser.parse_args()

    # Load the .pcd file

    pcd = o3d.io.read_point_cloud(args.pcd_file)
    print(f"Loading point cloud from: {args.pcd_file}, {pcd}")

    if pcd.is_empty():
        print("Error: The point cloud is empty or the file is invalid.")
        return

    # Downsample the point cloud

    downsampled_pcd = downsample_pcd(pcd)
    print(f"Downsampling the point cloud... {downsampled_pcd}")

    # Visualize if the flag is set
    if args.visualize:
        print("Visualizing the downsampled point cloud...")
        point = pick_points(pcd)
        print(f"Picked point {point}")
        # o3d.visualization.draw_geometries([downsampled_pcd])

    # Save the downsampled points to a pickle file
    print("Saving the downsampled points to downsampled.pkl...")
    save_pcd_as_pickle(
        downsampled_pcd, "spot_rl_experiments/spot_rl/utils/point_cloud_fre.pkl"
    )

    print("Processing completed!")


if __name__ == "__main__":
    main()

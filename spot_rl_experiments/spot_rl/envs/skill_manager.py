# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# black: ignore-errors

import time
from typing import Any, Dict, Tuple

import magnum as mn
import numpy as np
import rospy
from multimethod import multimethod
from perception_and_utils.utils.generic_utils import (
    conditional_print,
    map_user_input_to_boolean,
)
from spot_rl.skills.atomic_skills import (
    MobilePickEE,
    Navigation,
    OpenCloseDrawer,
    Pick,
    Place,
    SemanticPick,
    SemanticPlace,
    SemanticPlaceEE,
)
from spot_rl.utils.construct_configs import (
    construct_config_for_gaze,
    construct_config_for_nav,
    construct_config_for_open_close_drawer,
    construct_config_for_place,
    construct_config_for_semantic_place,
)
from spot_rl.utils.heuristic_nav import (
    ImageSearch,
    heurisitic_object_search_and_navigation,
)
from spot_rl.utils.pose_estimation import OrientationSolver
from spot_rl.utils.search_table_location import (
    contrained_place_point_estimation,
    convert_point_in_body_to_place_waypoint,
    detect_place_point_by_pcd_method,
    plot_place_point_in_gripper_image,
)
from spot_rl.utils.utils import (
    calculate_height,
    get_waypoint_yaml,
    nav_target_from_waypoint,
    place_target_from_waypoint,
)
from spot_wrapper.spot import Spot

#
# SKILL MANAGER is a collection of ALL SKILLS that spot exposes
#


class SpotSkillManager:
    """
    Interface class to invoke skills for Spot.
    Exposes skills like nav, pick, place, and dock as functions

    Args:
        verbose (bool): Set verbosity status for printing msgs on terminal
        use_policies (bool): Whether to use NN policy or BD API (currenly only place supports BD API)

    How to use:
        1. Create the SpotSkillManager object
        2. Call the skill functions as needed (nav, pick, place, dock)

    Examples:
        # Create the spot skill manager object (this will create spot object, init lease, and construct configs and power on the robot)
        spotskillmanager = SpotSkillManager()

        # Skill - Navigation
        # To navigate to a waypoint (str)
        status, msg = spotskillmanager.nav("test_square_vertex1")

        # To navigate to a waypoint (x, y, theta)
        status, msg = spotskillmanager.nav(x, y, theta)

        # Skill - Pick
        # To pick an object
        status, msg = spotskillmanager.pick("ball_plush")

        # Skill - Place
        # To place an object at a waypoint (str)
        status, msg = spotskillmanager.place("test_place_front")

        # To place an object at a location (x, y, z)
        status, msg = spotskillmanager.place(x, y, z)

        # To dock
        status, msg = spotskillmanager.dock()

        # This can can be used for multiple skills in a sequence like nav-pick-nav-place
        # Nav-Pick-Nav-Place sequence 1
        spotskillmanager.nav("test_square_vertex1")
        spotskillmanager.pick("ball_plush")
        spotskillmanager.nav("test_place_front")
        spotskillmanager.place("test_place_front")
    """

    def __init__(
        self,
        spot: Spot = None,
        nav_config=None,
        pick_config=None,
        place_config=None,
        open_close_drawer_config=None,
        use_mobile_pick: bool = False,
        use_semantic_place: bool = False,
        use_pick_ee: bool = False,
        use_place_ee: bool = False,
        verbose: bool = True,
        use_policies: bool = True,
    ):
        # Set the verbose flag
        self.verbose = verbose

        # Create the spot object, init lease, and construct configs
        self.spot: Spot = None  # will be initialized in __init_spot()

        # Create the spot object, init lease, and construct configs
        self.__init_spot(spot=spot)

        # Process the meta parameters
        self._use_mobile_pick = use_mobile_pick
        self.allow_semantic_place = (
            use_semantic_place and self.spot.is_intel_service_available()
        )
        self.use_pick_ee = use_pick_ee
        self.use_place_ee = use_place_ee

        # Construct configs for nav, gaze, place and open-close drawers
        self.__init_configs(
            nav_config=nav_config,
            pick_config=pick_config,
            place_config=place_config,
            open_close_drawer_config=open_close_drawer_config,
        )

        # Initiate the controllers for nav, gaze, and place
        self.__init_controllers(use_policies=use_policies)

        self.orientation_solver: OrientationSolver = OrientationSolver()

        # Power on the robot
        self.spot.power_robot()

        # Create a local waypoint dictionary
        self.waypoints_yaml_dict = get_waypoint_yaml()
        self.verbose = True

    def __del__(self):
        pass

    def __init_spot(
        self,
        spot: Spot = None,
    ):
        """
        Initialize the Spot object, and acquire lease
        """
        if spot is None:
            # Create Spot object
            self.spot = Spot("SkillManagerSpot")

            # Acquire spot's lease
            self.lease = self.spot.get_lease(hijack=True)
            if not self.lease:
                conditional_print(
                    message="Failed to get lease for Spot. Exiting!",
                    verbose=self.verbose,
                )
                exit(1)
        else:
            self.spot = spot

    def __init_configs(
        self,
        nav_config=None,
        pick_config=None,
        place_config=None,
        open_close_drawer_config=None,
    ):
        """
        Construct configs for nav, pick, place, open/close
        """

        self.nav_config = construct_config_for_nav() if not nav_config else nav_config
        self.pick_config = (
            construct_config_for_gaze(max_episode_steps=100)
            if not pick_config
            else pick_config
        )

        if place_config is None:
            self.place_config = (
                construct_config_for_semantic_place()
                if self.allow_semantic_place
                else construct_config_for_place()
            )
        else:
            self.place_config = place_config

        self.open_close_drawer_config = (
            construct_config_for_open_close_drawer()
            if not open_close_drawer_config
            else open_close_drawer_config
        )

    def __init_controllers(self, use_policies: bool = True):
        """
        Initiate the controllers for nav, pick, place, and open-close drawers
        """

        # Init nav controller
        self.nav_controller = Navigation(
            spot=self.spot,
            config=self.nav_config,
        )

        # Init pick controller
        if self.use_pick_ee:
            self.gaze_controller = MobilePickEE(
                spot=self.spot,
                config=self.pick_config,
                use_mobile_pick=self._use_mobile_pick,
            )
        else:
            self.gaze_controller = Pick(
                spot=self.spot,
                config=self.pick_config,
                use_mobile_pick=self._use_mobile_pick,
            )

        # Init place controller
        if self.allow_semantic_place:

            if self.use_place_ee:
                self.place_controller = SemanticPlaceEE(
                    spot=self.spot,
                    config=self.place_config,
                    use_semantic_place=self.allow_semantic_place,  # TODO: Mostly not needed
                )
            else:
                self.place_controller = SemanticPlace(
                    spot=self.spot,
                    config=self.place_config,
                )
        else:
            self.place_controller = Place(
                spot=self.spot,
                config=self.place_config,
                use_policies=use_policies,
            )

        # Init open-close drawer controller
        self.open_close_drawer_controller = OpenCloseDrawer(
            spot=self.spot,
            config=self.open_close_drawer_config,
        )

        # Init semantic pick controller
        self.semantic_gaze_controller = SemanticPick(
            spot=self.spot,
            config=self.pick_config,
        )

    def reset(self):
        # Reset the policies and environments via the controllers
        raise NotImplementedError

    @multimethod  # type: ignore
    def nav(self, nav_target: str = None) -> Tuple[bool, str]:  # type: ignore
        """
        Perform the nav action on the navigation target specified as a known string

        Args:
            nav_target (str): Name of the nav target (as stored in waypoints.yaml)

        Returns:
            bool: True if navigation was successful, False otherwise
            str: Message indicating the status of the navigation
        """
        conditional_print(
            message=f"Received nav target request for - {nav_target}",
            verbose=self.verbose,
        )
        should_dock = nav_target == "dock"
        if nav_target is not None:
            # Get the nav target coordinates
            try:
                nav_target_tuple = nav_target_from_waypoint(
                    nav_target, self.waypoints_yaml_dict
                )
                self.current_receptacle_name = nav_target
            except Exception:
                message = (
                    f"Failed - nav target {nav_target} not found - use the exact name"
                )
                conditional_print(message=message, verbose=self.verbose)
                return False, message
        else:
            msg = "No nav target specified, skipping nav"
            return False, msg

        nav_x, nav_y, nav_theta = nav_target_tuple
        status, message = self.nav(nav_x, nav_y, nav_theta, False, should_dock)
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    @multimethod  # type: ignore
    def nav(  # noqa
        self,
        x: float,
        y: float,
        theta: float,
        reset_current_receptacle_name: bool = True,
        should_dock: bool = False,
    ) -> Tuple[bool, str]:
        """
        Perform the nav action on the navigation target specified as a metric location

        Args:
            x (float): x coordinate of the nav target (in meters) specified in the world frame
            y (float): y coordinate of the nav target (in meters) specified in the world frame
            theta (float): yaw for the nav target (in radians) specified in the world frame
            reset_current_receptacle_name (bool): reset the current receptacle name to None

        Returns:
            bool: True if navigation was successful, False otherwise
            str: Message indicating the status of the navigation
        """
        # Keep track of the current receptacle that the robot navigates to for later grasping
        # mode of gaze skill
        if reset_current_receptacle_name:
            receptacle_name = getattr(self, "current_receptacle_name", None)
            setattr(self, "current_receptacle_name", receptacle_name)

        goal_dict = {
            "nav_target": (x, y, theta),
            "should_dock": should_dock,
        }  # type: Dict[str, Any]
        status, message = self.nav_controller.execute(goal_dict=goal_dict)
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    @multimethod  # type: ignore
    def nav(self, x: float, y: float) -> Tuple[bool, str]:  # noqa
        """
        Perform the nav action on the navigation target with yaw specified as a metric location
        Args:
            x (float): x coordinate of the nav target (in meters) specified in the world frame
            y (float): y coordinate of the nav target (in meters) specified in the world frame
        Returns:
            bool: True if navigation was successful, False otherwise
            str: Message indicating the status of the navigation
        """
        theta = 0.0
        goal_dict = {
            "nav_target": (x, y, theta),
            "dynamic_yaw": True,
        }  # type: Dict[str, Any]
        status, message = self.nav_controller.execute(goal_dict=goal_dict)
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    def heuristic_mobile_gaze(
        self,
        x: float,
        y: float,
        theta: float,
        object_target: str,
        image_search: ImageSearch = None,
        save_cone_search_images: bool = True,
        pull_back: bool = True,
    ) -> bool:
        """
        Perform Heuristic mobile navigation to the not very accurate pick target obtained from Aria glasses(x,y,theta)
        Step 1 goto x,y,theta with head first navigation method
        Step 2 search object_target using object detector in the list of -90 to 90 degrees (180 degrees) with 20 degrees interval
        Step 3 If found update given x, y, theta with new
        Step 4 Navigate to new x, y, theta in 50 steps
        Step 5 turn off head first navigation
        Step 6 Return Flag: bool signifying whether we found the object_target & whether is ready to pick ?

        Args:
            x (float): x coordinate of the nav target (in meters) specified in the world frame
            y (float): y coordinate of the nav target (in meters) specified in the world frame
            theta (float): yaw for the nav target (in radians) specified in the world frame
            object_target: str object to search
            image_search : spot_rl.utils.heuristic_nav.ImageSearch, Optional, default=None, ImageSearch (object detector wrapper), if none creates a new one for you uses OwlVit
            save_cone_search_images: bool, optional, default= True, saves image with detections in each search cone
            pull_back : bool, optional, default=True, pulls back x,y along theta direction
        Returns:
            bool: True if navigation was successful, False otherwise, if True you are good to fire .pick metho

        """
        print(f"Original Nav targets {x, y, theta}")
        return heurisitic_object_search_and_navigation(
            x,
            y,
            theta,
            object_target,
            image_search,
            save_cone_search_images,
            pull_back,
            self,
            angle_interval=self.nav_config.get("HEURISTIC_SEARCH_ANGLE_INTERVAL", 20),
        )

    def pick(
        self,
        target_obj_name: str = None,
        enable_pose_estimation: bool = False,
        enable_pose_correction: bool = False,
        enable_force_control: bool = False,
    ) -> Tuple[bool, str]:
        """
        Perform the pick action on the pick target specified as string

        Args:
            target_obj_name (str): Descriptive name of the pick target (eg: ball_plush)
            enable_pose_estimation (bool) : Enable pose estimation default : False
            enable_pose_correction (bool) : Enable pose correction default : False

        Returns:
            bool: True if pick was successful, False otherwise
            str: Message indicating the status of the pick
        """
        grasp_mode = "any"
        # Try to determine current receptacle and
        # see if we set any preferred grasping type for it
        current_receptacle_name = getattr(self, "current_receptacle_name", None)
        if current_receptacle_name is not None:
            receptacles = self.pick_config.get("RECEPTACLES", {})
            for receptacle_name, grasp_type in receptacles.items():
                if receptacle_name == current_receptacle_name:
                    grasp_mode = grasp_type
                    break

        # Determine the grasping type from ros
        place_target = rospy.get_param("nav_target_for_pick", f"{time.time()},None")
        place_target = place_target.split(",")[1]
        place_target = place_target.split("_", 1)
        place_target[1] = place_target[1].replace("_", " ")
        place_target = "_".join(place_target)
        height = calculate_height(place_target)
        if height > 1.0:
            grasp_mode = "side"

        self.gaze_controller.set_grasp_type(grasp_mode)
        goal_dict = {
            "target_object": target_obj_name,
            "take_user_input": False,
        }  # type: Dict[str, Any]
        if enable_pose_correction or enable_force_control:
            assert (
                enable_pose_estimation
            ), "Pose estimation must be enabled if you want to perform pose correction or force control"

        if enable_pose_estimation:
            object_meshes = self.pick_config.get("OBJECT_MESHES", [])
            found = False
            for object_mesh_name in object_meshes:
                if object_mesh_name in target_obj_name:
                    found = True
            assert (
                found
            ), f"{target_obj_name} not found in meshes that we have {object_meshes}"

        self.gaze_controller.set_pose_estimation_flags(
            enable_pose_estimation, enable_pose_correction
        )
        self.gaze_controller.set_force_control(enable_force_control)
        status, message = self.gaze_controller.execute(goal_dict=goal_dict)
        if status and enable_pose_correction:
            spinal_axis = rospy.get_param("spinal_axis")
            gamma = rospy.get_param("gamma", 0)
            spinal_axis = mn.Vector3(*spinal_axis)
            (
                correction_status,
                put_back_object_status,
            ) = self.orientation_solver.perform_orientation_correction(
                self.spot,
                spinal_axis,
                self.gaze_controller.ee_point_before_starting_the_skill,
                gamma,
                target_obj_name,
            )
            status = status and correction_status and put_back_object_status
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    def semanticpick(
        self, target_obj_name: str = None, grasping_type: str = "topdown"
    ) -> Tuple[bool, str]:
        """
        Perform the semantic pick action on the pick target specified as string

        Args:
            target_obj_name (str): Descriptive name of the pick target (eg: ball_plush)
            grasping_type (str): The grasping type

        Returns:
            bool: True if pick was successful, False otherwise
            str: Message indicating the status of the pick
        """
        assert grasping_type in [
            "topdown",
            "side",
        ], f"Do not support {grasping_type} grasping"

        goal_dict = {
            "target_object": target_obj_name,
            "take_user_input": False,
            "grasping_type": grasping_type,
        }  # type: Dict[str, Any]
        status, message = self.semantic_gaze_controller.execute(goal_dict=goal_dict)
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    def set_arm_joint_angles(self, place_target: str = None):
        height = calculate_height(place_target)
        if height < self.place_config.HEIGHT_THRESHOLD:
            return self.place_config.GAZE_ARM_JOINT_ANGLES_LOW_RECEPTACLES
        else:
            return self.place_config.GAZE_ARM_JOINT_ANGLES_HIGH_RECEPTACLES

    # estimating waypoint
    def waypoint_estimator(self, percentile, visualize, height_adjustment_threshold):
        try:
            (
                place_target_location,
                place_target_in_gripper_camera,
                edge_point_in_base,
                _,
            ) = detect_place_point_by_pcd_method(
                self.spot,
                self.arm_joint_angles,
                percentile=percentile,
                visualize=visualize,
                height_adjustment_offset=height_adjustment_threshold,
            )
            print(f"Estimate Place xyz: {place_target_location}")
            if visualize:
                plot_place_point_in_gripper_image(
                    self.spot, place_target_in_gripper_camera
                )
            return place_target_location, edge_point_in_base
        except Exception as e:
            message = f"Failed to estimate place way point due to {str(e)}"
            conditional_print(message=message, verbose=self.verbose)
            print(message)
            # Just return some default value when the waypoint estimation is wrong
            return np.array([0.7, 0.0, 0.4]), np.array([0.5, 0.0, 0.4])

    @multimethod  # type: ignore
    def place(self, place_target: str = None, ee_orientation_at_grasping: np.ndarray = None, is_local: bool = False, visualize: bool = False, enable_waypoint_estimation: bool = False) -> Tuple[bool, str]:  # type: ignore
        """
        Perform the place action on the place target specified as known string

        Args:
            place_target (str): Name of the place target (as stored in waypoints.yaml)
            ee_orientation_at_grasping (list): The ee orientation at grasping. If users specifiy, the robot will place the object in the desired pose
                This is only used for the semantic place skills.
            enable_waypoint_estimation (bool) : This flag determines whether the place policy must use waypoint estimation or not

        Returns:
            bool: True if place was successful, False otherwise
            str: Message indicating the status of the place
        """
        conditional_print(
            message=f"Received place target request for - {place_target}",
            verbose=self.verbose,
        )

        if not enable_waypoint_estimation:
            # Get the place target coordinates
            try:
                place_target_location = place_target_from_waypoint(
                    place_target, self.waypoints_yaml_dict
                )
            except Exception:
                message = f"Failed - place target {place_target} not found - use the exact name"
                conditional_print(message=message, verbose=self.verbose)
                return False, message
            if self.allow_semantic_place:
                # Convert HOME frame coordinates into body frame
                place_target_location = (
                    self.place_controller.env.get_target_in_base_frame(
                        mn.Vector3(*place_target_location.astype(np.float64).tolist())
                    )
                )
                is_local = True
        else:
            message = "No place target specified, estimating point through heuristic"
            # Estimate Joint angles from height of the object using Concept graph
            self.arm_joint_angles = self.set_arm_joint_angles(place_target)

            conditional_print(message=message, verbose=self.verbose)
            is_local = True
            percentile = 0 if visualize else 30
            if self.allow_semantic_place and self.use_place_ee:
                height_adjustment_threshold = 0.1
                percentile = 0 if visualize else 30
            elif self.allow_semantic_place:
                height_adjustment_threshold = 0.05
            else:
                height_adjustment_threshold = 0.23
            # estimate waypoint
            place_target_location, edge_point_in_base = self.waypoint_estimator(
                percentile, visualize, height_adjustment_threshold
            )

        edge_x = float(edge_point_in_base[0])

        # Move the base if the robot is too far away from the place target
        start_walking_distance_threshold = self.place_config.get(
            "MIN_DISTANCE_TO_PLACE_TARGET", 0.7
        )  # in meters
        travel_time_for_walking_to_target = 2  # in seconds
        if edge_x > start_walking_distance_threshold:
            walk_distance = edge_x - start_walking_distance_threshold
            (
                before_walking_x,
                before_walking_y,
                before_walking_yaw,
            ) = self.spot.get_xy_yaw()
            # Walk to the place target
            self.spot.set_base_position(
                walk_distance, 0, 0, travel_time_for_walking_to_target, True
            )
            # Wait for the robot to finish walking
            time.sleep(travel_time_for_walking_to_target)
            after_walking_x, after_walking_y, after_walking_yaw = self.spot.get_xy_yaw()
            walk_distance = after_walking_x - before_walking_x
            # Offset the place target location
            place_target_location[0] -= walk_distance
            # For EE place policy, we are re-estimating waypoint after the base is moved since the arm start to place from higher angle
            if self.use_place_ee:
                print("Re-estimating waypoint ....")
                rospy.set_param("is_gripper_blocked", 0)
                previous_place_target_location = place_target_location.copy()
                try:
                    place_target_location, edge_point_in_base = self.waypoint_estimator(
                        percentile, visualize, height_adjustment_threshold
                    )
                except Exception as e:
                    print(
                        f"Error during waypoint re-estimation: {str(e)}. Using previous point."
                    )
                    place_target_location = previous_place_target_location

        place_x, place_y, place_z = place_target_location.astype(np.float64).tolist()
        status, message = self.place(
            place_x,
            place_y,
            place_z,
            ee_orientation_at_grasping=ee_orientation_at_grasping,
            is_local=is_local,
        )
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    def contrainedplace(self, object_target: str = None, ee_orientation_at_grasping: np.ndarray = None, is_local: bool = False, visualize: bool = False, proposition: str = "left") -> Tuple[bool, str]:  # type: ignore
        """
        Perform the place action on the place target specified as known string

        Args:
            place_target (str): Name of the place target (as stored in waypoints.yaml)
            ee_orientation_at_grasping (list): The ee orientation at grasping. If users specifiy, the robot will place the object in the desired pose
                This is only used for the semantic place skills.
            is_local: if the target place point is in the local or global frame or not
            proposition: indicate the placement location relative to the object

        Returns:
            bool: True if place was successful, False otherwise
            str: Message indicating the status of the place
        """
        conditional_print(
            message=f"Received place target request for - {object_target}",
            verbose=self.verbose,
        )

        assert proposition in [
            "left",
            "right",
            "next-to",
        ], f"Place skill does not support proposition of {proposition}"

        # Estimate Joint angles from height of the object using Concept graph
        self.arm_joint_angles = self.set_arm_joint_angles(object_target)

        # Esitmate the waypoint
        (
            place_target_location,
            place_target_in_gripper_camera,
            _,
        ) = contrained_place_point_estimation(
            object_target,
            proposition,
            self.spot,
            self.arm_joint_angles,
            percentile=70,
            visualize=visualize,
            height_adjustment_offset=0.10 if self.allow_semantic_place else 0.23,
            image_scale=self.get_env().config.IMAGE_SCALE,
        )

        print(f"Estimate Place xyz: {place_target_location}")

        if visualize:
            plot_place_point_in_gripper_image(self.spot, place_target_in_gripper_camera)

        place_x, place_y, place_z = place_target_location.astype(np.float64).tolist()

        # Call place skill given the estimate waypoint
        status, message = self.place(
            place_x,
            place_y,
            place_z,
            ee_orientation_at_grasping=ee_orientation_at_grasping,
            is_local=is_local,
        )
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    @multimethod  # type: ignore
    def place(  # noqa
        self,
        x: float,
        y: float,
        z: float,
        is_local: bool = False,
        ee_orientation_at_grasping: np.ndarray = None,
    ) -> Tuple[bool, str]:
        """
        Perform the place action on the place target specified as metric location

        Args:
            x (float): x coordinate of the place target (in meters) specified in spot's frame if is_local is true, otherwise in world frame
            y (float): y coordinate of the place target (in meters) specified in spot's frame if is_local is true, otherwise in world frame
            z (float): z coordinate of the place target (in meters) specified in spot's frame if is_local is true, otherwise in world frame
            is_local (bool): If True, place in the spot's body frame, else in the world frame
            ee_orientation_at_grasping (np.ndarray): The grasping arm joint angles

        Returns:
            status (bool): True if place was successful, False otherwise
            message (str): Message indicating the status of the place
        """
        goal_dict = {
            "place_target": (x, y, z),
            "is_local": is_local,
            "ee_orientation_at_grasping": ee_orientation_at_grasping,
        }  # type: Dict[str, Any]
        status, message = self.place_controller.execute(goal_dict=goal_dict)
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    def opendrawer(self) -> Tuple[bool, str]:
        """
        Perform the open drawer skill

        Returns:
            bool: True if the open drawer skill was successful, False otherwise
            str: Message indicating the status of opening drawers
        """
        return self.openclosedrawer(open_mode=True)

    def closedrawer(self) -> Tuple[bool, str]:
        """
        Perform the close drawer skill

        Returns:
            bool: True if the close skill was successful, False otherwise
            str: Message indicating the status of closing drawers
        """
        return self.openclosedrawer(open_mode=False)

    def opencabinet(self, cab_door="left") -> Tuple[bool, str]:
        """
        Perform the open cabinet skill. You need to tell skills if the left door or right door

        Returns:
            bool: True if the open cabinet skill was successful, False otherwise
            str: Message indicating the status of opening cabinets
        """

        assert cab_door in [
            "left",
            "right",
        ], f"cab_door is not right or left but {cab_door}"

        return self.openclosedrawer(
            open_mode=True, rep_type="cabinet", cab_door=cab_door
        )

    def closecabinet(self, cab_door="left") -> Tuple[bool, str]:
        """
        Perform the close cabinet skill. You need to tell skills if the left door or right door

        Returns:
            bool: True if the close skill was successful, False otherwise
            str: Message indicating the status of closing cabinets
        """
        raise NotImplementedError

    def openclosedrawer(
        self, open_mode=True, rep_type="drawer", cab_door="left"
    ) -> Tuple[bool, str]:
        """
        Perform the open and close drawer skill

        Returns:
            bool: True if open close was successful, False otherwise
            str: Message indicating the status of the open/close drawers
        """

        assert rep_type in [
            "drawer",
            "cabinet",
        ], f"Do not support repcetacle type {rep_type} in open/close skills"

        goal_dict = {
            "mode": "open" if open_mode else "close",
            "rep_type": rep_type,
            "cab_door": cab_door,
        }  # type: Dict[str, Any]
        status, message = self.open_close_drawer_controller.execute(goal_dict=goal_dict)
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    def sit(self):
        """
        Sit the robot
        This needs to be a separate function as lease is acquired within skill manager is not available outside of it.
        """
        self.spot.sit()

    def get_env(self):
        "Get the env for the ease of the access"
        return self.nav_controller.env

    def dock(self, home_robot=True):
        # Stow back the arm
        self.get_env().reset_arm()

        status = False
        message = "Dock failed"
        try:
            # Navigate to the dock
            status, message = self.nav("dock")

            # Dock
            self.spot.shutdown(should_dock=True, home_robot=home_robot)
        except Exception:
            message = "Error encountered while docking"
            conditional_print(message=message, verbose=self.verbose)
            return status, message

        if status:
            message = "Successfully docked"
        return status, message


if __name__ == "__main__":

    # We initialize the skill using SpotSkillManager.
    # Note that if you want to use mobile gaze for pick,
    # instead of static gaze, you need to do
    # SpotSkillManager(use_mobile_pick=True)

    spotskillmanager = SpotSkillManager()

    # Nav-Pick-Nav-Place sequence 1
    spotskillmanager.nav("test_square_vertex1")
    spotskillmanager.pick("plush bear")
    spotskillmanager.nav("test_place_front")
    spotskillmanager.place("test_place_front")

    # Nav-Pick-Nav-Place sequence 2
    spotskillmanager.nav("test_square_vertex3")
    spotskillmanager.pick("caterpillar_plush")
    spotskillmanager.nav("test_place_left")
    spotskillmanager.place("test_place_left")

    # Navigate to dock and shutdown
    spotskillmanager.dock(home_robot=True)

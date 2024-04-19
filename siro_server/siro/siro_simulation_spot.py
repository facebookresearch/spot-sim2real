import logging

import numpy as np

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
)


# SIM:


class Sim:
    sim_spot_pos_x = 0.0
    sim_spot_pos_y = 0.0
    sim_spot_yaw = 0.0

    sim_spot_target_pos_x = 0.0
    sim_spot_target_pos_y = 0.0
    sim_spot_target_yaw = 0.0

    sim_spot_speed_meters_per_sec = 1.5
    sim_spot_speed_degrees_per_sec = 60
    sim_spot_currently_moving = False
    sim_spot_currently_rotating = False

    def reset(self):
        self.sim_spot_pos_x = 0.0
        self.sim_spot_pos_y = 0.0
        self.sim_spot_yaw = 0.0

        self.sim_spot_target_pos_x = 0.0
        self.sim_spot_target_pos_y = 0.0
        self.sim_spot_target_yaw = 0.0

        self.sim_spot_currently_moving = False
        self.sim_spot_currently_rotating = False

    def set_base_position(self, _x, _y, _yaw):
        self.sim_spot_target_pos_x = _x
        self.sim_spot_target_pos_y = _y
        self.sim_spot_target_yaw = _yaw
        current_in_360 = self.sim_spot_yaw % 360
        target_in_360 = self.sim_spot_target_yaw % 360
        logging.info(
            f"internal_sim_set_target {self.sim_spot_yaw} | {current_in_360} :: {self.sim_spot_target_yaw} | {target_in_360}"
        )

        clockwise = target_in_360 - current_in_360
        anti_clockwise = current_in_360 + (360 - target_in_360)

        if anti_clockwise < clockwise:
            self.sim_spot_speed_degrees_per_sec = (
                abs(self.sim_spot_speed_degrees_per_sec) * -1
            )
            # i.e. if current yaw is 10, and target is 350, lerp from 370 to 350
            self.sim_spot_yaw = self.sim_spot_yaw + 360
        else:
            self.sim_spot_speed_degrees_per_sec = abs(
                self.sim_spot_speed_degrees_per_sec
            )

    def get_xy_yaw(self):
        return self.sim_spot_pos_x, self.sim_spot_pos_y, self.sim_spot_yaw

    def get_is_moving_is_rotating(self):
        is_moving = self.sim_spot_currently_moving
        self.sim_spot_currently_moving = False
        is_rotating = self.sim_spot_currently_rotating
        self.sim_spot_currently_rotating = False

        return is_moving, is_rotating

    def internal_sim_tick(self, _delta_time):
        (
            self.sim_spot_pos_x,
            self.sim_spot_pos_y,
            self.sim_spot_currently_moving,
        ) = self.utils_xy_match_target(
            self.sim_spot_pos_x,
            self.sim_spot_pos_y,
            self.sim_spot_target_pos_x,
            self.sim_spot_target_pos_y,
            self.sim_spot_speed_meters_per_sec,
            _delta_time,
        )
        (
            self.sim_spot_yaw,
            self.sim_spot_currently_rotating,
        ) = self.utils_yaw_match_target(
            self.sim_spot_yaw,
            self.sim_spot_target_yaw,
            self.sim_spot_speed_degrees_per_sec,
            _delta_time,
        )

    def utils_xy_match_target(
        self, _x, _y, _target_x, _target_y, _speed_in_meters_per_sec, _time_delta
    ):
        xy = np.array([_x, _y])
        target_xy = np.array([_target_x, _target_y])
        dif = target_xy - xy
        mag = np.linalg.norm(dif)
        if mag <= 0.05:
            return _target_x, _target_y, False
        else:
            step = xy + dif / mag * _speed_in_meters_per_sec * _time_delta
            return step[0], step[1], True

    def utils_yaw_match_target(
        self, _yaw, _target_yaw, _speed_deg_per_sec, _time_delta
    ):
        # moving anti-clockwise
        if _speed_deg_per_sec < 0:
            if _yaw <= _target_yaw:
                return _target_yaw, False
        else:
            if _yaw >= _target_yaw:
                return _target_yaw, False

        dif = (_target_yaw - _yaw) if _speed_deg_per_sec >= 0 else (_yaw - _target_yaw)
        dif_abs = abs(dif)
        if dif_abs < 0.1:
            return _target_yaw, False
        else:
            to_rotate = _speed_deg_per_sec * _time_delta
            new_val = (_yaw - to_rotate) if dif < 0 else (_yaw + to_rotate)
            return new_val, True

#!/usr/bin/env python

import numpy as np
from .agent import Agent
from .world_model import WorldModel

class Supervisor(Agent):
    ACTION_DICT = {
        'turn+40': 0,
        'turn_ball': 1,
        'dash': 2,
        'kick_to_goal': 3,
        'turn-40': 4,
    }

    def __init__(self, agent, action_dict=None):
        super().__init__()
        self._agent = agent

        # Environment vector.
        self.env = np.zeros(9)

        self.goal_side = None
        self.goal = None
        self.own_goal = None

        self.action_dict = action_dict if action_dict is not None else Supervisor.ACTION_DICT
        self.reverse_action_dict = dict((v, k) for k, v in self.action_dict.items())

        # Action vector.
        self.action = np.zeros(len(self.action_dict))

        # Add a hook to the supervisor in the agent.
        self._agent.ctx.parent = self

    def transform_wm(self, wm):

        ball_visible = 1 if wm.ball else 0

        if ball_visible:
            ball_direction = wm.ball.direction if wm.ball.direction else 0
            ball_distance = wm.ball.distance if wm.ball.distance else 0
        else:
            ball_direction = self.env[0]
            ball_distance = self.env[1]

        # Resolve the goal side of the agent.
        if wm.side == WorldModel.SIDE_R:
            self.goal_side = WorldModel.SIDE_L
        else:
            self.goal_side = WorldModel.SIDE_R

        # Resolve the goal and own_goal objects.
        if wm.goals:
            goal_1 = wm.goals[0]

            if goal_1.goal_id == self.goal_side:
                goal = goal_1
                own_goal = wm.goals[1] if len(wm.goals) == 2 else None

            else:
                goal = wm.goals[1] if len(wm.goals) == 2 else None
                own_goal = goal_1
        else:
            goal = None
            own_goal = None

        goal_visible = 1 if goal else 0
        own_goal_visible = 1 if own_goal else 0

        if goal_visible:
            goal_direction = goal.direction if goal.direction else 0
            goal_distance = goal.distance if goal.distance else 0
        else:
            goal_direction = self.env[2]
            goal_distance = self.env[3]

        if own_goal_visible:
            own_goal_direction = own_goal.direction if own_goal.direction else 0
            own_goal_distance = own_goal.distance if own_goal.distance else 0
        else:
            own_goal_direction = self.env[4]
            own_goal_distance = self.env[5]

        # Add the 2 goals to the supervisor object.
        self.goal = goal
        self.own_goal = own_goal

        self.env[0] = ball_direction
        self.env[1] = ball_distance
        self.env[2] = goal_direction
        self.env[3] = goal_distance
        self.env[4] = own_goal_direction
        self.env[5] = own_goal_distance
        self.env[6] = ball_visible
        self.env[7] = goal_visible
        self.env[8] = own_goal_visible

        print('env = {}\n'.format(self.env))

        # Inject the goal object into the imitation/behavioural cloning agent.
        self._agent.ctx.goal = self.goal

        return self.env

    def transform_action(self, action):
        # Returns the action vector associated with the specified action.
        self.action = self.action_dict[action]
        return self.action

    def reverse_transform_action(self, action_idx):
        # Returns the action associated with the specified action_vector.
        action_key = self.reverse_action_dict[action_idx]
        return action_key

    def think(self):
        if not self.in_kick_off_formation:
            self._think()
            return

        self._agent.ctx.env = self.transform_wm(self.wm)
        self._think()
        self._agent.think()

    def _think(self):
        # This method should select an action based on the current environment,
        # then encode the action into a vector and save it in the act parameter of
        # the ctx object in the agent.
        # The agent will have a vector representation of the environment in its ctx
        # field.
        raise Exception('_think() method not implemented.')

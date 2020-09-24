import numpy as np
import math
from utils import wrapToPi

# command zero velocities once we are this close to the goal
RHO_THRES = 0.05
ALPHA_THRES = 0.1
DELTA_THRES = 0.1

# Switch over to sinc() function when abs(alpha) is less than this
SINC_THRES = 0.001

class PoseController:
    """ Pose stabilization controller """
    def __init__(self, k1, k2, k3, V_max=0.5, om_max=1):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max

    def load_goal(self, x_g, y_g, th_g):
        """ Loads in a new goal position """
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs: 
            V, om: Control actions

        Hints: You'll need to use the wrapToPi function. The np.sinc function
        may also be useful, look up its documentation
        """
        ########## Code starts here ##########
        # define errors in x and y position
        x_error = self.x_g - x
        y_error = self.y_g - y

        # change coordinate system
        rho = math.sqrt(pow(x_error,2) + pow(y_error,2))
        alpha = wrapToPi(math.atan2(y_error, x_error) - th)
        delta = wrapToPi(math.atan2(y_error, x_error) - self.th_g)

        # calculate V with control law
        V = self.k1*rho*math.cos(alpha)

        # near alpha = 0, sinc(alpha) = sin(pi*alpha)/(pi*alpha) ~= sin(2*alpha)/(2*alpha) = sin(alpha)*cos(alpha)/alpha
        # but we have the benefit that sinc(alpha) is defined at alpha = 0
        if (abs(alpha) < SINC_THRES):
            sinc = np.sinc(alpha)
        else:
            sinc = math.sin(alpha)*math.cos(alpha)/alpha
            
        # calculate om with control law
        om = self.k2*alpha + self.k1*sinc*(alpha+self.k3*delta)
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om

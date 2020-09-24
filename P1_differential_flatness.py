import numpy as np
import math
from numpy import linalg
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from utils import *

class State:
    def __init__(self,x,y,V,th):
        self.x = x
        self.y = y
        self.V = V
        self.th = th

    @property
    def xd(self):
        return self.V*np.cos(self.th)

    @property
    def yd(self):
        return self.V*np.sin(self.th)


def compute_traj_coeffs(initial_state, final_state, tf):
    """
    Inputs:
        initial_state (State)
        final_state (State)
        tf (float) final time
    Output:
        coeffs (np.array shape [8]), coefficients on the basis functions

    Hint: Use the np.linalg.solve function.
    """
    ########## Code starts here ##########

    # columns:    x1/y1 x2/y2      x3/y3        x4/y4
    A = np.array([[1,    0,         0,           0],  # x(0) = x1 and y(0) = y1
                  [0,    1,         0,           0],  # xdot(0) = x2 and ydot(0) = y2
                  [1,   tf, pow(tf,2),   pow(tf,3)],  # x(tf) = x1 + tf*x2 + tf^2*x3 + tf^3*x4 and y(tf) = y1 + tf*y2 + tf^2*y3 + tf^3*y4
                  [0,    1,      2*tf, 3*pow(tf,2)]]) # xdot(tf) = x2 + 2*tf*x3 + 3*tf^2*x4 and ydot(tf) = y2 + 2*tf*y3 + 3*tf^2*y4

    # calculate initial/final x/y velocities based on initial/final V and th
    initial_xdot = initial_state.V * math.cos(initial_state.th)
    final_xdot = final_state.V * math.cos(final_state.th)
    initial_ydot = initial_state.V * math.sin(initial_state.th)
    final_ydot = final_state.V * math.sin(final_state.th)

    b_x = np.array([initial_state.x, # x(0)
                    initial_xdot,    # xdot(0)
                    final_state.x,   # x(tf)
                    final_xdot])     # xdot(tf)

    b_y = np.array([initial_state.y, # y(0)
                    initial_ydot,    # ydot(0)
                    final_state.y,   # y(tf)
                    final_ydot])     # ydot(tf)

    # Now we have two matrix equations: A*x = b_x and A*y = b_y, where x = (x1,x2,x3,x4) and y = (y1,y2,y3,y4)
    x = np.linalg.solve(A, b_x)
    y = np.linalg.solve(A, b_y)

    coeffs = np.concatenate((x, y)) # coeffs is (x1,x2,x3,x4,y1,y2,y3,y4)
    ########## Code ends here ##########
    return coeffs

def compute_traj(coeffs, tf, N):
    """
    Inputs:
        coeffs (np.array shape [8]), coefficients on the basis functions
        tf (float) final_time
        N (int) number of points
    Output:
        traj (np.array shape [N,7]), N points along the trajectory, from t=0
            to t=tf, evenly spaced in time
    """
    t = np.linspace(0,tf,N) # generate evenly spaced points from 0 to tf
    traj = np.zeros((N,7))
    ########## Code starts here ##########
    x1 = coeffs[0]
    x2 = coeffs[1]
    x3 = coeffs[2]
    x4 = coeffs[3]
    y1 = coeffs[4]
    y2 = coeffs[5]
    y3 = coeffs[6]
    y4 = coeffs[7]

    for i in range(N):
        traj[i, 0] = x1 + x2*t[i] + x3*pow(t[i],2) + x4*pow(t[i],3) # x
        traj[i, 1] = y1 + y2*t[i] + y3*pow(t[i],2) + y4*pow(t[i],3) # y
        traj[i, 3] = x2 + 2*x3*t[i] + 3*x4*pow(t[i],2) # xdot
        traj[i, 4] = y2 + 2*y3*t[i] + 3*y4*pow(t[i],2) # ydot
        traj[i, 2] = math.atan2(traj[i, 4], traj[i, 3]) # th = arctan(ydot/xdot)
        traj[i, 5] = 2*x3 + 6*x4*t[i] # xdotdot
        traj[i, 6] = 2*y3 + 6*y4*t[i] # ydotdot

    ########## Code ends here ##########

    return t, traj

def compute_controls(traj):
    """
    Input:
        traj (np.array shape [N,7])
    Outputs:
        V (np.array shape [N]) V at each point of traj
        om (np.array shape [N]) om at each point of traj
    """
    ########## Code starts here ##########
    N = traj.shape[0]
    V = np.zeros(N)
    om = np.zeros(N)

    for i in range(N):
        # V = sqrt(xdot^2 + ydot^2)
        V[i] = math.sqrt(pow(traj[i,3], 2) + pow(traj[i,4], 2))

        # matrix A = [cos(th), -Vsin(th)]
        #            [sin(th),  Vcos(th)]
        A = np.array([[math.cos(traj[i,2]), -V[i]*math.sin(traj[i,2])],
                      [math.sin(traj[i,2]),  V[i]*math.cos(traj[i,2])]])

        # vector b = (xdotdot, ydotdot)
        b = (traj[i,5], traj[i,6])

        # solve Ax=b where x = (a, om)
        x = np.linalg.solve(A, b)
        om[i] = x[1]

    ########## Code ends here ##########

    return V, om

def compute_arc_length(V, t):
    """
    This function computes arc-length s as a function of t.
    Inputs:
        V: a vector of velocities of length T
        t: a vector of time of length T
    Output:
        s: the arc-length as a function of time. s[i] is the arc-length at time
            t[i]. This has length T.

    Hint: Use the function cumtrapz. This should take one line.
    """
    s = None
    ########## Code starts here ##########
    s = cumtrapz(V, t, initial=0)
    ########## Code ends here ##########
    return s

def rescale_V(V, om, V_max, om_max):
    """
    This function computes V_tilde, given the unconstrained solution V, and om.
    Inputs:
        V: vector of velocities of length T. Solution from the unconstrained,
            differential flatness problem.
        om: vector of angular velocities of length T. Solution from the
            unconstrained, differential flatness problem.
    Output:
        V_tilde: Rescaled velocity that satisfies the control constraints.

    Hint: At each timestep V_tilde should be computed as a minimum of the
    original value V, and values required to ensure _both_ constraints are
    satisfied.
    Hint: This should only take one or two lines.
    """
    ########## Code starts here ##########
    # minimum of initial velocity, max velocity, and scaled velocity (which ensures om_max is not exeeded)
    V_tilde = [min(V[i], V_max, abs(V[i]*om_max/om[i])) for i in range(len(V))]
    ########## Code ends here ##########
    return V_tilde


def compute_tau(V_tilde, s):
    """
    This function computes the new time history tau as a function of s.
    Inputs:
        V_tilde: a sequence of scaled velocities of length T.
        s: a sequence of arc-length of length T.
    Output:
        tau: the new time history for the sequence. tau[i] is the time at s[i]. This has length T.

    Hint: Use the function cumtrapz. This should take one line.
    """
    ########## Code starts here ##########
    tau = cumtrapz([1/V_tilde[i] for i in range(len(V_tilde))], s, initial=0)
    ########## Code ends here ##########
    return tau

def rescale_om(V, om, V_tilde):
    """
    This function computes the rescaled om control.
    Inputs:
        V: vector of velocities of length T. Solution from the unconstrained, differential flatness problem.
        om:  vector of angular velocities of length T. Solution from the unconstrained, differential flatness problem.
        V_tilde: vector of scaled velocities of length T.
    Output:
        om_tilde: vector of scaled angular velocities

    Hint: This should take one line.
    """
    ########## Code starts here ##########
    om_tilde = [om[i]*V_tilde[i]/V[i] for i in range(len(V))]
    ########## Code ends here ##########
    return om_tilde

def compute_traj_with_limits(z_0, z_f, tf, N, V_max, om_max):
    coeffs = compute_traj_coeffs(initial_state=z_0, final_state=z_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V,om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)

    return traj, tau, V_tilde, om_tilde

def interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f):
    """
    Inputs:
        traj (np.array [N,7]) original unscaled trajectory
        tau (np.array [N]) rescaled time at orignal traj points
        V_tilde (np.array [N]) new velocities to use
        om_tilde (np.array [N]) new rotational velocities to use
        dt (float) timestep for interpolation

    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    """
    # Get new final time
    tf_new = tau[-1]

    # Generate new uniform time grid
    N_new = int(tf_new/dt)
    t_new = dt*np.array(range(N_new+1))

    # Interpolate for state trajectory
    traj_scaled = np.zeros((N_new+1,7))
    traj_scaled[:,0] = np.interp(t_new,tau,traj[:,0])   # x
    traj_scaled[:,1] = np.interp(t_new,tau,traj[:,1])   # y
    traj_scaled[:,2] = np.interp(t_new,tau,traj[:,2])   # th
    # Interpolate for scaled velocities
    V_scaled = np.interp(t_new, tau, V_tilde)           # V
    om_scaled = np.interp(t_new, tau, om_tilde)         # om
    # Compute xy velocities
    traj_scaled[:,3] = V_scaled*np.cos(traj_scaled[:,2])    # xd
    traj_scaled[:,4] = V_scaled*np.sin(traj_scaled[:,2])    # yd
    # Compute xy acclerations
    traj_scaled[:,5] = np.append(np.diff(traj_scaled[:,3])/dt,-s_f.V*om_scaled[-1]*np.sin(s_f.th)) # xdd
    traj_scaled[:,6] = np.append(np.diff(traj_scaled[:,4])/dt, s_f.V*om_scaled[-1]*np.cos(s_f.th)) # ydd

    return t_new, V_scaled, om_scaled, traj_scaled

if __name__ == "__main__":
    # traj, V, om = differential_flatness_trajectory()
    # Constants
    tf = 15.
    V_max = 0.5
    om_max = 1

    # time
    dt = 0.005
    N = int(tf/dt)+1
    t = dt*np.array(range(N))

    # Initial conditions
    s_0 = State(x=0, y=0, V=V_max, th=-np.pi/2)

    # Final conditions
    s_f = State(x=5, y=5, V=V_max, th=-np.pi/2)

    coeffs = compute_traj_coeffs(initial_state=s_0, final_state=s_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V,om = compute_controls(traj=traj)
    part_b_complete = False
    s = compute_arc_length(V, t)
    if s is not None:
        part_b_complete = True
        V_tilde = rescale_V(V, om, V_max, om_max)
        tau = compute_tau(V_tilde, s)
        om_tilde = rescale_om(V, om, V_tilde)

        t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)

        # Save trajectory data
        data = {'z': traj_scaled, 'V': V_scaled, 'om': om_scaled}
        save_dict(data, "data/differential_flatness.pkl")

    maybe_makedirs('plots')

    # Plots
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 2, 1)
    plt.plot(traj[:,0], traj[:,1], 'k-',linewidth=2)
    plt.grid(True)
    plt.plot(s_0.x, s_0.y, 'go', markerfacecolor='green', markersize=15)
    plt.plot(s_f.x, s_f.y, 'ro', markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title("Path (position)")
    plt.axis([-1, 6, -1, 6])

    ax = plt.subplot(2, 2, 2)
    plt.plot(t, V, linewidth=2)
    plt.plot(t, om, linewidth=2)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
    plt.title('Original Control Input')
    plt.tight_layout()

    plt.subplot(2, 2, 4, sharex=ax)
    if part_b_complete:
        plt.plot(t_new, V_scaled, linewidth=2)
        plt.plot(t_new, om_scaled, linewidth=2)
        plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
        plt.grid(True)
    else:
        plt.text(0.5,0.5,"[Problem iv not completed]", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.xlabel('Time [s]')
    plt.title('Scaled Control Input')
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    if part_b_complete:
        h, = plt.plot(t, s, 'b-', linewidth=2)
        handles = [h]
        labels = ["Original"]
        h, = plt.plot(tau, s, 'r-', linewidth=2)
        handles.append(h)
        labels.append("Scaled")
        plt.legend(handles, labels, loc="best")
    else:
        plt.text(0.5,0.5,"[Problem iv not completed]", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Arc-length [m]')
    plt.title('Original and scaled arc-length')
    plt.tight_layout()
    plt.savefig("plots/differential_flatness.png")
    plt.show()

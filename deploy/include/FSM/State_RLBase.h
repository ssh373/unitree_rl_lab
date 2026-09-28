// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once
#include <deque>
#include <cmath>

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "TelemetryLogger.h"
#include <unitree/dds_wrapper/robots/go2/go2_sub.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <thread>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        open_logger();

        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();
                sample_velocity();  // 1 s window for the robot-frame velocity log

                const long velocity_log_interval = std::max(
                    1L, static_cast<long>(std::lround(0.5 / env->step_dt))
                );
                if (env->episode_length % velocity_log_interval == 0)
                {
                    log_velocity();
                }

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
        logger.close();
    }

private:
    void log_velocity();
    void sample_velocity();
    void open_logger();
    // Robot-frame velocity readout for the periodic log: heading (yaw from the IMU quaternion,
    // unwrapped) and the world-frame base velocity, sampled every policy step and averaged over
    // the last second. The old readout printed the pelvis-frame gyro z of one instant, which in a
    // running gait is dominated by the stride oscillation and does not measure the turn rate.
    struct VelSample { double t; double yaw; float vx_w; float vy_w; bool has_lin; };
    std::deque<VelSample> vel_hist_;
    double yaw_unwrapped_ = 0.0, yaw_prev_ = 0.0;
    bool yaw_init_ = false;

    // Replica of the sim2real CheckSafetyLimits kill (g1_sim2real/wholebody_rl.hpp), so the
    // simulated robot collapses into Passive exactly where the real one would. Thresholds come
    // from config.yaml (FSM.<state>.protect); omit that block to disable the whole check.
    bool motor_limits_exceeded();

    // Replica of the sim2real LowCommandWriter torque clamp: cap the commanded PD torque at
    // clamp_ratio x torque_limit by handing that joint a pure torque command for the tick.
    // Without it MuJoCo saturates at <motor ctrlrange> instead, i.e. at the kill threshold.
    void apply_torque_clamp();

    std::vector<float> protect_tau_limit_;   // per motor, SDK order [Nm]; empty = off
    float protect_upper_dq_limit_ = 0.0f;    // upper body only (sdk >= 12) [rad/s]; 0 = off
    float protect_clamp_ratio_ = 0.0f;       // clamp commands at this fraction of the limit; 0 = off
    float protect_warn_ratio_ = 0.0f;        // periodically report peak |tau_est| / limit; 0 = off
    float protect_peak_ratio_ = 0.0f;
    int protect_peak_idx_ = -1;
    long protect_tick_ = 0;

protected:
    TelemetryLogger logger;

private:

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::shared_ptr<unitree::robot::go2::subscription::SportModeState> highstate;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)

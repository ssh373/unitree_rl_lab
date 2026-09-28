#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>
#include <algorithm>
#include <mutex>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();

    // Command is retained between key presses
    static std::vector<float> cmd = {0.0f, 0.0f, 0.0f};

    // Used for detecting a new key press
    static std::string last_key = "";

    // Step size
    constexpr float VX_STEP  = 0.1f;
    constexpr float VY_STEP  = 0.1f;
    constexpr float YAW_STEP = 0.1f;

    // Limits
    constexpr float VX_MAX  = 2.0f;
    constexpr float VX_MIN  = -1.0f;

    constexpr float VY_MAX  = 0.5f;
    constexpr float VY_MIN  = -0.5f;

    constexpr float YAW_MAX = 0.5f;
    constexpr float YAW_MIN = -0.5f;
    // Only process when a new key is pressed.
    // This prevents one key press from being processed every policy step.
    if (!key.empty() && key != last_key)
    {
	if (key == "w")
	{
	    cmd[0] += VX_STEP;
	}
	else if (key == "s")
	{
	    cmd[0] -= VX_STEP;
	}
	else if (key == "a")
	{
	    cmd[1] += VY_STEP;
	}
	else if (key == "d")
	{
	    cmd[1] -= VY_STEP;
	}
	else if (key == "q")
	{
	    cmd[2] += YAW_STEP;
	}
	else if (key == "e")
	{
	    cmd[2] -= YAW_STEP;
	}
	else if (key == " ")
	{
	    cmd = {0.0f, 0.0f, 0.0f};
	}

        // Clamp commands
	cmd[0] = std::clamp(cmd[0], VX_MIN, VX_MAX);
	cmd[1] = std::clamp(cmd[1], VY_MIN, VY_MAX);
	cmd[2] = std::clamp(cmd[2], YAW_MIN, YAW_MAX);

        spdlog::info(
            "Command | vx: {:.2f} m/s | vy: {:.2f} m/s | yaw: {:.2f} rad/s",
            cmd[0], cmd[1], cmd[2]
        );
    }

    last_key = key;

    env->robot->data.velocity_command_b << cmd[0], cmd[1], cmd[2];

    return cmd;
}

// Straight-line heading hold for long runs (2026-09-28). Same keys as keyboard_velocity_commands for
// vx (w/s) and vy (a/d, for manual lane correction), but the yaw-rate command is closed-loop:
//     wz = clamp(kp * wrap(yaw_ref - yaw), -YAW_MAX, YAW_MAX)
// with yaw from the IMU quaternion. The reference is captured on the first call (policy start) and
// re-captured with 'h'; 'g' releases it (wz = 0, plain open loop); space zeroes vx/vy but keeps the
// hold. The policy drifts 0.01-0.03 rad/s at wz = 0 even in Isaac, which is several full turns
// over 30 min, so an open-loop yaw command cannot keep a heading. deploy.yaml params (optional):
//   heading_kp (default 1.0; Isaac Lab's heading_control_stiffness is 0.5), vx0 (initial vx, default 0).
// Change the "velocity_commands" observation name in deploy.yaml to "heading_hold_velocity_commands".
REGISTER_OBSERVATION(heading_hold_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    static std::string last_key = "";
    static bool armed = false;
    static float yaw_ref = 0.0f;
    static int tick = 0;
    static int64_t last_call_ms = 0;

    constexpr float VX_STEP = 0.1f, VY_STEP = 0.1f;
    constexpr float VX_MAX = 2.0f, VX_MIN = -1.0f, VY_MAX = 0.5f, VY_MIN = -0.5f, YAW_MAX = 0.5f;

    float kp = 1.0f;
    try { kp = params["heading_kp"].as<float>(); } catch(const std::exception&) {}

    const auto & q = env->robot->data.root_quat_w;  // IMU orientation, world frame
    const float yaw = std::atan2(2.0f * (q.w() * q.z() + q.x() * q.y()),
                                 1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z()));

    // Observations are computed once at construction and then only while this state runs, so a gap
    // in calls marks a (re)entry into the RL state: capture the heading there, not at program start.
    const int64_t now_ms = unitree::common::GetCurrentTimeMillisecond();
    if (now_ms - last_call_ms > 500)
    {
        armed = true; yaw_ref = yaw; cmd = {0.0f, 0.0f, 0.0f};
        try { cmd[0] = std::clamp(params["vx0"].as<float>(), VX_MIN, VX_MAX); } catch(const std::exception&) {}
        spdlog::info("[heading-hold] armed at yaw {:.3f} rad, kp {:.2f}, vx {:.2f} (h: re-arm, g: release)", yaw_ref, kp, cmd[0]);
    }
    last_call_ms = now_ms;

    if (!key.empty() && key != last_key)
    {
        if      (key == "w") cmd[0] += VX_STEP;
        else if (key == "s") cmd[0] -= VX_STEP;
        else if (key == "a") cmd[1] += VY_STEP;
        else if (key == "d") cmd[1] -= VY_STEP;
        else if (key == " ") { cmd[0] = 0.0f; cmd[1] = 0.0f; }
        else if (key == "h") { yaw_ref = yaw; armed = true; spdlog::info("[heading-hold] re-armed at yaw {:.3f} rad", yaw_ref); }
        else if (key == "g") { armed = false; spdlog::info("[heading-hold] released"); }
        cmd[0] = std::clamp(cmd[0], VX_MIN, VX_MAX);
        cmd[1] = std::clamp(cmd[1], VY_MIN, VY_MAX);
        spdlog::info("Command | vx: {:.2f} m/s | vy: {:.2f} m/s | heading hold: {}", cmd[0], cmd[1], armed ? "on" : "off");
    }
    last_key = key;

    float err = yaw_ref - yaw;
    err = std::atan2(std::sin(err), std::cos(err));  // wrap to [-pi, pi]
    cmd[2] = armed ? std::clamp(kp * err, -YAW_MAX, YAW_MAX) : 0.0f;

    if (++tick % 250 == 0)  // policy runs at 50 Hz -> every 5 s
    {
        spdlog::info("[heading-hold] yaw {:.3f} ref {:.3f} err {:+.3f} rad ({:+.1f} deg) | wz cmd {:+.3f} actual {:+.3f} rad/s",
                     yaw, yaw_ref, err, err * 57.2958f, cmd[2], env->robot->data.root_ang_vel_b[2]);
    }

    env->robot->data.velocity_command_b << cmd[0], cmd[1], cmd[2];
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");
    highstate = std::make_shared<unitree::robot::go2::subscription::SportModeState>(
        "rt/sportmodestate"
    );

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );

    // sim2real motor protection replica (see motor_limits_exceeded)
    if(auto protect = cfg["protect"])
    {
        if(protect["torque_limit"]) protect_tau_limit_ = protect["torque_limit"].as<std::vector<float>>();
        if(protect["upper_joint_vel_limit"]) protect_upper_dq_limit_ = protect["upper_joint_vel_limit"].as<float>();
        if(protect["clamp_ratio"]) protect_clamp_ratio_ = protect["clamp_ratio"].as<float>();
        if(protect["warn_ratio"]) protect_warn_ratio_ = protect["warn_ratio"].as<float>();

        if(!protect_tau_limit_.empty() || protect_upper_dq_limit_ > 0.0f)
        {
            this->registered_checks.emplace_back(
                std::make_pair(
                    [this]()->bool{ return this->motor_limits_exceeded(); },
                    FSMStringMap.right.at("Passive")
                )
            );
            spdlog::info("Motor protection: kill on {} joint torque limits, upper-body |dq| limit {:.1f} rad/s, "
                         "command clamp at {:.0f}% of limit",
                         protect_tau_limit_.size(), protect_upper_dq_limit_, 100.0f * protect_clamp_ratio_);
        }
    }
    else
    {
        spdlog::info("Motor protection: disabled (no 'protect' block in config)");
    }
}

void State_RLBase::apply_torque_clamp()
{
    if(protect_tau_limit_.empty() || protect_clamp_ratio_ <= 0.0f) return;

    const auto & kp_cfg = env->robot->data.joint_stiffness;  // SDK order, from deploy.yaml
    const auto & kd_cfg = env->robot->data.joint_damping;

    std::lock_guard<std::mutex> lock(lowstate->mutex_);
    const auto & ms = lowstate->msg_.motor_state();

    for(size_t i(0); i < protect_tau_limit_.size() && i < ms.size() && i < kp_cfg.size(); ++i)
    {
        const float lim = protect_tau_limit_[i] * protect_clamp_ratio_;
        if(lim <= 0.0f) continue;

        auto & cmd = lowcmd->msg_.motor_cmd()[i];

        // Same estimate the sim2real command writer makes: PD torque from the latest state.
        // cmd.q() is the policy target, rewritten every tick by run() before this call, so a
        // previous tick's clamp (which zeroes it) cannot leak into this one.
        const float tau = kp_cfg[i] * (cmd.q() - ms[i].q()) + kd_cfg[i] * (0.0f - ms[i].dq());

        if(std::fabs(tau) > lim)
        {
            cmd.tau() = std::clamp(tau, -lim, lim);
            cmd.q()   = 0.0f;
            cmd.dq()  = 0.0f;
            cmd.kp()  = 1e-5f;
            cmd.kd()  = 1e-5f;
        }
        else  // restore the PD command a previous tick may have overwritten
        {
            cmd.tau() = 0.0f;
            cmd.dq()  = 0.0f;
            cmd.kp()  = kp_cfg[i];
            cmd.kd()  = kd_cfg[i];
        }
    }
}

bool State_RLBase::motor_limits_exceeded()
{
    if(protect_tau_limit_.empty() && protect_upper_dq_limit_ <= 0.0f) return false;

    std::lock_guard<std::mutex> lock(lowstate->mutex_);
    const auto & ms = lowstate->msg_.motor_state();

    for(size_t i(0); i < protect_tau_limit_.size() && i < ms.size(); ++i)
    {
        const float lim = protect_tau_limit_[i];
        if(lim <= 0.0f) continue;

        const float tau = std::fabs(ms[i].tau_est());
        if(tau > lim)
        {
            spdlog::critical("[protect] torque: sdk{} |tau_est| {:.1f} > {:.1f} Nm -> Passive", i, tau, lim);
            return true;
        }
        // MuJoCo reports the post-saturation actuator force, so |tau_est| can sit exactly at the
        // limit without ever exceeding it. Track how close it gets so a no-kill run is still readable.
        if(tau / lim > protect_peak_ratio_) { protect_peak_ratio_ = tau / lim; protect_peak_idx_ = int(i); }
    }

    if(protect_upper_dq_limit_ > 0.0f)
    {
        for(size_t i(12); i < ms.size(); ++i)  // upper body: waist + arms
        {
            const float dq = std::fabs(ms[i].dq());
            if(dq > protect_upper_dq_limit_)
            {
                spdlog::critical("[protect] upper-body velocity: sdk{} |dq| {:.1f} > {:.1f} rad/s -> Passive",
                                 i, dq, protect_upper_dq_limit_);
                return true;
            }
        }
    }

    if(protect_warn_ratio_ > 0.0f && ++protect_tick_ % 500 == 0)  // FSM runs at 1 kHz
    {
        if(protect_peak_idx_ >= 0 && protect_peak_ratio_ >= protect_warn_ratio_)
        {
            spdlog::warn("[protect] peak torque sdk{} at {:.0f}% of {:.0f} Nm over the last 0.5 s",
                         protect_peak_idx_, 100.0f * protect_peak_ratio_, protect_tau_limit_[protect_peak_idx_]);
        }
        protect_peak_ratio_ = 0.0f;
        protect_peak_idx_ = -1;
    }

    return false;
}

void State_RLBase::sample_velocity()
{
    const auto & q = env->robot->data.root_quat_w;
    const double yaw = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                                  1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
    if (!yaw_init_) { yaw_prev_ = yaw; yaw_unwrapped_ = yaw; yaw_init_ = true; }
    double d = yaw - yaw_prev_;
    d = std::atan2(std::sin(d), std::cos(d));  // wrap the increment, accumulate unwrapped
    yaw_unwrapped_ += d; yaw_prev_ = yaw;

    VelSample sm{ (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3, yaw_unwrapped_, 0.0f, 0.0f, false };
    if (!highstate->isTimeout())
    {
        std::lock_guard<std::mutex> lock(highstate->mutex_);
        const Eigen::Vector3f v = highstate->velocity();
        sm.vx_w = v[0]; sm.vy_w = v[1]; sm.has_lin = true;
    }
    vel_hist_.push_back(sm);
    const size_t window = std::max<size_t>(2, (size_t)std::lround(1.0 / env->step_dt));
    while (vel_hist_.size() > window) vel_hist_.pop_front();
}

void State_RLBase::log_velocity()
{
    const auto command = env->robot->data.velocity_command_b;
    const float gyro_z = env->robot->data.root_ang_vel_b[2];  // pelvis-frame instant, for reference only

    if (vel_hist_.size() < 2)
    {
        spdlog::info("Velocity | command: vx={:.2f}, vy={:.2f}, yaw={:.2f} | actual: (collecting)", command[0], command[1], command[2]);
        return;
    }
    const auto & a = vel_hist_.front(); const auto & b = vel_hist_.back();
    const double dt = b.t - a.t;
    const double heading_rate = dt > 1e-3 ? (b.yaw - a.yaw) / dt : 0.0;

    // Mean world velocity over the window, expressed in the heading frame (yaw only, no roll/pitch):
    // this is the "robot" forward/lateral speed the command refers to.
    double sx = 0.0, sy = 0.0, syaw = 0.0; int n = 0;
    for (const auto & s : vel_hist_) { if (s.has_lin) { sx += s.vx_w; sy += s.vy_w; syaw += s.yaw; ++n; } }
    if (n == 0)
    {
        spdlog::info("Velocity | command: vx={:.2f}, vy={:.2f}, yaw={:.2f} | actual (1 s avg): vx=N/A, vy=N/A, "
                     "heading rate={:+.3f} rad/s | pelvis gyro z={:+.2f}",
                     command[0], command[1], command[2], heading_rate, gyro_z);
        return;
    }
    const double mx = sx / n, my = sy / n, myaw = syaw / n;
    const double vx_h =  std::cos(myaw) * mx + std::sin(myaw) * my;
    const double vy_h = -std::sin(myaw) * mx + std::cos(myaw) * my;

    spdlog::info("Velocity | command: vx={:.2f}, vy={:.2f}, yaw={:.2f} | actual (1 s avg): vx={:.2f}, vy={:.2f}, "
                 "heading rate={:+.3f} rad/s | pelvis gyro z={:+.2f}",
                 command[0], command[1], command[2], vx_h, vy_h, heading_rate, gyro_z);
}

void State_RLBase::open_logger()
{
    const int n = env->robot->data.joint_ids_map.size();
    std::vector<std::string> columns{"t"};
    for(int i(0); i < n; i++) columns.push_back(fmt::format("qdes_{:02d}", i));
    for(int i(0); i < n; i++) columns.push_back(fmt::format("q_{:02d}", i));
    for(int i(0); i < n; i++) columns.push_back(fmt::format("dq_{:02d}", i));
    for(int i(0); i < n; i++) columns.push_back(fmt::format("tau_{:02d}", i));
    logger.open(columns);

    if(logger.enabled())
    {
        std::string mapping;
        for(int i(0); i < n; i++) {
            mapping += fmt::format("{}->sdk{} ", i, int(env->robot->data.joint_ids_map[i]));
        }
        spdlog::info("TelemetryLogger: policy joint order -> sdk index: {}", mapping);
    }
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    const int n = env->robot->data.joint_ids_map.size();
    for(int i(0); i < n; i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

    apply_torque_clamp();  // must run after the targets are written, before post_run() publishes

    if(logger.enabled())
    {
        std::vector<float> row;
        row.reserve(4 * n);
        for(int i(0); i < n; i++) row.push_back(action[i]);
        {
            std::lock_guard<std::mutex> lock(lowstate->mutex_);
            for(int i(0); i < n; i++) {
                row.push_back(lowstate->msg_.motor_state()[env->robot->data.joint_ids_map[i]].q());
            }
            for(int i(0); i < n; i++) {
                row.push_back(lowstate->msg_.motor_state()[env->robot->data.joint_ids_map[i]].dq());
            }
            for(int i(0); i < n; i++) {
                row.push_back(lowstate->msg_.motor_state()[env->robot->data.joint_ids_map[i]].tau_est());
            }
        }
        logger.append(row);
    }
}

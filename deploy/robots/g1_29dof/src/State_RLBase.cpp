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
}

void State_RLBase::log_velocity()
{
    const auto command = env->robot->data.velocity_command_b;
    const float actual_yaw_rate = env->robot->data.root_ang_vel_b[2];

    if (highstate->isTimeout())
    {
        spdlog::info(
            "Velocity | command: vx={:.2f}, vy={:.2f}, yaw={:.2f} | "
            "actual: vx=N/A, vy=N/A, yaw={:.2f}",
            command[0], command[1], command[2], actual_yaw_rate
        );
        return;
    }

    Eigen::Vector3f actual_linear_velocity_w;
    {
        std::lock_guard<std::mutex> lock(highstate->mutex_);
        actual_linear_velocity_w = highstate->velocity();
    }

    const Eigen::Vector3f actual_linear_velocity_b =
        env->robot->data.root_quat_w.conjugate() * actual_linear_velocity_w;

    spdlog::info(
        "Velocity | command: vx={:.2f}, vy={:.2f}, yaw={:.2f} | "
        "actual: vx={:.2f}, vy={:.2f}, yaw={:.2f}",
        command[0], command[1], command[2],
        actual_linear_velocity_b[0], actual_linear_velocity_b[1], actual_yaw_rate
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}

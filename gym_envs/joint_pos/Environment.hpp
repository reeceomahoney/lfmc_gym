//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <cstdlib>
#include <set>

#include "RaisimGymEnv.hpp"
#include "actuation_dynamics/Actuation.hpp"

#include "helpers/Command.hpp"
#include "helpers/Observation.hpp"
#include "helpers/Reward.hpp"
#include "helpers/Visualization.hpp"


namespace raisim {

    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable),
                normalDistribution_(0, 1), uniformRealDistribution_(-1, 1),
                actuation_(resourceDir + "/parameters/anymal_c_actuation", Eigen::Vector2d{1., 0.1}, 100., 12),
                velocityCommandHandler_(cfg["velocity_command"], cfg["control_dt"].template As<double>()),
                rewardHandler_(cfg["control_dt"].template As<float>(),
                               cfg["simulation_dt"].template As<float>()),
                visualizationHandler_(visualizable) {

            /// create world
            world_ = std::make_unique<raisim::World>();

            setControlTimeStep(cfg["control_dt"].template As<double>());
            setSimulationTimeStep(cfg["simulation_dt"].template As<double>());

            /// add objects
            robot_ = world_->addArticulatedSystem(resourceDir_ + "/models/anymal_d/urdf/anymal_d.urdf");
            robot_->setName("anymal_c");
            robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

            baseMassMean_ = robot_->getMass(0);

            auto ground = world_->addGround(0., "ground_material");

            /// get robot data
            gcDim_ = static_cast<int>(robot_->getGeneralizedCoordinateDim());
            gvDim_ = static_cast<int>(robot_->getDOF());
            nJoints_ = gvDim_ - 6;

            /// initialize containers
            gc_.setZero(gcDim_);
            gc_init_.setZero(gcDim_);
            gv_.setZero(gvDim_);
            gv_init_.setZero(gvDim_);
            pTarget_.setZero(gcDim_);
            vTarget_.setZero(gvDim_);
            pTarget12_.setZero(nJoints_);
            prevPTarget12_.setZero(nJoints_);
            gf_.setZero(gvDim_);

            /// this is nominal configuration of anymal_c
            gc_init_ = observationHandler_.getNominalGeneralizedCoordinates();

            useActuatorNetwork_ = cfg["use_actuator_network"].template As<bool>();

            /// set pd gains
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero();
            jointDgain.setZero();

            if (!useActuatorNetwork_) {
                jointPgain.tail(nJoints_).setConstant(80.0);
                jointDgain.tail(nJoints_).setConstant(2.0);
            }

            robot_->setPdGains(jointPgain, jointDgain);
            robot_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_ = 48;
            actionDim_ = nJoints_;
            actionMean_.setZero(actionDim_);
            actionStd_.setZero(actionDim_);
            obDouble_.setZero(obDim_);
            prevPTarget12_ = actionMean_;

            /// action scaling
            actionMean_ = gc_init_.tail(nJoints_);
            actionStd_.setConstant(cfg["action_scaling"].template As<double>());

            /// Reward coefficients
            rewards_.initializeFromConfigurationFile(cfg["reward"]);
            rewards_.setLimits(-100.f, 100.f);

            Yaml::Node cfgCurriculum = cfg["curriculum"];
            rewardCurriculumFactor_ = cfgCurriculum["reward_factor"].template As<float>();
            rewardAdvanceRate_ = cfgCurriculum["reward_advance_rate"].template As<float>();

            /// Set the material property for each of the collision bodies of the robot
            for (auto &collisionBody: robot_->getCollisionBodies()) {
                if (collisionBody.colObj->name.find("FOOT") != std::string::npos) {
                    collisionBody.setMaterial("foot_material");
                } else {
                    collisionBody.setMaterial("robot_material");
                }
            }

            auto materialPairGroundFootProperties =
                    world_->getMaterialPairProperties("ground_material", "foot_material");
            world_->setMaterialPairProp(
                    "ground_material", "foot_material",
                    0.6, materialPairGroundFootProperties.c_r,
                    materialPairGroundFootProperties.r_th
            );

            auto materialPairGroundRobotProperties =
                    world_->getMaterialPairProperties("ground_material", "robot_material");
            world_->setMaterialPairProp("ground_material", "robot_material",
                                        0.4, materialPairGroundRobotProperties.c_r,
                                        materialPairGroundRobotProperties.r_th
            );

            /// Episode Length
            maxEpisodeLength_ = std::floor(
                    cfg_["max_time"].template As<double>() / cfg_["control_dt"].template As<double>());

            /// Dynamics Randomization
            enableDynamicsRandomization_ = cfg["enable_dynamics_randomization"].template As<bool>();

            /// visualize if it is the first environment
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer(cfg["server"]["port"].template As<int>());
                server_->focusOn(robot_);
                visualizationHandler_.setServer(server_);
            }
        }

        void init() final {}

        void reset() final {
            Eigen::VectorXd gc = gc_init_, gv = gv_init_;

            if (enableDynamicsRandomization_) {
                gc[2] += 0.05 * std::abs(uniformRealDistribution_(gen_));

                gc.tail(12) +=
                        0.1 * Eigen::VectorXd::NullaryExpr(12, [&]() { return uniformRealDistribution_(gen_); });

                gv.head(3) += 0.1 * Eigen::VectorXd::NullaryExpr(3, [&]() { return uniformRealDistribution_(gen_); });
                gv.segment(3, 3) +=
                        0.15 * Eigen::VectorXd::NullaryExpr(3, [&]() { return uniformRealDistribution_(gen_); });
                gv.tail(12) += 0.05 * Eigen::VectorXd::NullaryExpr(12, [&]() { return uniformRealDistribution_(gen_); });
            }

            robot_->setState(gc, gv);

            if (enableDynamicsRandomization_) {
                robot_->setMass(0, baseMassMean_ + std::clamp(normalDistribution_(gen_) * 10., -15., 15.));
                robot_->updateMassInfo();

                if (useActuatorNetwork_) {
                    actuationPositionErrorInputScaling_ = std::clamp(1. + normalDistribution_(gen_) * 0.05, 0.975, 1.025);
                    actuationVelocityInputScaling_ = std::clamp(1. + normalDistribution_(gen_) * 0.025, 0.975, 1.025);
                    actuationOutputTorqueScaling_ = std::clamp(1. + normalDistribution_(gen_) * 0.05, 0.95, 1.05);
                } else {
                    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);

                    jointPgain.setZero();
                    jointDgain.setZero();
                    
                    jointPgain.tail(nJoints_).setConstant(80.0 * (std::clamp(1. + normalDistribution_(gen_) * 0.05, 0.975, 1.025)));
                    jointDgain.tail(nJoints_).setConstant(2.0 * (std::clamp(1. + normalDistribution_(gen_) * 0.05, 0.975, 1.025)));

                    robot_->setPdGains(jointPgain, jointDgain);
                }
            }

            rewardHandler_.reset();

            velocityCommandHandler_.reset(robot_->getBaseOrientation().e());
            prevPTarget12_ = actionMean_;

            observationHandler_.reset(robot_, velocityCommandHandler_.getVelocityCommand());
            actuation_.reset();

            stepCount_ = 0;
        }

        bool conditionalReset() final {
            if (stepCount_ >= maxEpisodeLength_) {
                reset();
                return true;
            }

            return false;
        }

        float step(const Eigen::Ref<EigenVec> &action) final {
            /// action scaling
            pTarget12_ = action.cast<double>();
            pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
            pTarget12_ += actionMean_;
            pTarget_.tail(nJoints_) = pTarget12_;

            float jointTorqueSquaredNorm = 0.f;

            for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
                if (server_) server_->lockVisualizationServerMutex();

                /// Use the actuation network to compute torques
                if (i % 2 == 0 && useActuatorNetwork_) {
                    robot_->getState(gc_, gv_);

                    gf_.setZero();

                    gf_.tail(12) = actuationOutputTorqueScaling_ * actuation_.getActuationTorques(
                            actuationPositionErrorInputScaling_ * (pTarget12_ - gc_.tail(12)),
                            actuationVelocityInputScaling_ * gv_.tail(12)
                    ).cwiseMax(-80.).cwiseMin(80.);

                    robot_->setGeneralizedForce(gf_);
                } else if (!useActuatorNetwork_) {
                    robot_->setPdTarget(pTarget_, Eigen::VectorXd::Zero(gvDim_));
                }

                world_->integrate();

                jointTorqueSquaredNorm = robot_->getGeneralizedForce().squaredNorm();

                observationHandler_.updateObservation(robot_, pTarget12_);
                rewardHandler_.simStep(robot_, observationHandler_, velocityCommandHandler_, jointTorqueSquaredNorm);

                if (visualizing_ && visualizable_) {
                    visualizationHandler_.updateVelocityVisual(robot_, observationHandler_, server_);
                }

                if (server_) server_->unlockVisualizationServerMutex();
            }

            velocityCommandHandler_.step(robot_->getBaseOrientation().e());
            observationHandler_.updateVelocityCommand(velocityCommandHandler_.getVelocityCommand());

            recordRewards();

            ++stepCount_;
            return std::fmax(rewards_.sum() * static_cast<float>(control_dt_), 0.f);
        }

        void observe(Eigen::Ref<EigenVec> ob) final {
            /// convert it to float
            ob = observationHandler_.getObservation().cast<float>();
        }

        void getTorques(Eigen::Ref<EigenVec> torques) {
            torques = gf_.cast<float>();
        }

        void recordRewards() {
            rewardHandler_.computeRewards(observationHandler_);
            rewards_.record("base_orientation", rewardCurriculumFactor_ * rewardHandler_.getBaseOrientationReward());
            rewards_.record("base_linear_velocity_tracking", rewardHandler_.getBaseLinearVelocityTrackingReward());
            rewards_.record("base_angular_velocity_tracking", rewardHandler_.getBaseAngularVelocityTrackingReward());
            rewards_.record("joint_torque", rewardCurriculumFactor_ * rewardHandler_.getJointTorqueReward());
            rewards_.record("joint_velocity", rewardCurriculumFactor_ * rewardHandler_.getJointVelocityReward());
            rewards_.record("action_smoothness",
                            rewardCurriculumFactor_ * rewardHandler_.getActionSmoothnessReward());
            rewards_.record("feet_clearance", rewardHandler_.getFeetClearanceReward());
            rewards_.record("trotting", rewardHandler_.getTrottingReward());
            rewards_.record("feet_slip", rewardCurriculumFactor_ * rewardHandler_.getFeetSlipReward());
            rewards_.record("joint_position", rewardCurriculumFactor_ * rewardHandler_.getJointPositionReward());
            rewards_.record("pronking", rewardCurriculumFactor_ * rewardHandler_.getPronkingReward());
            rewards_.record("base_height", rewardCurriculumFactor_ * rewardHandler_.getBaseHeightReward());
            rewards_.record("symmetry_z", rewardCurriculumFactor_ * rewardHandler_.getSymmetryZReward());
            rewards_.record("feet_deviation", rewardCurriculumFactor_ * rewardHandler_.getFeetDeviationReward());
            rewards_.record("joint_jerk", rewardCurriculumFactor_ * rewardHandler_.getJointJerkReward());
            rewards_.record("vertical_linear_velocity", rewardCurriculumFactor_ * rewardHandler_.getVerticalLinearVelocityReward());
            rewards_.record("horizontal_angular_velocity",
                            rewardCurriculumFactor_ * rewardHandler_.getHorizontalAngularVelocityReward());
            rewardHandler_.clearBuffers();
        }

        bool isTerminalState(float &terminalReward) final {
            terminalReward = terminalRewardCoeff_;

            /// if the contact body is not feet
            for (auto &contact : robot_->getContacts()) {
            if ((contact.getCollisionBodyA()->material == "ground_material" &&
                contact.getCollisionBodyB()->material != "foot_material") ||
                (contact.getCollisionBodyA()->material == "foot_material" &&
                contact.getCollisionBodyB()->material != "ground_material")) {
                return true;
            }
            }

            terminalReward = 0.f;
            return false;
        }

        void curriculumUpdate() final {
            rewardCurriculumFactor_ = std::pow(rewardCurriculumFactor_, rewardAdvanceRate_);
        }

        void setMaxEpisodeLength(const double &timeInSeconds) final {
            maxEpisodeLength_ = std::floor(timeInSeconds / cfg_["control_dt"].template As<double>());
        }

        void setSeed(int seed) final {
            gen_.seed(seed);
            srand(seed);

            velocityCommandHandler_.setSeed(seed);
        }

        void turnOnVisualization() final {
            server_->wakeup();
            visualizing_ = true;
        }

        void turnOffVisualization() final {
            server_->hibernate();
            visualizing_ = false;
        }

        void getBasePosition(Eigen::Ref<EigenVec> basePos) {
            basePos = robot_->getBasePosition().e().cast<float>();
        }

        void getBaseOrientation(Eigen::Ref<EigenVec> baseOrientation) {
            baseOrientation = observationHandler_.getGeneralizedCoordinate().segment(3, 4).cast<float>();
        }

    private:
        int gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false, visualizing_ = false;
        raisim::ArticulatedSystem *robot_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, prevPTarget12_, gf_;
        float terminalRewardCoeff_ = 0.f;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;

        // Actuator network
        Actuation actuation_;

        // Conditional Reset
        int maxEpisodeLength_;
        int stepCount_ = 0;

        /// Dynamics Randomization Parameters
        bool enableDynamicsRandomization_ = false;
        double baseMassMean_;

        bool useActuatorNetwork_ = true;

        double actuationPositionErrorInputScaling_ = 1.;
        double actuationVelocityInputScaling_ = 1.;
        double actuationOutputTorqueScaling_ = 1.;

        // Randomization engine
        std::normal_distribution<double> normalDistribution_;
        std::uniform_real_distribution<double> uniformRealDistribution_;
        thread_local static std::mt19937 gen_;

        // Helper Classes
        VelocityCommand velocityCommandHandler_;
        ObservationHandler observationHandler_;
        RewardHandler rewardHandler_;
        VisualizationHandler visualizationHandler_;

        // Curriculum Factors
        float rewardCurriculumFactor_ = 1.f, rewardAdvanceRate_ = 1.f;
    };

    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
}

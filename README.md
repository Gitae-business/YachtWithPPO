[English](#english) | [한국어](#한국어)

---

## <a name="english"></a>Yacht with PPO (English)

### 1. Overview

This project aims to train an AI agent to play the popular dice game, Yacht, using Reinforcement Learning. It is based on the Proximal Policy Optimization (PPO) algorithm and is designed for the agent to interact with the game through a custom environment similar to OpenAI Gym.

### 2. Directory Explanation

The main components of the project are as follows:

*   `config.py`: Defines hyperparameters and other configuration settings used for training.
*   `main.py`: The main entry point script to start training the PPO agent.
*   `checkpoints/`: The directory where checkpoints of the trained model are saved and loaded during training.
*   `training_log.csv`: A CSV log file where the agent's training progress (average rewards and final scores per episode) is recorded.
*   `requirements.txt`: A list of Python libraries required to run the project.

#### 2-1. PPO and Agent

The `ppo/` directory contains the core implementation of the PPO algorithm and agent-related code.

*   `ppo/model.py`: Defines the Actor-Critic neural network architecture that serves as the agent's brain. The model consists of a Multi-Layer Perceptron (MLP) with 3 hidden layers (256 neurons each).
*   `ppo/ppo_agent.py`: Implements the core logic of the PPO agent. This includes experience storage, advantage calculation using Generalized Advantage Estimation (GAE), and the update mechanisms for the policy (Actor) and value (Critic) networks.
*   `ppo/train.py`: The main training script that orchestrates the interaction with the environment, agent updates, logging of training progress, and saving/loading of model checkpoints.

#### 2-2. Yacht Game Logic

The `game/` directory implements the Yacht game environment and game rule logic.

*   `game/yacht_env.py`: A custom OpenAI Gym-like environment for the Yacht game. It defines the observation space, which includes dice values, used categories, and remaining rolls, and the action space, which includes reroll combinations (32) and scoring categories (12).
*   `game/dice.py`: Responsible for the dice rolling mechanism of the game.
*   `game/scoreboard.py`: Manages the game scoring logic, including calculating scores for each category and the upper section bonus.

### 3. Inference and Learning Method

This project trains the agent using the **Proximal Policy Optimization (PPO)** algorithm. PPO is based on an Actor-Critic architecture, where the Actor learns a policy to select the optimal action in a given state, and the Critic estimates the value of the current state (the expected total future reward).

**Observation**: The agent receives a vector observation that includes the current dice values, the used scoring categories, and the number of rolls remaining in the current turn.

**Action**: The agent performs one of two types of actions: choosing one of 32 dice reroll combinations or selecting one of 12 scoring categories.

#### Detailed Reward System

To maximize the agent's learning efficiency and performance, a sophisticated reward system was designed. All rewards are normalized based on the highest possible score in the game, the **Yacht (50 points)**, to enhance training stability. The agent receives immediate and clear feedback on the consequences of each action.

| Situation | Reward Logic | Design Intention |
| :--- | :--- | :--- |
| **Dice Reroll** | `(Potential Score After Reroll - Potential Score Before Reroll) / 50.0` | Encourages rerolls that can lead to higher scores. |
| **Score Decrease After Reroll** | The above calculation is `* 2` (penalty enhancement) | Strongly discourages bad rerolls that reduce potential score. |
| **Score Unchanged After Reroll** | `+0.05` (small bonus) | Encourages exploration to find better combinations, even if the score doesn't change, preventing stagnation. |
| **Scoring a Category** | `Score Obtained / 50.0` | Directly teaches the value of actions that yield high scores. |
| **Scoring in Upper Section** | `+ (Score Obtained * 0.1) / 50.0` added to the above reward | Guides the agent to fill the upper section (Aces to Sixes) to strategically aim for the 35-point bonus. |
| **Upper Section 35-Point Bonus** | `+ 35.0 / 50.0` | Strongly motivates the agent to achieve this key game bonus. |
| **Game End** | `+ Final Total Score / 50.0` | Reflects the overall performance of the entire episode (game), encouraging long-term high-score strategies over short-term gains. |

*Note: Invalid actions, such as selecting an already used category, result in a reward of 0, naturally teaching the agent to avoid them.*

#### Key Hyperparameters (`config.py`)

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `LR` | `2e-4` | Learning Rate |
| `GAMMA` | `0.99` | Discount Factor |
| `CLIP_EPSILON` | `0.2` | PPO Clipping Parameter |
| `GAE_LAMBDA` | `0.95` | GAE Lambda |
| `PPO_EPOCHS` | `10` | Number of mini-batch iterations during PPO update |
| `MINI_BATCH_SIZE` | `128` | Size of the mini-batch used for PPO updates |
| `UPDATE_TIMESTEPS` | `4096` | Number of steps to collect experience before a policy update |

### 4. Performance

The goal of this project is to achieve performance that far exceeds a random agent (average 50-80 points) in the Yacht game. A well-trained agent is expected to consistently achieve an average final score of **150-200 points or more**. This is comparable to or better than the performance of a typical human player.

Training is currently ongoing, and efforts to improve performance through reward shaping, model architecture adjustments, and hyperparameter tuning are continuous.

### 5. Usage

#### 5-1. Installation

This project recommends using `uv` for package management. First, install `uv`, then use the following command to install the required libraries.

```bash
# Install dependencies from requirements.txt
uv pip install -r requirements.txt
```

#### 5-2. Start Training

From the project's root directory, run the `main.py` script to start training the PPO agent.

```bash
python main.py
```

*   Training progress is periodically printed to the console and logged in the `training_log.csv` file.
*   Model checkpoints are periodically saved to `checkpoints/latest_checkpoint.pth`, overwriting the previous checkpoint. This file is automatically loaded when resuming training.

#### 5-3. Human Play Mode

You can play the Yacht game directly in the terminal using the `play_yacht.py` script. This helps in understanding how the environment works and simulating the agent's actions.

```bash
python play_yacht.py
```

### 6. Conclusion

This project demonstrates the process of developing a reinforcement learning agent for the Yacht game. It highlights the iterative nature of reinforcement learning development, including reward shaping, hyperparameter tuning, and model architecture selection. Through continuous improvement, the agent will be able to learn complex game strategies and achieve high performance.

---

## <a name="한국어"></a>Yacht with PPO (한국어)

### 1. 개요 (Overview)

이 프로젝트는 강화 학습(Reinforcement Learning)을 사용하여 인기 있는 주사위 게임인 Yacht를 플레이하는 AI 에이전트를 훈련하는 것을 목표로 합니다. 근사 정책 최적화(Proximal Policy Optimization, PPO) 알고리즘을 기반으로 하며, OpenAI Gym과 유사한 사용자 정의 환경을 통해 에이전트가 게임과 상호작용하도록 설계되었습니다.

### 2. 디렉토리 설명 (Directory Explanation)

프로젝트의 주요 구성 요소는 다음과 같습니다.

*   `config.py`: 학습에 사용되는 하이퍼파라미터 및 기타 구성 설정을 정의합니다.
*   `main.py`: PPO 에이전트 학습을 시작하는 메인 진입점 스크립트입니다.
*   `checkpoints/`: 훈련 중 학습된 모델의 체크포인트가 저장되고 불러와지는 디렉토리입니다.
*   `training_log.csv`: 에이전트의 학습 진행 상황(에피소드별 평균 보상 및 최종 점수)이 기록되는 CSV 로그 파일입니다.
*   `requirements.txt`: 프로젝트 실행에 필요한 Python 라이브러리 목록입니다.

#### 2-1. PPO 및 Agent

`ppo/` 디렉토리에는 PPO 알고리즘의 핵심 구현과 에이전트 관련 코드가 포함되어 있습니다.

*   `ppo/model.py`: 에이전트의 두뇌 역할을 하는 Actor-Critic 신경망 아키텍처를 정의합니다. 이 모델은 3개의 은닉 레이어(각 256개 뉴런)를 가진 다층 퍼셉트론(MLP)으로 구성됩니다.
*   `ppo/ppo_agent.py`: PPO 에이전트의 핵심 로직을 구현합니다. 여기에는 경험 저장, 일반화된 어드밴티지 추정(GAE)을 통한 어드밴티지 계산, 그리고 정책(Actor) 및 가치(Critic) 네트워크의 업데이트 메커니즘이 포함됩니다.
*   `ppo/train.py`: 환경과의 상호작용, 에이전트 업데이트, 학습 진행 상황 로깅, 모델 체크포인트 저장 및 불러오기를 조율하는 메인 학습 스크립트입니다.

#### 2-2. Yacht Game logic

`game/` 디렉토리에는 Yacht 게임 환경 및 게임 규칙 관련 로직이 구현되어 있습니다.

*   `game/yacht_env.py`: Yacht 게임을 위한 사용자 정의 OpenAI Gym과 유사한 환경입니다. 주사위 값, 사용된 카테고리, 남은 굴림 횟수를 포함하는 관찰 공간과 리롤 조합(32가지), 점수화 카테고리(12가지)를 포함하는 행동 공간을 정의합니다.
*   `game/dice.py`: 게임의 주사위 굴림 메커니즘을 담당합니다.
*   `game/scoreboard.py`: 게임 점수화 로직을 관리하며, 각 카테고리의 점수 계산 및 상위 섹션 보너스 계산을 포함합니다.

### 3. 추론 및 학습 방식 (Inference and Learning Method)

이 프로젝트는 **근사 정책 최적화(PPO)** 알고리즘을 사용하여 에이전트를 훈련합니다. PPO는 Actor-Critic 아키텍처를 기반으로 하며, Actor는 주어진 상태에서 최적의 행동을 선택하는 정책을 학습하고, Critic은 현재 상태의 가치(미래에 예상되는 총 보상)를 추정합니다.

**관찰(Observation)**: 에이전트는 현재 주사위 값, 사용된 점수 카테고리, 현재 턴에서 남은 굴림 횟수를 포함하는 벡터 형태의 관찰을 받습니다.

**행동(Action)**: 에이전트는 32가지의 주사위 리롤 조합 또는 12가지의 점수화 카테고리 선택 중 하나의 행동을 수행합니다.

#### 보상 시스템 상세 (Detailed Reward System)

에이전트의 학습 효율과 성능을 극대화하기 위해 정교한 보상 시스템을 설계했습니다. 모든 보상은 게임 내 최고 점수인 **Yacht(50점)**를 기준으로 정규화되어 학습 안정성을 높입니다. 에이전트는 각 행동의 결과를 즉각적이고 명확한 피드백으로 받게 됩니다.

| 상황 (Situation) | 보상 계산 로직 (Reward Logic) | 설계 의도 (Design Intention) |
| :--- | :--- | :--- |
| **주사위 리롤 (Reroll)** | `(리롤 후 잠재 점수 - 리롤 전 잠재 점수) / 50.0` | 더 높은 점수를 얻을 수 있는 리롤을 장려합니다. |
| **리롤 후 점수 감소** | 위 계산 결과에 `* 2` (페널티 강화) | 잠재 점수를 깎아 먹는 나쁜 리롤을 강력하게 억제합니다. |
| **리롤 후 점수 동일** | `+0.05` (작은 보너스) | 점수 변화가 없더라도, 현재 상태에 안주하지 않고 더 나은 조합을 찾기 위한 탐색을 장려합니다. |
| **카테고리 점수 획득** | `획득 점수 / 50.0` | 높은 점수를 얻는 행동의 가치를 직접적으로 학습시킵니다. |
| **상위 섹션 점수 획득** | 위 보상에 `+ (획득 점수 * 0.1) / 50.0` 추가 | 상위 섹션(Aces~Sixes)을 채우도록 유도하여 35점 보너스를 전략적으로 노리게 합니다. |
| **상위 섹션 35점 보너스** | `+ 35.0 / 50.0` | 게임의 핵심적인 보너스 목표를 달성하도록 강력하게 동기를 부여합니다. |
| **게임 종료** | `+ 최종 총점 / 50.0` | 에피소드(게임) 전체의 최종 성과를 반영하여, 단기적인 이득보다 장기적인 고득점 전략을 학습하도록 합니다. |

*참고: 이미 사용한 카테고리를 선택하는 등 유효하지 않은 행동은 보상을 0으로 만들어, 에이전트가 자연스럽게 해당 행동을 회피하도록 학습합니다.*

#### 주요 하이퍼파라미터 (config.py)

| 하이퍼파라미터 | 값 | 설명 |
| :--- | :--- | :--- |
| `LR` | `2e-4` | 학습률 (Learning Rate) |
| `GAMMA` | `0.99` | 감가율 (Discount Factor) |
| `CLIP_EPSILON` | `0.2` | PPO 클리핑 파라미터 (PPO Clipping Parameter) |
| `GAE_LAMBDA` | `0.95` | 일반화된 어드밴티지 추정 람다 (GAE Lambda) |
| `PPO_EPOCHS` | `10` | PPO 업데이트 시 미니 배치 반복 횟수 |
| `MINI_BATCH_SIZE` | `128` | PPO 업데이트에 사용되는 미니 배치의 크기 |
| `UPDATE_TIMESTEPS` | `4096` | 정책 업데이트 전 경험을 수집하는 스텝 수 |

### 4. 성능 (Performance)

이 프로젝트의 목표는 Yacht 게임에서 무작위 에이전트(평균 50-80점)를 훨씬 뛰어넘는 성능을 달성하는 것입니다. 잘 훈련된 에이전트는 평균 최종 점수가 **150-200점 이상**을 꾸준히 기록할 것으로 예상됩니다. 이는 일반적인 인간 플레이어의 성능에 필적하거나 그 이상입니다.

현재 학습은 진행 중이며, 보상 설계, 모델 아키텍처, 하이퍼파라미터 튜닝을 통해 성능 개선을 위한 노력이 계속되고 있습니다.

### 5. 사용법 (Usage)

#### 5-1. 설치 (Installation)

이 프로젝트는 `uv`를 사용하여 패키지를 관리하는 것을 권장합니다. 먼저 `uv`를 설치한 후, 다음 명령어를 사용하여 필요한 라이브러리를 설치하세요.

```bash
# requirements.txt 파일로부터 종속성 설치
uv pip install -r requirements.txt
```

#### 5-2. 학습 시작 (Start Training)

프로젝트의 루트 디렉토리에서 `main.py` 스크립트를 실행하여 PPO 에이전트 학습을 시작합니다.

```bash
python main.py
```

*   학습 진행 상황은 콘솔에 주기적으로 출력되며, `training_log.csv` 파일에 기록됩니다.
*   모델 체크포인트는 `checkpoints/latest_checkpoint.pth` 파일에 주기적으로 저장되며, 이전 체크포인트를 덮어씁니다. 학습을 재개할 때 이 파일이 자동으로 불러와집니다.

#### 5-3. 사람 플레이 모드 (Human Play Mode)

`play_yacht.py` 스크립트를 사용하여 터미널에서 직접 Yacht 게임을 플레이해 볼 수 있습니다. 이는 환경의 작동 방식을 이해하고 에이전트의 행동을 시뮬레이션하는 데 도움이 됩니다.

```bash
python play_yacht.py
```

### 6. 결론 (Conclusion)

이 프로젝트는 Yacht 게임을 위한 강화 학습 에이전트를 개발하는 과정을 보여줍니다. 보상 형성, 하이퍼파라미터 튜닝, 모델 아키텍처 선택 등 강화 학습 개발의 반복적인 특성을 강조합니다. 지속적인 개선 노력을 통해 에이전트가 복잡한 게임 전략을 학습하고 높은 성능을 달성할 수 있도록 할 것입니다.

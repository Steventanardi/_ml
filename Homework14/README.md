# 🎯 Exercise 14: CartPole Fixed Strategy (No Learning)

This project solves the **CartPole-v1** problem using a **fixed, hand-written strategy**, not machine learning. The goal is to keep the pole upright as long as possible using only simple rules.

---

## 🧠 Strategy

- Observe the pole's **angle**.
- If leaning **left**, push the cart **left**.
- If leaning **right**, push the cart **right**.
- This simple rule helps the cart respond to the pole's direction.

---

## 📌 Environment Info

- Environment: `CartPole-v1` (from `gymnasium`)
- Observation: `[cart_pos, cart_vel, pole_angle, pole_vel]`
- Action: `0 = left`, `1 = right`

---


https://gymnasium.farama.org/environments/classic_control/cart_pole/


## ▶️ How to Run

### Requirements

```bash
pip install gymnasium[all]


python cartpole_fixed_strategy.py

import argparse
import importlib

import numpy as np
# np.random.seed(3)  # for reproducible Keras operations

from utils import *


parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_to_load', action="store",
                    dest="model_to_load", default='DQN_ep10', help="model name")
parser.add_argument('--stock_name', action="store",
                    dest="stock_name", default='^GSPC_2018', help="stock name")
parser.add_argument('--initial_balance', action="store",
                    dest="initial_balance", default=50000, type=int, help='initial balance')
inputs = parser.parse_args()

model_to_load = inputs.model_to_load
model_name = model_to_load.split('_')[0]
stock_name = inputs.stock_name
initial_balance = inputs.initial_balance
display = True
window_size = 10
action_dict = {0: 'Hold', 1: 'Hold', 2: 'Sell'}

# select evaluation model
model = importlib.import_module(f'agents.{model_name}')


def hold():
    print('Hold')


def buy(t):
    agent.balance -= stock_prices[t]
    agent.inventory.append(stock_prices[t])
    agent.buy_dates.append(t)
    print('Buy:  ${:.2f}'.format(stock_prices[t]))


def sell(t):
    agent.balance += stock_prices[t]
    bought_price = agent.inventory.pop(0)
    profit = stock_prices[t] - bought_price
    global reward
    reward = profit
    agent.sell_dates.append(t)
    print('Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit))


portfolio_return = 0
while portfolio_return == 0:  # a hack to avoid stationary case
    agent = model.Agent(state_dim=13, balance=initial_balance,
                        is_eval=True, model_name=model_to_load)
    stock_prices = stock_close_prices(stock_name)
    trading_period = len(stock_prices) - 1
    state = generate_combined_state(
        0, window_size, stock_prices, agent.balance, len(agent.inventory))

    for t in range(1, trading_period + 1):
        actions = agent.model.predict(state)[0]
        action = agent.act(state)

        next_state = generate_combined_state(
            t, window_size, stock_prices, agent.balance, len(agent.inventory))
        previous_portfolio_value = len(
            agent.inventory) * stock_prices[t] + agent.balance

        # execute position
        if action == 0:
            hold()  # hold
        if action == 1 and agent.balance > stock_prices[t]:
            buy(t)  # buy
        if action == 2 and len(agent.inventory) > 0:
            sell(t)  # sell

        current_portfolio_value = len(
            agent.inventory) * stock_prices[t] + agent.balance
        agent.return_rates.append(
            (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
        agent.portfolio_values.append(current_portfolio_value)
        state = next_state

        done = True if t == trading_period else False
        if done:
            portfolio_return = evaluate_portfolio_performance(agent)

if display:
    plot_all(stock_name, agent)

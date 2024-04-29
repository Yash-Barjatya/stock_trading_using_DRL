import argparse
import importlib
import sys
import time

from utils import *


parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_name', action="store",
                    dest="model_name", default='DQN', help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name",
                    default='^GSPC_2010-2015', help="stock name")
parser.add_argument('--window_size', action="store", dest="window_size",
                    default=10, type=int, help="span (days) of observation")
parser.add_argument('--num_episode', action="store",
                    dest="num_episode", default=10, type=int, help='episode number')
parser.add_argument('--initial_balance', action="store",
                    dest="initial_balance", default=50000, type=int, help='initial balance')
inputs = parser.parse_args()

model_name = inputs.model_name
stock_name = inputs.stock_name
window_size = inputs.window_size
num_episode = inputs.num_episode
initial_balance = inputs.initial_balance

stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1
returns_across_episodes = []
num_experience_replay = 0
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

# select learning model
model = importlib.import_module(f'agents.{model_name}')
agent = model.Agent(state_dim=window_size + 3, balance=initial_balance)


def hold(actions):
    # encourage selling for profit and liquidity
    next_probable_action = np.argsort(actions)[1]
    if next_probable_action == 2 and len(agent.inventory) > 0:
        max_profit = stock_prices[t] - min(agent.inventory)
        if max_profit > 0:
            sell(t)
            # reset this action's value to the highest
            actions[next_probable_action] = 1
            return 'Hold', actions


def buy(t):
    if agent.balance > stock_prices[t]:
        agent.balance -= stock_prices[t]
        agent.inventory.append(stock_prices[t])
        return 'Buy: ${:.2f}'.format(stock_prices[t])


def sell(t):
    if len(agent.inventory) > 0:
        agent.balance += stock_prices[t]
        bought_price = agent.inventory.pop(0)
        profit = stock_prices[t] - bought_price
        global reward
        reward = profit
        return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)


start_time = time.time()
for e in range(1, num_episode + 1):

    agent.reset()  # reset to initial balance and hyperparameters
    state = generate_combined_state(
        0, window_size, stock_prices, agent.balance, len(agent.inventory))

    for t in range(1, trading_period + 1):

        reward = 0
        next_state = generate_combined_state(
            t, window_size, stock_prices, agent.balance, len(agent.inventory))
        previous_portfolio_value = len(
            agent.inventory) * stock_prices[t] + agent.balance

        actions = agent.model.predict(state)[0]
        action = agent.act(state)

        # execute position
        if action == 0:  # hold
            execution_result = hold(actions)
        if action == 1:  # buy
            execution_result = buy(t)
        if action == 2:  # sell
            execution_result = sell(t)

        # check execution result
        if execution_result is None:
            reward -= treasury_bond_daily_return_rate() * agent.balance  # missing opportunity
        else:
            if isinstance(execution_result, tuple):  # if execution_result is 'Hold'
                actions = execution_result[1]
                execution_result = execution_result[0]

        # calculate reward
        current_portfolio_value = len(
            agent.inventory) * stock_prices[t] + agent.balance
        unrealized_profit = current_portfolio_value - agent.initial_portfolio_value
        reward += unrealized_profit

        agent.portfolio_values.append(current_portfolio_value)
        agent.return_rates.append(
            (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)

        done = True if t == trading_period else False
        agent.remember(state, actions, reward, next_state, done)

        # update state
        state = next_state

        # experience replay
        if len(agent.memory) > agent.buffer_size:
            num_experience_replay += 1
            agent.experience_replay()
        if done:
            portfolio_return = evaluate_portfolio_performance(agent)
            returns_across_episodes.append(portfolio_return)

    # save models periodically
    if e % 5 == 0:
        agent.model.save('saved_models/DQN_ep' + str(e) + '.h5')

print('total training time: {0:.2f} min'.format(
    (time.time() - start_time)/60))
plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)

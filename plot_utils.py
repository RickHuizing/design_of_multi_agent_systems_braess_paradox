import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_travel_times(dataframe: pd.DataFrame, model_name: str):
    """
    plot the average travel times per route per 1000 monte carlo sweeps
    """
    travel_times = None
    if dataframe.shape[1] == 4:
        # 4-link model
        travel_times = dataframe.iloc[:, 0:2]
    elif dataframe.shape[1] == 6:
        # 5-link model
        travel_times = dataframe.iloc[:, 0:3]
    elif dataframe.shape[1] == 7:
        # 5-link model with toll
        travel_times: pd.DataFrame = dataframe.iloc[:, 0:3]
        travel_times.insert(3, "toll", dataframe.iloc[:, 6])
        # raise NotImplementedError("toll not implemented")
    else:
        raise ValueError("dataframe must have 4 to 7 columns")

    travel_times = travel_times.rolling(1000).mean()
    travel_times = travel_times[::1000]

    avg_travel_times = travel_times.mean(axis=1)
    travel_times["avg travel time"] = avg_travel_times

    fig = plt.figure()
    g = sns.lineplot(data=travel_times)
    g.set(title="", ylabel="travel time", xlabel="ticks")
    g.set(ylim=(0, 1000))

    fig.savefig(f"{model_name}_travel_times.svg", format="svg")


def final_average_travel_times(histories):
    # assert one history per agent
    assert len(histories) == 248
    # assert each agent has a history of 30 steps
    for history in histories:
        assert len(history) == 30

    travel_times = {}
    for history in histories:
        for step in history:
            # step: [route, travel_time]
            if step[0] not in travel_times.keys():
                travel_times[step[0]] = []
            travel_times[step[0]].append(step[1])
    print("Total occurrences: ", {route: len(v) for route, v in travel_times.items()})
    print("Averages: ", {route: round(np.mean(v)) for route, v in travel_times.items()})

    return {route: round(np.mean(v)) for route, v in travel_times.items()}


def plot_throughput(dataframe: pd.DataFrame, model_name: str):
    """
    Plot the average throughput of the network per route per 1000 ticks
    """
    throughput: pd.DataFrame | None = None
    if dataframe.shape[1] == 4:
        # 4-link model
        throughput = dataframe.iloc[:, 2:4]
    elif dataframe.shape[1] == 6:
        # 5-link model
        throughput = dataframe.iloc[:, 3:6]
    elif dataframe.shape[1] == 7:
        # 5-link model with toll
        throughput = dataframe.iloc[:, 3:6]
        # raise NotImplementedError("toll not implemented")
    else:
        raise ValueError("dataframe must have 4 to 6 columns")

    # data is stored cumulatively, so use diff() to get the increments
    # (in our case 0 or 1 because of TASEP only letting one car through at a time)
    throughput = throughput.diff()

    # get total throughput per route per 1000 ticks
    # got this from https://stackoverflow.com/questions/47239332/take-the-sum-of-every-n-rows-in-a-pandas-series
    throughput = throughput.groupby(throughput.index // 1000).sum()

    # get average throughput per route per 1000 ticks, over 3000 ticks
    throughput = throughput.rolling(3).mean()

    # get total throughput of the network (all routes)
    throughput["total throughput"] = throughput.sum(axis=1)

    # drop the last row, as this is not a full 1000 ticks
    throughput = throughput[:-1]

    # drop the first 2 rows, as these are NaN because of the rolling mean
    throughput = throughput[2:]

    fig = plt.figure()
    g = sns.lineplot(data=throughput)
    g.set(title="", ylabel="throughput", xlabel="1000 ticks")
    g.set(ylim=(0, 550))

    fig.savefig(f"{model_name}_throughput.svg", format="svg")

    return throughput


def find_agent_strategy(histories):
    agent_strategies = {}
    for history in histories:
        agent_strategy = []
        for step in history:
            if step[0] not in agent_strategy:
                agent_strategy.append(step[0])
        agent_strategy = tuple(sorted(agent_strategy))
        if agent_strategy not in agent_strategies.keys():
            agent_strategies[agent_strategy] = 0
        agent_strategies[agent_strategy] += 1

    return agent_strategies


def interpret_agent_strategies(histories):
    print("we found that")
    for strat, count in find_agent_strategy(histories).items():
        print(f" - {count} agents took route {' and '.join(strat)}")
    print("in their last 30 passes through the network")


def create_latex_table(model_histories, model_names, toll=-1):
    model_routes = []
    route_code_to_names = {"R14": "Route 14", "R23": "Route 23", "R153": "Route 153"}
    agent_count = len(model_histories[0])
    for h in model_histories:
        assert (agent_count == len(h))

    for model_history in model_histories:
        routes = {}

        for agent_history in model_history:
            for route, time in agent_history:
                if route not in routes.keys():
                    routes[route] = []
                routes[route].append(time)
        model_routes.append(routes)

    column_widths = [10, 7, 7]
    # print dummy latex table header
    fake_header = f"{'':{column_widths[0]}}"
    for model_name in model_names:
        fake_header += f"&{model_name:^{sum(column_widths[1:3]) + 1}}"
    print(fake_header)

    # for each route, print a line in the latex table
    # for each model, print the mean route time and route share
    # table line: route_name & mean_route_time_1 & route_share1 | mean_route_time_2 & route_share2 | ... "
    for route_code, route_name in route_code_to_names.items():
        route_latex_table_line = f"{route_name:{column_widths[0]}}"

        for i, model_history in enumerate(model_histories):
            route_latex_table_line += "&"

            if route_code in model_routes[i].keys():
                route_times = model_routes[i][route_code]

                mean_route_time = round(np.mean(route_times))
                route_share = round(len(route_times) / (len(model_history) * 30), 3)
                route_latex_table_line += f"{mean_route_time:^{column_widths[1]}}&{route_share:^{column_widths[2]}}"
            else:
                route_latex_table_line += f" -  &   -   "

        route_latex_table_line += " \\\\"
        print(route_latex_table_line)

    # for each model, find total average travel time
    average_travel_times_latex_table_line = "Average Travel &&&&&&&&&&\\\nTime Per Agent"
    average_travel_times_readable_table_line = f"{'':{column_widths[0]}}"

    for i, model_history in enumerate(model_histories):
        all_travel_times = [time for agent_history in model_history for (route_code, time) in agent_history]
        mean_tt = round(np.mean([time for time in all_travel_times]), 1)
        average_travel_times_latex_table_line += f"& \\multicolumn{{2}}{{c|}}{{{mean_tt}}}"
        average_travel_times_readable_table_line += f"&{mean_tt:^{sum(column_widths[1:3]) + 1}}"
    average_travel_times_latex_table_line += " \\\\"
    # print(average_travel_times_latex_table_line)
    print(average_travel_times_readable_table_line)

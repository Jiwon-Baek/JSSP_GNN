import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def KPI_analysis(file_dir_list):
    for file_dir in zip(file_dir_list):
        df_results = pd.read_excel(file_dir + "summary.xlsx", index_col=0, engine='openpyxl')


def winning_rate(file_dir):
    df_results = pd.read_excel(file_dir + "summary.xlsx", index_col=0, engine='openpyxl')
    df_results = df_results.iloc[:-1]
    df_results = df_results[["RL-DG", "GP-1", "GP-2", "GP-3", "GP-4", "GP-5"]]

    max_columns = df_results.idxmax(axis=1)
    max_counts = max_columns.value_counts()

    print(max_counts)


def load_analysis(data_dir, res_dir):
    all_list = []

    for res_dir_temp in res_dir:
        if not os.path.exists(res_dir_temp):
            os.makedirs(res_dir_temp)

    for data_dir_temp, res_dir_temp in zip(data_dir, res_dir):
        file_paths = os.listdir(data_dir_temp)

        size = data_dir_temp.split('/')[-2]
        num_quays, num_ships = int(size.split('-')[0]), int(size.split('-')[1])

        index = ["P%d" % i for i in range(1, len(file_paths) + 1)]

        df_analysis = pd.DataFrame(index=index, columns=["max_load", "avg_load", "num_ops_ratio", "num_lng"])
        max_loads = []
        avg_loads = []
        num_ops_ratios = []
        num_lngs = []

        for prob, path in zip(index, file_paths):
            df_scenario = pd.read_excel(data_dir_temp + path, sheet_name="ship", engine='openpyxl')
            start = df_scenario["Start_Date"].min()
            finish = df_scenario["Finish_Date"].max()
            load = np.zeros(finish - start)

            for i, row in df_scenario.iterrows():
                load[int(row["Start_Date"]):int(row["Finish_Date"])] += 1

            max_load = np.max(load) / num_quays
            avg_load = np.mean(load) / num_quays
            num_ops_ratio = len(df_scenario) / (int(row["Finish_Date"]) - int(row["Start_Date"]))
            num_lng = len(df_scenario[df_scenario["Ship_Type"] == "LNG"].groupby(["Ship_Name"])) / num_ships

            max_loads.append(max_load)
            avg_loads.append(avg_load)
            num_ops_ratios.append(num_ops_ratio)
            num_lngs.append(num_lng)

        df_analysis["max_load"] = max_loads
        df_analysis["avg_load"] = avg_loads
        df_analysis["num_ops_ratio"] = num_ops_ratios
        df_analysis["num_lng"] = num_lngs

        df_results = pd.read_excel(res_dir_temp + "summary.xlsx", sheet_name="results", index_col=0, engine="openpyxl")
        df_results = df_results.iloc[:-1]
        df_results_aug = pd.concat([df_results, df_analysis], axis=1)

        all_list.append(df_results_aug)

    df_all = pd.concat(all_list, axis=0)
    df_all = df_all.reset_index(drop=True)

    columns_heuristics = ["SPT-MFQ", "SPT-LUQ", "SPT-HPQ","MOR-MFQ", "MOR-LUQ", "MOR-HPQ", "MWKR-MFQ", "MWKR-LUQ", "MWKR-HPQ"]
    columns_GP = ["GP-1", "GP-2", "GP-3", "GP-4", "GP-5"]
    columns_all = columns_heuristics + columns_GP
    df_all["BASELINE_H"] = df_all.apply(lambda row: row[columns_heuristics].min(), axis=1)
    df_all["BASELINE_G"] = df_all.apply(lambda row: row[columns_GP].min(), axis=1)
    df_all["BASELINE_A"] = df_all.apply(lambda row: row[columns_all].min(), axis=1)
    df_all["GAP_H"] = (df_all["BASELINE_H"] - df_all["RL-DG"]) / df_all["RL-DG"] * 100
    df_all["GAP_G"] = (df_all["BASELINE_G"] - df_all["RL-DG"]) / df_all["RL-DG"] * 100
    df_all["GAP_A"] = (df_all["BASELINE_A"] - df_all["RL-DG"]) / df_all["RL-DG"] * 100

    df_all_max_load = df_all.groupby(["max_load"]).mean()
    df_all_avg_load = df_all.groupby(["avg_load"]).mean()
    df_all_num_ops_ratio = df_all.groupby(["num_ops_ratio"]).mean()
    df_all_num_lng = df_all.groupby(["num_lng"]).mean()

    # plt.plot(df_all_max_load["max_load"], df_all_max_load["GAP"], 'ro')
    # plt.show()

    writer = pd.ExcelWriter('./analysis_simple.xlsx')
    df_all_max_load.to_excel(writer, sheet_name="max_load")
    df_all_avg_load.to_excel(writer, sheet_name="avg_load")
    df_all_num_ops_ratio.to_excel(writer, sheet_name="num_ops_ratio")
    df_all_num_lng.to_excel(writer, sheet_name="num_lng")
    writer.close()



def get_ideal_cost():
    data_dir = ["./input/test/28-100/", "./input/test/28-90/", "./input/test/28-80/", "./input/test/28-70/",
                "./input/test/28-60/",
                "./input/test/40-120/", "./input/test/40-100/", "./input/test/40-80/",
                "./input/test/35-100/", "./input/test/35-80/", "./input/test/35-60/",
                "./input/test/25-100/", "./input/test/25-80/", "./input/test/25-60/",
                "./input/test/20-80/", "./input/test/20-60/", "./input/test/20-40/"]

    res_dir = ["./output/test/28-100/", "./output/test/28-90/", "./output/test/28-80/", "./output/test/28-70/",
               "./output/test/28-60/",
               "./output/test/40-120/", "./output/test/40-100/", "./output/test/40-80/",
               "./output/test/35-100/", "./output/test/35-80/", "./output/test/35-60/",
               "./output/test/25-100/", "./output/test/25-80/", "./output/test/25-60/",
               "./output/test/20-80/", "./output/test/20-60/", "./output/test/20-40/"]

    for res_dir_temp in res_dir:
        if not os.path.exists(res_dir_temp):
            os.makedirs(res_dir_temp)

    for data_dir_temp, res_dir_temp in zip(data_dir, res_dir):
        file_paths = os.listdir(data_dir_temp)
        index = ["P%d" % i for i in range(1, len(file_paths) + 1)] + ["avg"]
        columns = ["Ideal"]
        costs = []

        df_ideal = pd.DataFrame(index=index, columns=columns)
        for prob, path in zip(index, file_paths):
            df_scenario = pd.read_excel(data_dir_temp + path, sheet_name="ship", engine='openpyxl')
            df_quay = df_scenario[(df_scenario["Operation_Type"] != "시운전") & (df_scenario["Operation_Type"] != "G/T")]
            df_sea = df_scenario[(df_scenario["Operation_Type"] == "시운전") | (df_scenario["Operation_Type"] == "G/T")]

            delay_cost = 0
            moving_cost = 4000 * (2 * len(df_sea) + 2)
            loss_cost = 12 * 5 * (df_quay["Duration"].sum())

            ideal_cost = delay_cost + moving_cost + loss_cost
            costs.append(ideal_cost)

        df_ideal["Ideal"] = costs + [sum(costs) / len(costs)]

        writer = pd.ExcelWriter(res_dir_temp + 'baseline.xlsx')
        df_ideal.to_excel(writer, sheet_name="ideal")
        writer.close()


def summary(file_dir):
    ideal_cost = pd.read_excel(file_dir + "baseline.xlsx", index_col=0, engine='openpyxl').squeeze()

    file_list = ["(GP) test_results.xlsx"]

    df_list = []
    for temp in file_list:
        delay_cost = pd.read_excel(file_dir + temp, sheet_name="delay_cost", index_col=0, engine="openpyxl")
        moving_cost = pd.read_excel(file_dir + temp, sheet_name="move_cost", index_col=0, engine="openpyxl")
        loss_cost = pd.read_excel(file_dir + temp, sheet_name="loss_cost", index_col=0, engine="openpyxl")
        total_cost = pd.DataFrame(0.0, columns=delay_cost.columns, index=delay_cost.index)

        total_cost = total_cost.add(delay_cost)
        total_cost = total_cost.add(moving_cost)
        total_cost = total_cost.add(loss_cost)

        if "RL" in temp:
            total_cost = total_cost.rename(columns={"RL": temp.split()[0][1:-1]})

        df_list.append(total_cost)

    df = pd.concat(df_list, axis=1)
    df = df.div(ideal_cost, axis="index")
    df = df.mul(100)

    df.loc["avg"] = df.apply(np.mean, axis=0)

    writer = pd.ExcelWriter(file_dir + '(GP) summary.xlsx')
    df.to_excel(writer, sheet_name="results")
    writer.close()


if __name__ == "__main__":
    # file_dir = ["./output/test/28-100/", "./output/test/28-90/", "./output/test/28-80/",
    #             "./output/test/28-70/", "./output/test/28-60/",
    #            "./output/test/40-120/", "./output/test/40-100/", "./output/test/40-80/",
    #            "./output/test/35-100/", "./output/test/35-80/", "./output/test/35-60/",
    #            "./output/test/25-100/", "./output/test/25-80/", "./output/test/25-60/",
    #            "./output/test/20-80/", "./output/test/20-60/", "./output/test/20-40/"]
    #
    # for temp in file_dir:
    #     summary(temp)

    # data_dir = ["./input/test/v1/28-100/",
    #             "./input/test/v1/28-90/",
    #             "./input/test/v1/28-80/",
    #             "./input/test/v1/28-70/",
    #             "./input/test/v1/28-60/"]
    # res_dir = ["./output/test/28-100/",
    #            "./output/test/28-90/",
    #            "./output/test/28-80/",
    #            "./output/test/28-70/",
    #            "./output/test/28-60/"]

    data_dir = ["./input/test/v1/28-80/"]
    res_dir = ["./output/test/28-80/"]

    df_all = load_analysis(data_dir, res_dir)

    # file_dir = ["./output/test/28-100/", "./output/test/28-90/", "./output/test/28-80/",
    #             "./output/test/28-70/", "./output/test/28-60/"]
    #
    # for temp in file_dir:
    #     winning_rate(temp)